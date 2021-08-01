from __future__ import division
import torch
import torch.optim as optim
import os
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import shutil
from collections import OrderedDict

from models.shape_decoder import ShapeDecoder, ShapeDecoderCurriculum
from models.pose_decoder import PoseDecoder, PoseDecoderSE3
import utils.deepsdf_utils as deepsdf_utils 
import utils.nnutils as nnutils

import config as cfg


# For DataParallel: https://discuss.pytorch.org/t/how-could-i-train-on-multi-gpu-and-infer-with-single-gpu/22838/6


class Trainer():

    def __init__(
        self, 
        debug,
        device, 
        train_dataset,
        exp_dir, exp_name,
        train_to_augmented
    ):
        self.do_code_regularization = cfg.do_code_regularization
        self.shape_reg_lambda = cfg.shape_reg_lambda
        self.pose_reg_lambda = cfg.pose_reg_lambda
        
        self.debug = debug

        self.init_from     = cfg.init_from
        self.continue_from = cfg.continue_from

        self.train_to_augmented = train_to_augmented

        ###############################################################################################
        # Model
        ###############################################################################################

        # Code dim
        self.shape_codes_dim = cfg.shape_codes_dim
        self.pose_codes_dim = cfg.pose_codes_dim

        self.num_identities = train_dataset.get_num_identities()
        self.num_train_samples = len(train_dataset)

        # SHAPE decoder
        if cfg.use_curriculum:
            self.shape_decoder = ShapeDecoderCurriculum(
                self.shape_codes_dim, cfg.curriculum_dist, cfg.epochs, **cfg.shape_network_specs
            ).to(device)
        else:
            self.shape_decoder = ShapeDecoder(self.shape_codes_dim, **cfg.shape_network_specs).to(device)
        
        # POSE decoder (note that we input both shape and pose codes, not only pose codes)
        if cfg.use_se3:
            self.pose_decoder = PoseDecoderSE3(self.shape_codes_dim + self.pose_codes_dim, **cfg.pose_network_specs).to(device)
        else:
            self.pose_decoder = PoseDecoder(self.shape_codes_dim + self.pose_codes_dim, **cfg.pose_network_specs).to(device)

        # Freeze if necessary
        if cfg.freeze_shape_decoder:
            for p in self.shape_decoder.parameters():
                p.requires_grad = False

        if cfg.freeze_pose_decoder:
            for p in self.pose_decoder.parameters():
                p.requires_grad = False

        # CODES
        self.shape_codes = torch.ones(self.num_identities, 1, self.shape_codes_dim).normal_(0, 1.0 / self.shape_codes_dim).to(device)
        self.shape_codes.requires_grad = True

        self.pose_codes = torch.ones(self.num_train_samples, 1, self.pose_codes_dim).normal_(0, 1.0 / self.pose_codes_dim).to(device)
        self.pose_codes.requires_grad = True

        print()
        print("Initialized SHAPE codes with mean magnitude {} and shape {}".format(
            deepsdf_utils.get_mean_latent_code_magnitude(self.shape_codes), self.shape_codes.shape)
        )
        print("Initialized POSE  codes with mean magnitude {} and shape {}".format(
            deepsdf_utils.get_mean_latent_code_magnitude(self.pose_codes), self.pose_codes.shape)
        )

        print()
        
        n_all_shape_params = int(sum([np.prod(p.size()) for p in self.shape_decoder.parameters()]))
        n_trainable_shape_params = int(sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, self.shape_decoder.parameters())]))
        print("Number of parameters in shape_decoder:      {0} / {1}".format(n_trainable_shape_params, n_all_shape_params))

        n_all_pose_params = int(sum([np.prod(p.size()) for p in self.pose_decoder.parameters()]))
        n_trainable_pose_params = int(sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, self.pose_decoder.parameters())]))
        print("Number of parameters in pose_decoder:       {0} / {1}".format(n_trainable_pose_params, n_all_pose_params))
        print()

        ###############################################################################################
        ###############################################################################################

        self.device = device

        #####################################################################################
        # Set up optimizer.
        #####################################################################################
        lr_schedule_shape_decoder = nnutils.StepLearningRateSchedule(cfg.learning_rate_schedules['shape_decoder'])
        lr_schedule_pose_decoder  = nnutils.StepLearningRateSchedule(cfg.learning_rate_schedules['pose_decoder'])
        lr_schedule_shape_codes   = nnutils.StepLearningRateSchedule(cfg.learning_rate_schedules['shape_codes'])
        lr_schedule_pose_codes    = nnutils.StepLearningRateSchedule(cfg.learning_rate_schedules['pose_codes'])
        
        self.lr_schedules = [
            lr_schedule_shape_decoder,
            lr_schedule_pose_decoder,
            lr_schedule_shape_codes,
            lr_schedule_pose_codes,
        ]

        learnable_params = [
            {
                "params": self.shape_decoder.parameters(),
                "lr": lr_schedule_shape_decoder.get_learning_rate(0),
            },
            {
                "params": self.pose_decoder.parameters(),
                "lr": lr_schedule_pose_decoder.get_learning_rate(0),
            },
            {
                "params": self.shape_codes,
                "lr": lr_schedule_shape_codes.get_learning_rate(0),
            },
            {
                "params": self.pose_codes,
                "lr": lr_schedule_pose_codes.get_learning_rate(0),
            },
        ]

        if cfg.optimizer == 'Adam':
            self.optimizer = optim.Adam(learnable_params)
        else:
            raise Exception("Others optimizers are not available yet.")

        self.train_dataset  = train_dataset

        self.num_points_sdf = self.train_dataset.get_num_samples_per_shape()['sdf']
        print("Num samples sdf:", self.num_points_sdf)
        
        self.num_points_flow = self.train_dataset.get_num_samples_per_shape()['flow']
        print("Num samples boundary:", self.num_points_flow)

        self.min_vec = torch.ones(self.num_points_sdf * cfg.batch_size, 1).cuda() * (-cfg.clamping_distance)
        self.max_vec = torch.ones(self.num_points_sdf * cfg.batch_size, 1).cuda() * cfg.clamping_distance
        
        print()
        
        self.batch_size = cfg.batch_size

        self.eval_every = cfg.eval_every

        #####################################################################################
        # Create dirs
        #####################################################################################
        self.exp_dir = exp_dir
        self.exp_name = exp_name

        if not self.debug:
            if not cfg.init_from and cfg.continue_from:
                self.exp_path = os.path.join(exp_dir, cfg.continue_from)
                self.checkpoints_dir = os.path.join(self.exp_path, 'checkpoints')
            else:
                self.exp_path = os.path.join(exp_dir, exp_name)
                self.checkpoints_dir = os.path.join(self.exp_path, 'checkpoints')
                if not os.path.exists(self.checkpoints_dir):
                    os.makedirs(self.checkpoints_dir)

                # Copy config.py over
                config_path = os.path.join("configs_train", f"config_train_{cfg.config_dataset}.py")
                shutil.copy(config_path, os.path.join(self.exp_path, f"config_{cfg.config_dataset}.py"))
            
            self.writer = SummaryWriter(os.path.join(self.exp_path, 'summary'))

        self.lambdas = cfg.lambdas_sdf
        assert self.lambdas['ref'] > 0 or self.lambdas['flow'] > 0, "What do you expect to train?"

        self.criterion_l1 = deepsdf_utils.SoftL1()

        if cfg.use_curriculum:
            self.curriculum_epochs = [int(ce * cfg.epochs) for ce in cfg.curriculum_dist]
            print()
            print("Using curriculum with", self.curriculum_epochs)
            print()

    def train_step(self, batch, epoch):

        self.shape_decoder.train()
        self.pose_decoder.train()

        #####################################################################################
        # Set gradients to None
        #####################################################################################
        # Decoders
        for param in self.shape_decoder.parameters():
            param.grad = None
        for param in self.pose_decoder.parameters():
            param.grad = None
        
        # Latent codes
        self.shape_codes.grad = None
        self.pose_codes.grad = None
        #####################################################################################

        loss, loss_dict = self.compute_loss(batch, epoch)

        loss.backward()

        self.optimizer.step()

        # Project latent vectors onto sphere
        if cfg.code_bound is not None:
            deepsdf_utils.project_latent_codes_onto_sphere(self.shape_codes, cfg.code_bound)
            deepsdf_utils.project_latent_codes_onto_sphere(self.pose_codes, cfg.code_bound)

        return loss_dict

    def compute_loss(self, batch, epoch):
        device = self.device

        ################################################################
        # Get data
        ################################################################

        ref = batch.get('ref')
        cur = batch.get('cur')
        indices = batch.get('idx')
        identity_ids = batch.get('identity_id')

        #############################################################################
        # SDF points
        #############################################################################
        p_sdf_ref = ref['points_sdf'].to(device) # [bs, N, 3]
        
        batch_size = p_sdf_ref.shape[0]

        sdf_data = p_sdf_ref.reshape(self.num_points_sdf * batch_size, 4)
        p_sdf  = sdf_data[:, 0:3]
        sdf_gt = sdf_data[:, 3].unsqueeze(1)

        #############################################################################
        # Points for learning flow
        #############################################################################
        p_flow_ref = ref['points_flow'].to(device) # [bs, N, 3]
        p_flow_cur = cur['points_flow'].to(device) # [bs, N, 3]

        p_flow_ref_flat = p_flow_ref.reshape(-1, 3) # [bs*N, 3]
        
        #############################################################################
        # Initialize losses
        #############################################################################
        loss = torch.tensor((0.0)).to(device)
        
        loss_ref_item  = 0.0
        loss_flow_item = 0.0

        ##########################################################################################
        ##########################################################################################
        # Forward pass
        ##########################################################################################
        ##########################################################################################

        # Get shape codes for batch samples
        assert torch.all(identity_ids < self.shape_codes.shape[0]), f"{identity_ids} vs {self.shape_codes.shape[0]}"
        shape_codes_batch = self.shape_codes[identity_ids, ...] # [bs, 1, C]

        ##########################################################################################
        # A) Reconstruct shape in reference frame
        ##########################################################################################
        if self.lambdas['ref'] > 0:
            # Extent latent code to all sampled points
            shape_codes_repeat = shape_codes_batch.expand(-1, self.num_points_sdf, -1) # [bs, N, C]
            shape_codes_inputs = shape_codes_repeat.reshape(-1, self.shape_codes_dim) # [bs*N, C]
            
            shape_inputs = torch.cat([shape_codes_inputs, p_sdf], 1)

            # Truncate groundtruth sdf
            if cfg.enforce_minmax:
                sdf_gt = deepsdf_utils.threshold_min_max(sdf_gt, self.min_vec, self.max_vec)

            pred_sdf = self.shape_decoder(shape_inputs, epoch=epoch)  # [bs*N, 1]

            # Truncate predicted sdf
            if cfg.enforce_minmax:
                pred_sdf = deepsdf_utils.threshold_min_max(pred_sdf, self.min_vec, self.max_vec)

            loss_ref, l1_ref_raw = self.criterion_l1(pred_sdf, sdf_gt, self.eps)
            loss_ref = loss_ref * (1.0 + self.lamb * torch.sign(sdf_gt) * torch.sign(sdf_gt - pred_sdf))
            loss_ref = torch.mean(loss_ref)
            
            loss += self.lambdas['ref'] * loss_ref

            loss_ref_item = l1_ref_raw.item()

        ##########################################################################################
        # B) Flow
        ##########################################################################################
        if self.lambdas['flow'] > 0.0 :

            # Extent latent shape codes to all sampled points
            shape_codes_repeat = shape_codes_batch.expand(-1, self.num_points_flow, -1) # [bs, N, C]
            shape_codes_inputs = shape_codes_repeat.reshape(-1, self.shape_codes_dim) # [bs*N, C]
            
            # Get codes for batch samples
            assert torch.all(indices < self.pose_codes.shape[0]), f"{indices} vs {self.pose_codes.shape[0]}"
            pose_codes_batch = self.pose_codes[indices, ...] # [bs, 1, C]

            # Extent latent code to all sampled points
            pose_codes_repeat = pose_codes_batch.expand(-1, self.num_points_flow, -1) # [bs, N, C]
            pose_codes_inputs = pose_codes_repeat.reshape(-1, self.pose_codes_dim) # [bs*N, C]

            # Concatenate pose and shape codes
            shape_pose_codes_inputs = torch.cat([shape_codes_inputs, pose_codes_inputs], 1)

            # Concatenate (for each sample point), the corresponding code and the p_cur coords
            pose_inputs = torch.cat([shape_pose_codes_inputs, p_flow_ref_flat], 1)

            # Predict ref warped into cur
            p_flow_ref_warped_flat, _ = self.pose_decoder(pose_inputs, self.current_epoch) # [bs*N, 3]
            # Reshape
            p_flow_ref_warped_flat = p_flow_ref_warped_flat.reshape(batch_size, -1, 3)
           
            assert p_flow_cur.shape == p_flow_ref_warped_flat.shape, f"{p_flow_cur.shape} vs {p_flow_ref_warped_flat.shape}"
            assert p_flow_cur.shape[-1] == 3

            loss_flow = (p_flow_cur - p_flow_ref_warped_flat) * (p_flow_cur - p_flow_ref_warped_flat)

            ############# viz ##################
            ############# viz ##################
            # if self.current_epoch % 1000 == 0:
            #     import open3d as o3d
                
            #     if not predict_nocs_directly:
            #         p_cur_warped_to_ref = p_cur[:, :self.num_flow_samples, :] + delta_cur_to_ref_flow
            #         p_ref_flow = p_ref[:, :self.num_flow_samples, :]
            #         p_cur_flow = p_cur[:, :self.num_flow_samples, :]

            #     # p_cur_test = p_cur_warped_to_ref[0, :self.num_samples_per_sigma, :].detach().cpu().numpy()
            #     p_cur_test = p_cur_flow[0, :self.num_samples_per_sigma, :].detach().cpu().numpy()
            #     p_cur_test_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p_cur_test))
            #     p_cur_test_pcd.paint_uniform_color([0, 1, 0]) # blue are source
                
            #     p_ref_test = p_ref_flow[0, :self.num_samples_per_sigma, :].detach().cpu().numpy()
            #     p_ref_test_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p_ref_test))
            #     p_ref_test_pcd.paint_uniform_color([1, 0, 0]) # blue are source

            #     correspondences = [(i, i) for i in range(0, p_ref_test.shape[0])]
            #     lines = o3d.geometry.LineSet.create_from_point_cloud_correspondences(p_ref_test_pcd, p_cur_test_pcd, correspondences)
            #     lines.paint_uniform_color([0.5, 0.2, 0.7])

            #     o3d.visualization.draw_geometries([lines, p_ref_test_pcd, p_cur_test_pcd])
            #     o3d.visualization.draw_geometries([p_cur_test_pcd])
            #     o3d.visualization.draw_geometries([p_ref_test_pcd, p_cur_test_pcd])
            ############# viz ##################
            ############# viz ##################

            loss_flow = torch.mean(
                torch.sum(loss_flow, dim=-1) / 2.0
            )

            loss += self.lambdas['flow'] * loss_flow

            loss_flow_item = loss_flow.item()
            
        ##########################################################################################
        # L2 regularization
        ##########################################################################################
        if self.do_code_regularization:
            
            if self.lambdas['ref'] > 0 and self.shape_codes.requires_grad:
                # Shape codes reg            
                l2_shape_loss = deepsdf_utils.latent_size_regul(self.shape_codes, identity_ids.numpy())
                l2_shape_reg = self.shape_reg_lambda * min(1, epoch / 100) * l2_shape_loss
                loss += l2_shape_reg

            if self.lambdas['flow'] > 0.0 and self.pose_codes.requires_grad:
                # Pose codes reg            
                l2_pose_loss = deepsdf_utils.latent_size_regul_no_index(pose_codes_batch)
                l2_pose_reg = self.pose_reg_lambda * min(1, epoch / 100) * l2_pose_loss
                loss += l2_pose_reg

        # Prepare dict of losses
        loss_dict = {
            'total': loss.item(),
            'loss_ref': loss_ref_item, 
            'loss_flow': loss_flow_item, 
        }

        return loss, loss_dict

    def train_model(self):
        start = 0

        if self.init_from:
            print()
            print("Initializing from", self.init_from)
            self.load_from_other()

        if not self.debug:
            if not self.init_from:
                print()
                print("Continuing from", self.continue_from)
                start = self.load_latest_checkpoint()

        # Multi-GPU
        if torch.cuda.device_count() > 1:
            print()
            print(f"Using {torch.cuda.device_count()} GPUs")
            print()
            self.shape_decoder = torch.nn.DataParallel(self.shape_decoder)
            self.pose_decoder = torch.nn.DataParallel(self.pose_decoder)

        for epoch in range(start, cfg.epochs + cfg.epochs_extra):
            ############################################################
            # Assert models
            nnutils.assert_models(
                self.shape_decoder, cfg.freeze_shape_decoder,
                self.pose_decoder,  cfg.freeze_pose_decoder,
                self.shape_codes,   cfg.freeze_shape_codes,
                self.pose_codes,    cfg.freeze_pose_codes,
            )
            ############################################################
            
            sum_loss_total = 0
            sum_loss_ref   = 0
            sum_loss_flow  = 0

            self.current_epoch = epoch

            ############################################################################
            # Curriculum DeepSDF hyperparms
            ############################################################################
            if cfg.use_curriculum:
                if epoch < self.curriculum_epochs[0]:
                    self.eps = 2.5e-2 
                    self.lamb = 0
                elif epoch < self.curriculum_epochs[1]:
                    self.eps = 1e-2 
                    self.lamb = 0.1
                elif epoch < self.curriculum_epochs[2]:
                    self.eps = 2.5e-3 
                    self.lamb = 0.2
                else:
                    self.eps = 0
                    self.lamb = 0.5
            else:
                self.eps = 0.0
                self.lamb = 0.0
            ############################################################################
            
            print()
            print(f'Epoch {epoch} - {self.exp_name}')
            
            train_data_loader = self.train_dataset.get_loader(shuffle=True)

            nnutils.adjust_learning_rate(self.lr_schedules, self.optimizer, epoch)

            ############################################################
            # Store checkpoint
            ############################################################
            if epoch % self.eval_every == 0:

                # Store latest checkpoint
                if not self.debug:
                    # Store latest checkpoint
                    self.save_special_checkpoint(epoch, "latest")
                    
                    # Store a checkpoint every 10 epochs
                    self.save_checkpoint(epoch)

            ############################################################
            # Assert models
            nnutils.assert_models(
                self.shape_decoder, cfg.freeze_shape_decoder,
                self.pose_decoder,  cfg.freeze_pose_decoder,
                self.shape_codes,   cfg.freeze_shape_codes,
                self.pose_codes,    cfg.freeze_pose_codes,
            )
            ############################################################

            ############################################################
            # TRAIN
            ############################################################
            num_samples_in_batch = len(train_data_loader)

            for batch in tqdm(train_data_loader):
                
                loss_dict = self.train_step(batch, epoch)

                loss_total = loss_dict['total']
                loss_ref   = loss_dict['loss_ref']
                loss_flow  = loss_dict['loss_flow']
                
                sum_loss_total += loss_total
                sum_loss_ref   += loss_ref
                sum_loss_flow  += loss_flow

            sum_loss_total = sum_loss_total / num_samples_in_batch
            sum_loss_ref   = sum_loss_ref / num_samples_in_batch
            sum_loss_flow  = sum_loss_flow / num_samples_in_batch

            if not self.debug and epoch % self.eval_every == 0:
                self.writer.add_scalar('train/0_loss', sum_loss_total, epoch)       
                self.writer.add_scalar('train/1_ref',  sum_loss_ref  , epoch)        
                self.writer.add_scalar('train/2_flow', sum_loss_flow , epoch)

            print(
                "Current loss: {:.4f} - "
                "ref: {:.4f} ({:.4f}) - flow: {:.4f} ({:.4f})".format(
                    sum_loss_total,
                    sum_loss_ref  , self.lambdas['ref'] * sum_loss_ref,
                    sum_loss_flow , self.lambdas['flow'] * sum_loss_flow,
                )
            )


    def save_base(self, path, epoch):

        if isinstance(self.shape_decoder, torch.nn.DataParallel):
            shape_decoder_state_dict = self.shape_decoder.module.state_dict()
            pose_decoder_state_dict  = self.pose_decoder.module.state_dict()
        else:
            shape_decoder_state_dict = self.shape_decoder.state_dict()
            pose_decoder_state_dict  = self.pose_decoder.state_dict()

        torch.save(
            {
                'epoch':epoch,
                'model_state_dict_shape_decoder': shape_decoder_state_dict,
                'model_state_dict_pose_decoder': pose_decoder_state_dict,
                'shape_codes': self.shape_codes.detach().cpu(),
                'pose_codes': self.pose_codes.detach().cpu(),
                'optimizer_state_dict': self.optimizer.state_dict()
            }, 
            path
        )


    def save_checkpoint(self, epoch):
        path = os.path.join(self.checkpoints_dir, f'checkpoint_epoch_{epoch}.tar')
        if not os.path.exists(path):
            self.save_base(path, epoch)


    def save_special_checkpoint(self, epoch, special_name):
        path = os.path.join(self.checkpoints_dir, f'{special_name}_checkpoint.tar')
        self.save_base(path, epoch)


    def load_latest_checkpoint(self):

        checkpoints = [m for m in os.listdir(self.checkpoints_dir)]

        if len(checkpoints) == 0:
            print()
            print('No checkpoints found at {}'.format(self.checkpoints_dir))
            return 0

        # If we're here, we have at least 1 checkpoint
        latest_checkpoint_path = os.path.join(self.checkpoints_dir, "latest_checkpoint.tar")

        if not os.path.exists(latest_checkpoint_path):
            raise Exception(f'Latest checkpoint {latest_checkpoint_path} does not exist!')

        checkpoint = torch.load(latest_checkpoint_path)

        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.shape_decoder.load_state_dict(checkpoint['model_state_dict_shape_decoder'])
        self.pose_decoder.load_state_dict(checkpoint['model_state_dict_pose_decoder'])

        assert checkpoint['shape_codes'].shape == self.shape_codes.shape
        self.shape_codes = checkpoint['shape_codes'].to(self.device).detach().clone()
        self.shape_codes.requires_grad = False if cfg.freeze_shape_codes else True

        assert checkpoint['pose_codes'].shape == self.pose_codes.shape
        self.pose_codes = checkpoint['pose_codes'].to(self.device).detach().clone()
        self.pose_codes.requires_grad = False if cfg.freeze_pose_codes else True

        epoch = checkpoint['epoch']

        print("Loaded epoch", epoch)
        print("Optim shape code", self.shape_codes.requires_grad)
        print("Optim pose code", self.pose_codes.requires_grad)
        
        print()
        print('Loaded checkpoint from: {}'.format(latest_checkpoint_path))

        print()
        print("Looking good???")
        print()
        
        return epoch

    def load_checkpoint(self, exp_name, checkpoint):
        checkpoint_dir = os.path.join(self.exp_dir, exp_name, "checkpoints")

        if isinstance(checkpoint, int): 
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{checkpoint}.tar")
        else:
            checkpoint_path = os.path.join(checkpoint_dir, f"{checkpoint}_checkpoint.tar")

        if not os.path.exists(checkpoint_path):
            raise Exception(f'Other checkpoint {checkpoint_path} does not exist!')

        return torch.load(checkpoint_path)


    def get_new_state_dict(state_dict):
        new_state_dict = OrderedDict()

        for k, v in state_dict.items():
            if 'module' not in k:
                k = 'module.'+k
            else:
                k = k.replace('features.module.', 'module.features.')
            new_state_dict[k]=v

        return new_state_dict
    

    def load_from_other(self):

        checkpoint = self.load_checkpoint(cfg.init_from, cfg.checkpoint)
        checkpoint_pose  = checkpoint
        checkpoint_shape = checkpoint

        if cfg.init_from_pose:
            print("hi")
            checkpoint_pose = self.load_checkpoint(cfg.init_from_pose, cfg.checkpoint_pose)

        #######################################################################################################################
        # SHAPE DECODER
        #######################################################################################################################
        if cfg.load_shape_decoder:
            print("Loaded SHAPE decoder")
            self.shape_decoder.load_state_dict(checkpoint_shape['model_state_dict_shape_decoder'])

        #######################################################################################################################
        # POSE DECODER
        #######################################################################################################################
        if cfg.load_pose_decoder:
            print("Loaded POSE  decoder")
            self.pose_decoder.load_state_dict(checkpoint_pose['model_state_dict_pose_decoder'])

        #######################################################################################################################
        # SHAPE CODES
        #######################################################################################################################
        if cfg.load_shape_codes:
            print("Loaded SHAPE codes")
            pretrained_shape_codes = checkpoint_shape['shape_codes'].to(self.device).detach().clone()
            num_pretrained_shape_codes = pretrained_shape_codes.shape[0]

            # Initialize from other shape codes if we have the correct mapping
            if self.shape_codes.shape[0] != num_pretrained_shape_codes and self.train_to_augmented is not None:
                self.shape_codes = pretrained_shape_codes[list(self.train_to_augmented.values())]
                # Make sure we have the first dimension expanded
                if len(self.shape_codes.shape) == 2:
                    self.shape_codes = self.shape_codes.unsqueeze(0)
            else:
                assert self.shape_codes.shape[0] == num_pretrained_shape_codes, f"{self.shape_codes.shape[0]} != {num_pretrained_shape_codes}"
                self.shape_codes = pretrained_shape_codes

            self.shape_codes.requires_grad = False if cfg.freeze_shape_codes else True

        #######################################################################################################################
        # POSE CODES
        #######################################################################################################################
        if cfg.load_pose_codes:
            print("Loaded POSE  codes")
            pretrained_pose_codes = checkpoint_pose['pose_codes'].to(self.device).detach().clone()
            num_pretrained_pose_codes = pretrained_pose_codes.shape[0]
            
            assert self.pose_codes.shape[0] == num_pretrained_pose_codes, f"{self.pose_codes.shape[0]} != {num_pretrained_pose_codes}"
            
            self.pose_codes = pretrained_pose_codes
            self.pose_codes.requires_grad = False if cfg.freeze_pose_codes else True

        print()
        print('Loaded checkpoint from:      {}'.format(cfg.init_from))
        print('Loaded pose checkpoint from: {}'.format(cfg.init_from_pose))