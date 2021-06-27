import sys,os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import os
import numpy as np
import open3d as o3d
import torch
from torch.nn import functional as F
from tqdm import tqdm
import argparse
from torch.utils.tensorboard import SummaryWriter
from glob import glob
import shutil

import models.inference_encoder as encoders
from models.pose_decoder import PoseDecoder, PoseDecoderSE3

import datasets.voxels_dataset as data_sdflow

from utils.pcd_utils import (BBox,
                            rotate_around_axis)

import utils.nnutils as nnutils
from data_scripts.prepare_labels_pose_encoder import compute_labels_pose_encoder_training

import config as cfg


class Viewer:
    def __init__(
        self,
        split,
        labels,
        num_to_eval=-1,
        load_every=None,
        from_frame_id=0,
        res=64
    ):
        self.time = 0

        self.split = split

        self.ref  = None
        self.cur  = None
        self.refw  = None

        self.show_ref  = False
        self.show_cur  = False
        self.show_refw = False

        self.labels = labels
        self.num_to_eval = num_to_eval
        self.load_every = load_every
        self.from_frame_id = from_frame_id
        
        self.res = res

        self.initialize()
    
    def initialize(self):

        self.ref_cached = None
        self.cur_list  = []
        self.refw_list = []

        self.epes = []

        self.loaded_frames = 0

        # Go over sequence and compute losses 
        for frame_i in tqdm(range(len(self.labels))):

            if self.load_every and frame_i % self.load_every != 0:
                continue

            if frame_i < self.from_frame_id:
                continue
            
            label = self.labels[frame_i]

            dataset, identity_id, identity_name, animation_name, sample_id = \
                label['dataset'], label['identity_id'], label['identity_name'], label['animation_name'], label['sample_id']

            ref_path = os.path.join(data_dir, dataset, identity_name, "a_t_pose", "000000")
            cur_path = os.path.join(data_dir, dataset, identity_name, animation_name, sample_id)

            # Ref
            ref_sample_path = os.path.join(ref_path, f'flow_samples_{test_sigma[0]}.npz')
            ref_samples_npz = np.load(ref_sample_path)
            p_ref = ref_samples_npz['points'].astype(np.float32)
            p_ref_cuda = torch.from_numpy(p_ref)[None, :].cuda()
                
            # Cur
            cur_sample_path = os.path.join(cur_path, f'flow_samples_{test_sigma[0]}.npz')
            cur_samples_npz = np.load(cur_sample_path)
            p_cur = cur_samples_npz['points'].astype(np.float32)
            p_cur_cuda = torch.from_numpy(p_cur)[None, :].cuda()

            ########################################################################
            # Inference for points REF 2 CUR
            ########################################################################

            points    = p_ref_cuda    # [1, 100000, 3]
            gt_points = p_cur_cuda # [1, 100000, 3]

            points_flat = points.reshape(-1, 3) # [100000, 3]
            gt_points_flat = gt_points.reshape(-1, 3) # [100000, 3]

            with torch.no_grad():

                ##########################################################################################
                # Prepare shape codes
                shape_codes_batch = trainer.shape_codes[[identity_id], :] # [bs, 1, C]

                assert shape_codes_batch.shape == (1, 1, shape_codes_dim), f"{shape_codes_batch.shape[0]} vs {(1, 1, shape_codes_dim)}"

                # Extent latent code to all sampled points
                shape_codes_repeat = shape_codes_batch.expand(-1, points_flat.shape[0], -1) # [bs, N, C]
                shape_codes_inputs = shape_codes_repeat.reshape(-1, shape_codes_dim) # [bs*N, C]
                ##########################################################################################
                
                ##########################################################################################
                # Prepare pose codes
                if self.split == "train":
                    pose_codes_batch = trainer.pose_codes_train[[frame_i], ...] # [bs, 1, C]
                elif self.split == "val":
                    pose_codes_batch = trainer.pose_codes_val[[frame_i], ...] # [bs, 1, C]
                else:
                    raise Exception(f"Split {self.split} is not defined")

                assert pose_codes_batch.shape == (1, 1, pose_codes_dim), f"{pose_codes_batch.shape[0]} vs {(1, 1, pose_codes_dim)}"

                # Extent latent code to all sampled points
                pose_codes_repeat = pose_codes_batch.expand(-1, points_flat.shape[0], -1) # [bs, N, C]
                pose_codes_inputs = pose_codes_repeat.reshape(-1, pose_codes_dim) # [bs*N, C]
                ##########################################################################################

                # Concatenate pose and shape codes
                shape_pose_codes_inputs = torch.cat([shape_codes_inputs, pose_codes_inputs], 1)
                
                # Concatenate (for each sample point), the corresponding code and the p_cur coords
                pose_inputs = torch.cat([shape_pose_codes_inputs, points_flat], 1)

                # Predict delta flow
                p_ref_warped_to_ref, _ = pose_decoder(pose_inputs) # [bs*N, 3]
                
            assert p_ref_warped_to_ref.shape == gt_points_flat.shape
            assert gt_points_flat.shape[-1] == 3

            # loss_flow = torch.mean(torch.sum((gt_points_flat - p_ref_warped_to_ref) * (gt_points_flat - p_ref_warped_to_ref), dim=-1) / 2.0)
            # loss_flow = loss_flow.item()
            # print(f"flow {0}: {loss_flow}")
            
            # Compute EPE
            current_epe = gt_points_flat - p_ref_warped_to_ref
            current_epe = torch.mean(torch.norm(current_epe, dim=-1)).item()
            # print(f"epe  {0}:", current_epe)
            self.epes.append(current_epe)

            # Cache point clouds
            if self.ref_cached is None:
                points_flat = points_flat.detach().cpu().numpy()
                p_ref_pcd   = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_flat))
                p_ref_pcd.paint_uniform_color([1, 0, 0]) # ref is in red
                p_ref_pcd.estimate_normals()
                self.ref_cached = p_ref_pcd
            
            gt_points_flat = gt_points_flat.detach().cpu().numpy()
            p_cur_pcd      = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gt_points_flat))
            p_cur_pcd.paint_uniform_color([0, 1, 0]) # current is in green
            p_cur_pcd.estimate_normals()
            self.cur_list.append(p_cur_pcd)
            
            p_ref_warped_to_ref = p_ref_warped_to_ref.detach().cpu().numpy()
            p_ref_warped_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p_ref_warped_to_ref))
            p_ref_warped_pcd.paint_uniform_color([0, 0, 1]) # ref warped to current is in blue
            p_ref_warped_pcd.estimate_normals()
            self.refw_list.append(p_ref_warped_pcd)

            # Increase counter of evaluated frames
            self.loaded_frames += 1

            print(f'Loaded {self.loaded_frames} frames')

            if self.loaded_frames == self.num_to_eval:
                print()
                print(f"Stopping early. Already loaded {self.loaded_frames}")
                print()
                break


        print()
        print("#####################################################################")
        print("Mean EPE:", sum(self.epes) / len(self.epes))
        print("Num cur", len(self.cur_list))
        print("Num refw", len(self.refw_list))
        print("#####################################################################")
        print()

        ###############################################################################################
        # Generate additional meshes.
        ###############################################################################################
        # unit bbox
        p_min = -0.5
        p_max =  0.5
        self.unit_bbox = BBox.compute_bbox_from_min_point_and_max_point(
            np.array([p_min]*3), np.array([p_max]*3)
        )
        self.unit_bbox = rotate_around_axis(self.unit_bbox, axis_name="x", angle=-np.pi) 

    def update_ref(self, vis):
        param = vis.get_view_control().convert_to_pinhole_camera_parameters()

        # We remove a mesh if it's currently stored.
        if self.ref is not None:
            vis.remove_geometry(self.ref)
            self.ref = None

        # If requested, we show a (new) mesh.
        if self.show_ref:
            self.ref = self.ref_cached
            vis.add_geometry(self.ref)

        ctr = vis.get_view_control()
        ctr.convert_from_pinhole_camera_parameters(param)

    def update_cur(self, vis):
        param = vis.get_view_control().convert_to_pinhole_camera_parameters()

        if self.cur is not None:
            vis.remove_geometry(self.cur)
        
        if self.show_cur:
            self.cur = self.cur_list[self.time]
            vis.add_geometry(self.cur)

        # print(f"EPE - {self.time}: {self.epes[self.time]}")

        ctr = vis.get_view_control()
        ctr.convert_from_pinhole_camera_parameters(param)

    def update_refw(self, vis):
        param = vis.get_view_control().convert_to_pinhole_camera_parameters()

        if self.refw is not None:
            vis.remove_geometry(self.refw)
        
        if self.show_refw:
            self.refw = self.refw_list[self.time]
            vis.add_geometry(self.refw)

        print(f"EPE - {self.time}: {self.epes[self.time]}")

        ctr = vis.get_view_control()
        ctr.convert_from_pinhole_camera_parameters(param)

    def run(self):
        # Define callbacks.
        def toggle_next(vis):
            self.time += 1
            if self.time >= self.loaded_frames:
                self.time = 0
            print("Showing time", self.time, self.loaded_frames)
            self.update_ref(vis)
            self.update_cur(vis)
            self.update_refw(vis)
            return False
        
        def toggle_previous(vis):
            self.time -= 1
            if self.time < 0:
                self.time = self.loaded_frames - 1
            self.update_ref(vis)
            self.update_cur(vis)
            self.update_refw(vis)
            return False

        def toggle_ref(vis):
            self.show_ref = not self.show_ref
            self.update_ref(vis)
            return False

        def toggle_cur(vis):
            self.show_cur = not self.show_cur
            self.update_cur(vis)
            return False
        
        def toggle_refw(vis):
            self.show_refw = not self.show_refw
            self.update_refw(vis)
            return False
     
        key_to_callback = {}
        key_to_callback[ord("D")] = toggle_next
        key_to_callback[ord("A")] = toggle_previous
        key_to_callback[ord("R")] = toggle_ref
        key_to_callback[ord("C")] = toggle_cur
        key_to_callback[ord("W")] = toggle_refw

        # Add mesh at initial time step.
        assert self.time < self.loaded_frames

        print("Showing time", self.time)

        self.refw = self.refw_list[self.time]
        self.show_refw = True 

        self.cur = self.cur_list[self.time]
        self.show_cur = True

        o3d.visualization.draw_geometries_with_key_callbacks([self.unit_bbox, self.refw, self.cur], key_to_callback)

################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################

def predict_pose_codes(labels, split):
    #######################################################
    # Predict the pose codes with the learned encoder
    #######################################################

    print("Predict codes again...")
    print()

    # Go over sequence and compute losses 
    trainer.pose_encoder.eval()

    for p in trainer.pose_encoder.parameters():
        p.requires_grad = False

    for frame_i in tqdm(range(len(labels))):
        
        label = labels[frame_i]

        identity_name, animation_name, sample_id = label['identity_name'], label['animation_name'], label['sample_id']

        cur_path = os.path.join(data_dir, identity_name, animation_name, sample_id)

        # print(cur_path)

        ################################################################################################
        # VOXEL GRIDS
        ################################################################################################
        # Load input voxel grids
        inputs_path = os.path.join(cur_path, f'partial_views/voxelized_view0_{res}res.npz')
        occupancies = np.unpackbits(np.load(inputs_path)['compressed_occupancies'])
        inputs_np = np.reshape(occupancies, (res,)*3).astype(np.float32)
        inputs = torch.from_numpy(inputs_np)[None, :].cuda()

        #######################################################
        # voxels_trimesh = VoxelGrid(inputs_np).to_mesh()
        # voxels_mesh = o3d.geometry.TriangleMesh(
        #     o3d.utility.Vector3dVector(voxels_trimesh.vertices),
        #     o3d.utility.Vector3iVector(voxels_trimesh.faces)
        # )
        # voxels_mesh.compute_vertex_normals()
        # voxels_mesh.paint_uniform_color(Colors.green)
        # o3d.visualization.draw_geometries([voxels_mesh])
        #######################################################

        predicted_pose_code = trainer.pose_encoder(inputs)

        if split == "train":
            trainer.pose_codes_train[frame_i] = predicted_pose_code
        elif split == "val":
            trainer.pose_codes_val[frame_i] = predicted_pose_code
        else:
            exit()

    print("Codes predicted for", split)

################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################

class Trainer():

    def __init__(
        self, 
        debug,
        train_dataset, val_dataset,
        exp_dir, exp_name,
        eval_every,
        loaded_checkpoint
    ):
        self.debug = debug

        ###############################################################################################
        # Model
        ###############################################################################################

        ########################
        # SHAPE Codes
        ########################
        self.shape_codes = torch.ones(num_identities, 1, shape_codes_dim).normal_(0, 0.1).cuda()
        
        pretrained_shape_codes = loaded_checkpoint['shape_codes'].cuda().detach().clone()

        print(self.shape_codes.shape)
        print(pretrained_shape_codes.shape)

        if self.shape_codes.shape[0] != pretrained_shape_codes.shape[0]:
            print(self.shape_codes.shape, "vs", pretrained_shape_codes.shape)
            raise Exception("Shape codes shape does not match pretrained codes")
        else:
            self.shape_codes = pretrained_shape_codes

        ########################
        # Pose Codes
        ########################
        train_set_size = len(train_dataset)
        val_set_size   = len(val_dataset)
        total_size     = train_set_size + val_set_size

        assert train_set_size > 0, train_set_size
        assert val_set_size   > 0, val_set_size

        pose_codes = torch.ones(total_size, 1, pose_codes_dim).normal_(0, 0.1).cuda()

        pretrained_pose_codes = loaded_checkpoint['pose_codes'].cuda().detach().clone()
        print("pretrained_pose_codes.shape =", pretrained_pose_codes.shape)        

        if pose_codes.shape[0] != pretrained_pose_codes.shape[0]:
            raise Exception("Pose codes shape does not match pretrained codes")
        else:
            pose_codes = pretrained_pose_codes

        pose_codes.requires_grad = False

        self.pose_codes_train = torch.ones(train_set_size, 1, pose_codes_dim).normal_(0, 0.1).cuda()
        self.pose_codes_train = pose_codes[:train_set_size]
        self.pose_codes_val   = torch.ones(val_set_size, 1, pose_codes_dim).normal_(0, 0.1).cuda()
        self.pose_codes_val   = pose_codes[train_set_size:]

        print("Pose codes ALL:   shape =", pose_codes.shape,            pose_codes.requires_grad)        
        print("Pose codes train: shape =", self.pose_codes_train.shape, self.pose_codes_train.requires_grad)        
        print("Pose codes val:   shape =", self.pose_codes_val.shape,   self.pose_codes_val.requires_grad)     
        print()

        assert len(train_dataset) == self.pose_codes_train.shape[0], f"{len(train_dataset)} vs {self.pose_codes_train.shape[0]}"
        assert len(val_dataset)   == self.pose_codes_val.shape[0],   f"{len(val_dataset)} vs {self.pose_codes_val.shape[0]}"

        ###############################################################################################
        ###############################################################################################

        ########################
        # Encoder
        ########################
        self.pose_encoder = encoders.PoseEncoder(
            code_dim=pose_codes_dim,
            res=res
        ).cuda()
        nnutils.print_num_parameters(self.pose_encoder)

        #####################################################################################
        # Set up optimizer.
        #####################################################################################
        lr_schedule = nnutils.StepLearningRateSchedule(learning_rate_schedules['pose_encoder'])
        self.lr_schedules = [
            lr_schedule,
        ]

        learnable_parameters = [
            {
                "params": self.pose_encoder.parameters(),
                "lr": lr_schedule.get_learning_rate(0),
            }
        ]

        self.optimizer = torch.optim.Adam(learnable_parameters)
        
        #####################################################################################
        # Datasets
        #####################################################################################
        self.train_dataset  = train_dataset
        self.val_dataset    = val_dataset

        self.eval_every = eval_every

        self.exp_dir = exp_dir
        self.exp_name = exp_name
        self.exp_path = os.path.join(exp_dir, exp_name)

        if not self.debug:
            self.checkpoints_dir = os.path.join(self.exp_path, 'checkpoints')
            if not os.path.exists(self.checkpoints_dir):
                os.makedirs(self.checkpoints_dir)

            # Copy config.py over
            shutil.copy("encode_pose_codes.py", os.path.join(self.exp_path, "encode_pose_codes.config.py"))
            
            self.writer = SummaryWriter(os.path.join(self.exp_path, 'summary'))
            
        self.val_min = None

    def train_step(self, batch, epoch):

        self.pose_encoder.train()

        #####################################################################################
        # Set gradients to None
        #####################################################################################
        # Decoders
        for param in self.pose_encoder.parameters():
            param.grad = None
        #####################################################################################

        loss, loss_dict = self.compute_loss(batch, "train")

        loss.backward()

        self.optimizer.step()

        return loss_dict

    def compute_loss(self, batch, split):

        ################################################################
        # Get data
        ################################################################

        data = batch.get('data')
        indices = batch.get('idx')

        inputs = data['inputs'].cuda() # [bs, res, res, res]

        predicted_pose_codes_batch = self.pose_encoder(inputs)

        if split == "train":
            gt_pose_codes_batch = self.pose_codes_train[indices, ...] # [bs, 1, C]
        elif split == "val":
            gt_pose_codes_batch = self.pose_codes_val[indices, ...] # [bs, 1, C]
        
        gt_pose_codes_batch.requires_grad = False

        loss = F.mse_loss(predicted_pose_codes_batch, gt_pose_codes_batch)

        # Prepare dict of losses
        loss_dict = {
            'total': loss.item(),
        }

        return loss, loss_dict

    def train_model(self, epochs):

        for p in trainer.pose_encoder.parameters():
            p.requires_grad = True

        start = 0

        if not self.debug:
            start = self.load_latest_checkpoint()

        # Multi-GPU
        if torch.cuda.device_count() > 1:
            print()
            print(f"########## Using {torch.cuda.device_count()} GPUs")
            print()
            self.pose_encoder = torch.nn.DataParallel(self.pose_encoder)

        for epoch in range(start, epochs):

            self.current_epoch = epoch
            
            print()
            print()
            print("########################################################################################")
            print("########################################################################################")
            print(f'Start epoch {epoch} - {self.exp_name}')
            
            train_data_loader = self.train_dataset.get_loader()

            nnutils.adjust_learning_rate(self.lr_schedules, self.optimizer, epoch)

            ############################################################
            # VALIDATION
            ############################################################
            if epoch % self.eval_every == 0:

                # Store latest checkpoint
                if not self.debug:
                    # Store latest checkpoint
                    self.save_special_checkpoint(epoch, "latest")
                    
                    # Store a checkpoint every 10 epochs
                    # self.save_checkpoint(epoch)

                if do_validation:

                    print()
                    print("-----------------------------------------------")
                    print('\tValidation...')
                    
                    val_losses = self.compute_val_loss()
                    val_loss = val_losses['total']
                    
                    print()
                    print(f'\tValidation done - val loss {val_loss}')

                    if self.val_min is None:
                        print("\tInitializing best checkpoint with first val error")
                        self.val_min = val_loss

                    # Update best model
                    if val_loss < self.val_min:

                        print("\tNew best checkpoint!")

                        self.val_min = val_loss
                        
                        if not self.debug:
                            # Saving new best checkpoint
                            self.save_special_checkpoint(epoch, "best")

                            for path in glob(os.path.join(self.exp_path, 'val_min_encoder=*')):
                                os.remove(path)
                            
                            np.save(
                                os.path.join(self.exp_path, f'val_min_encoder={epoch}'), 
                                [epoch, val_loss]
                            )

                    if not self.debug:
                        self.writer.add_scalar('val/encoder', val_losses['total'], epoch)

                    print("-----------------------------------------------")
                    print()

            ############################################################

            ############################################################
            # TRAIN
            ############################################################
            num_batches = len(train_data_loader)

            sum_loss_total = 0

            for batch in tqdm(train_data_loader):
                loss_dict = self.train_step(batch, epoch)
                loss_total = loss_dict['total']
                sum_loss_total += loss_total

            print(
                "Epoch {} / {} - Current loss: {:.16f}".format(
                    epoch, epochs, 
                    loss_total,
                ),
                flush=True
            )

            if not self.debug and epoch % self.eval_every == 0:
                self.writer.add_scalar('train/encoder', sum_loss_total / num_batches, epoch)       

    def save_checkpoint(self, epoch):
        path = os.path.join(self.checkpoints_dir, f'pose_encoder_{epoch}.tar')

        if isinstance(self.pose_encoder, torch.nn.DataParallel):
            pose_encoder_state_dict = self.pose_encoder.module.state_dict()
        else:
            pose_encoder_state_dict = self.pose_encoder.state_dict()

        if not os.path.exists(path):
            torch.save(
                {
                    'epoch':epoch,
                    'model_state_dict_pose_encoder': pose_encoder_state_dict,
                    'optimizer_state_dict': self.optimizer.state_dict()
                }, 
                path
            )

    def save_special_checkpoint(self, epoch, special_name):
        path = os.path.join(self.checkpoints_dir, f'pose_encoder_{special_name}.tar')

        if isinstance(self.pose_encoder, torch.nn.DataParallel):
            pose_encoder_state_dict = self.pose_encoder.module.state_dict()
        else:
            pose_encoder_state_dict = self.pose_encoder.state_dict()

        torch.save(
            {
                'epoch':epoch,
                'model_state_dict_pose_encoder': pose_encoder_state_dict,
                'optimizer_state_dict': self.optimizer.state_dict()
            }, 
            path
        )

    def load_latest_checkpoint(self):
        checkpoints = [m for m in os.listdir(self.checkpoints_dir)]

        if len(checkpoints) == 0:
            print()
            print('No checkpoints found at {}'.format(self.checkpoints_dir))
            return 0

        # If we're here, we have at least 1 checkpoint
        latest_checkpoint_path = os.path.join(self.checkpoints_dir, "pose_encoder_latest.tar")

        if not os.path.exists(latest_checkpoint_path):
            raise Exception(f'Latest checkpoint {latest_checkpoint_path} does not exist!')

        checkpoint = torch.load(latest_checkpoint_path)

        self.pose_encoder.load_state_dict(checkpoint['model_state_dict_pose_encoder'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        epoch = checkpoint['epoch']
        
        print()
        print('Loaded checkpoint from: {}'.format(latest_checkpoint_path))
        
        return epoch

    def compute_val_loss(self):
        self.pose_encoder.eval()

        assert not self.pose_encoder.training

        sum_val_loss = 0

        num_batches = target_num_val_batches
        max_batches_val = len(self.val_dataset) // batch_size
        num_batches = min(max_batches_val, num_batches)

        print(f"\tValidation on {num_batches} batches (out of {max_batches_val})")

        # Go over the validation batches
        for i in range(num_batches):
            
            try:
                val_batch = self.val_data_iterator.next()
            except:
                self.val_data_iterator = self.val_dataset.get_loader(shuffle=True).__iter__()
                val_batch = self.val_data_iterator.next()

            ################################################################################################
            # Compute loss
            ################################################################################################
            _, loss_dict = self.compute_loss(
                val_batch,
                "val"
            )

            sum_val_loss += loss_dict['total']

        return {
            'total': sum_val_loss / num_batches,
        }


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Run Model'
    )

    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-n', '--extra_name', default="", type=str)

    try:
        args = parser.parse_args()
    except:
        args = parser.parse_known_args()[0]

    torch.backends.cudnn.benchmark = True

    viz_before_optim = False
    viz_after_optim  = False

    if viz_before_optim and not viz_after_optim:
        input("Viz before. Continue?")
    if viz_after_optim and not viz_before_optim:
        input("Viz after. Continue?")
    if viz_before_optim and viz_after_optim:
        input("Viz before and after. Continue?")

    test_sigma = [0.002]

    ########################################################################################################################
    # Experiment and data dir
    ########################################################################################################################
    print()
    print("#"*100)

    exp_version = "npms"
    exps_dir  = os.path.join(cfg.exp_dir, exp_version)
    print("exps dir", exps_dir)

    # data_dir = "/cluster/lothlann/ppalafox/datasets"
    data_dir = f"/cluster_HDD/lothlann/ppalafox/datasets_SSD@20_04_2021"
    print("data dir", data_dir)
    
    print("#"*100)

    #######################################################################################################
    # Checkpoint
    #######################################################################################################
    
    # Default
    shape_codes_dim = 256
    pose_codes_dim = 256
    fs = 512
    use_se3 = False
    positional_enc = False
    dropout = None # [0, 1, 2, 3, 4, 5, 6, 7]
    dropout_prob = 0.2
    norm_layers = None # [0, 1, 2, 3, 4, 5, 6, 7]
    weight_norm = False

    # ----------------------------------------------------------------------------------------------------

    exp_name = "2021-03-15__NESHPOD__bs4__lr-0.0005-0.0005-0.001-0.001_intvl30__s256-512-8l__p256-1024-8l__woSE3__wShapePosEnc__wPosePosEnc__woDroutS__woDroutP__wWNormS__wWNormP__ON__MIX-POSE__AMASS-50id-5000__MIXAMO-165id-20000__CAPE-35id-20533"
    checkpoint = 150
    shape_codes_dim = 256
    pose_codes_dim = 256
    pose_fs = 1024
    positional_enc = True
    use_se3 = False
    norm_layers = [0, 1, 2, 3, 4, 5, 6, 7]
    weight_norm = True

    # ----------------------------------------------------------------------------------------------------

    exp_dir = os.path.join(exps_dir, exp_name)
    loaded_checkpoint = nnutils.load_checkpoint(exp_dir, checkpoint)

    ########################################################################################################################
    # Options
    ########################################################################################################################
    # If debugging, nothing is stored
    debug = args.debug

    epochs = 100
    do_validation = True
    eval_every = 1

    lr = 1e-6 # 5e-5 #1e-5 #5e-4
    adjust_lr_every = 15
    factor = 0.5 # 0.5

    pose_network_specs = {
        "dims" : [pose_fs] * 8,
        "dropout" : dropout,
        "dropout_prob" : dropout_prob,
        "norm_layers" : norm_layers,
        "latent_in" : [4],
        "xyz_in_all" : False,
        "latent_dropout" : False,
        "weight_norm" : weight_norm,
        "positional_enc": positional_enc,
        "n_positional_freqs": 8,
        "n_alpha_epochs": 0, 
    }

    learning_rate_schedules = {
        "pose_encoder": {
            "type": "step",
            "initial": lr,
            "interval": adjust_lr_every,
            "factor": factor,
        }
    }

    # Data
    res = 256
    batch_size  = 16 # 4 # 12
    num_workers = 16 # 4 # 8
    target_num_val_batches = 8
    cache_data = False
    
    num_to_eval = 10
    load_every = 1
    from_frame_id = 0

    ########################################################################################################################
    
    ################################ HUMAN ################################
    dataset_train_name = "MIX-POSE__AMASS-50id-5000__MIXAMO-165id-20000__CAPE-35id-20533"

    ################################# MANO ################################
    # dataset_train_name = "MANO-POSE-TRAIN-200id-40000ts-40000seqs"
    
    ########################################################################################################################
    
    from utils.parsing_utils import get_dataset_type_from_dataset_name
    dataset_type = get_dataset_type_from_dataset_name(dataset_train_name)
    splits_dir = f"{cfg.splits_dir}_{dataset_type}"

    labels_train, labels_val, labels_tpose = compute_labels_pose_encoder_training(
        data_dir, dataset_train_name, splits_dir
    )

    #######################################################################################################
    # Experiment name
    #######################################################################################################
    exp_name_encoder = f"POSE_ENCODER__lr{lr}_bs{batch_size}__cpt{checkpoint}_p{pose_codes_dim}_fs{pose_fs}_res{res}_adj{factor}every{adjust_lr_every}"
    exp_name_encoder += "_wPosEnc" if positional_enc else "_woPosEnc"
    exp_name_encoder += f"__ON__{dataset_train_name}"
    if args.extra_name != "":
        exp_name_encoder += f"__{args.extra_name}"
    
    print("#"*100)
    print(f"Exp name: {exp_name_encoder}")
    print("#"*100)

    #######################################################################################################
    # Data
    #######################################################################################################
 
    num_identities = len(labels_tpose)
    batch_size = min(batch_size, len(labels_train))

    train_dataset = data_sdflow.VoxelsDataset(
        data_dir=data_dir,
        labels_json={'labels': labels_train, 'labels_tpose': labels_tpose,},
        batch_size=batch_size, 
        num_workers=num_workers,
        res=res,
        cache_data=cache_data
    )

    val_dataset = data_sdflow.VoxelsDataset(
        data_dir=data_dir,
        labels_json={'labels': labels_val, 'labels_tpose': labels_tpose,},
        batch_size=batch_size, 
        num_workers=num_workers,
        res=res,
        cache_data=cache_data
    )

    print("Train dataset", len(train_dataset))
    print("Val dataset  ", len(val_dataset))

    
    ########################
    # Pose decoder
    ########################
    total_codes_dim = pose_codes_dim + shape_codes_dim
    if use_se3:
        print()
        print("Using SE(3) formulation for the PoseDecoder")
        pose_decoder = PoseDecoderSE3(total_codes_dim, **pose_network_specs).cuda()
    else:
        print()
        print("Using normal (translation) formulation for the PoseDecoder")
        pose_decoder = PoseDecoder(total_codes_dim, **pose_network_specs).cuda()
    pose_decoder.load_state_dict(loaded_checkpoint['model_state_dict_pose_decoder'])
    for param in pose_decoder.parameters():
        param.requires_grad = False
    pose_decoder.eval()
    nnutils.print_num_parameters(pose_decoder)

    ########################
    # Pose encoder
    ########################
    # Trainer
    trainer = Trainer(
        debug,
        train_dataset, val_dataset,
        exps_dir, exp_name_encoder,
        eval_every,
        loaded_checkpoint
    )

    # Viz before optimizing
    if viz_before_optim:
        # predict_pose_codes(labels_train, "train")
        viewer = Viewer(
            "train",
            labels_train,
            num_to_eval=num_to_eval,
            load_every=load_every,
            from_frame_id=from_frame_id,
            res=res
        )
        viewer.run()

        # predict_pose_codes(labels_val, "val")
        viewer = Viewer(
            "val",
            labels_val,
            num_to_eval=num_to_eval,
            load_every=load_every,
            from_frame_id=from_frame_id,
            res=res
        )
        viewer.run()

    ########################
    # Train
    ########################    
    trainer.train_model(epochs=epochs)
    ########################
    ########################

    print()
    print("###################################################################")
    print("###################################################################")
    print("###################################################################")
    print("Optimization done!")
    print("###################################################################")
    print("###################################################################")
    print("###################################################################")
    print()

    if viz_after_optim:
        # Predict pose codes for train
        predict_pose_codes(labels_train, "train")
        viewer = Viewer(
            "train",
            labels_train,
            num_to_eval=num_to_eval,
            load_every=load_every,
            from_frame_id=from_frame_id,
            res=res
        )
        viewer.run()

        # Predict pose codes for val
        predict_pose_codes(labels_val, "val")
        viewer = Viewer(
            "val",
            labels_val,
            num_to_eval=num_to_eval,
            load_every=load_every,
            from_frame_id=from_frame_id,
            res=res
        )
        viewer.run()


