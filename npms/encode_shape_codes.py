import sys,os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import os
import trimesh
import numpy as np
import json
import open3d as o3d
import torch
from torch.nn import functional as F
from tqdm import tqdm
from timeit import default_timer as timer
import argparse
from torch.utils.tensorboard import SummaryWriter
from glob import glob

import models.inference_encoder as encoders
from models.shape_decoder import ShapeDecoder

import datasets.voxels_dataset as data_sdflow

from utils.utils import compute_dataset_mapping
import utils.nnutils as nnutils
from data_scripts.prepare_labels_shape_encoder import compute_labels_shape_encoder_training

import config as cfg


def predict_shape_codes(res, data_dir, labels, shape_encoder, shape_codes_shape, only_once_per_identity=True, compute_median_shape_code=True):
    #######################################################
    # Predict the shape codes with the learned encoder
    #######################################################

    shape_codes = torch.zeros(shape_codes_shape, dtype=torch.float32).cuda()
    shape_codes.requires_grad = False

    shape_codes_tmp = None
    if not only_once_per_identity:
        shape_codes_tmp = torch.zeros((len(labels), 1, shape_codes_shape[-1]), dtype=torch.float32).cuda()

    num_samples_used = 0

    print("Predict codes again...")
    print()

    # Go over sequence and compute losses 
    # shape_encoder.eval()

    for p in shape_encoder.parameters():
        p.requires_grad = False

    is_identity_seen_dict = {i: False for i in range(shape_codes.shape[0])}

    for frame_i in range(len(labels)):
        
        label = labels[frame_i]

        dataset, identity_id, identity_name, animation_name, sample_id = \
            label['dataset'], label['identity_id'], label['identity_name'], label['animation_name'], label['sample_id']

        if only_once_per_identity and is_identity_seen_dict[identity_id]:
            continue

        cur_path = os.path.join(data_dir, dataset, identity_name, animation_name, sample_id)

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

        predicted_shape_code = shape_encoder(inputs).squeeze(0)

        if only_once_per_identity:
            shape_codes[identity_id] = predicted_shape_code
        else:
            # Typically, we want to be here when we only have a dataset of a single identity
            if compute_median_shape_code:
                shape_codes_tmp[frame_i] = predicted_shape_code
            else:
                shape_codes[identity_id] += predicted_shape_code
                num_samples_used += 1

        # Mark identity as seen
        is_identity_seen_dict[identity_id] = True

        print(f"\t Identity {identity_id} / {len(labels)}: inputs mean {inputs.mean().item()} ({inputs.std().item()}) | pred code mean {predicted_shape_code.mean().item()} ({predicted_shape_code.std().item()})")

    print()
    print("#"*60)
    print(f"Codes predicted for {len(is_identity_seen_dict)} identities")
    print("#"*60)

    # Make sure we saw all identities
    for identity_id, seen in is_identity_seen_dict.items():
        assert seen, f"Identity {identity_id} was not seen!"

    # Get our shape codes for the 1-identity-dataset case
    if not only_once_per_identity:
        if compute_median_shape_code:
            print(shape_codes_tmp.shape)
            shape_codes, _ = torch.median(shape_codes_tmp, dim=0, keepdim=True)
        else:
            shape_codes /= num_samples_used

    print()
    print("#"*60)
    print(f"Statistics for computed shape_codes ({shape_codes.shape}):")
    print(shape_codes.mean().item(), shape_codes.std().item())
    print("#"*60)
    # input("Press ENTER to continue...")

    return shape_codes

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
        # Shape Codes
        ########################
        train_set_size = len(train_to_orig)
        val_set_size   = len(val_to_orig)
        total_size     = train_set_size + val_set_size

        assert train_set_size > 0, train_set_size
        assert val_set_size   > 0, val_set_size

        pretrained_shape_codes = loaded_checkpoint['shape_codes'].cuda().detach().clone()
        print("pretrained_shape_codes.shape =", pretrained_shape_codes.shape)   

        self.shape_codes_train = pretrained_shape_codes[list(train_to_orig.values())].clone()
        self.shape_codes_val   = pretrained_shape_codes[list(val_to_orig.values())].clone()

        self.shape_codes_train.requires_grad = False
        self.shape_codes_val.requires_grad = False

        print("Shape codes train: shape =", self.shape_codes_train.shape, self.shape_codes_train.requires_grad)        
        print("Shape codes val:   shape =", self.shape_codes_val.shape,   self.shape_codes_val.requires_grad)     
        print()

        ###############################################################################################
        ###############################################################################################

        ########################
        # Encoder
        ########################
        self.shape_encoder = encoders.ShapeEncoder(
            code_dim=shape_codes_dim,
            res=res
        ).cuda()
        nnutils.print_num_parameters(self.shape_encoder)

        #####################################################################################
        # Set up optimizer.
        #####################################################################################
        lr_schedule = nnutils.StepLearningRateSchedule(learning_rate_schedules['shape_encoder'])
        self.lr_schedules = [
            lr_schedule,
        ]

        learnable_parameters = [
            {
                "params": self.shape_encoder.parameters(),
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
            
            self.writer = SummaryWriter(os.path.join(self.exp_path, 'summary'))
            
        self.val_min = None

    def train_step(self, batch, epoch):

        self.shape_encoder.train()

        #####################################################################################
        # Set gradients to None
        #####################################################################################
        # Decoders
        for param in self.shape_encoder.parameters():
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
        identity_ids = batch.get('identity_id')

        inputs = data['inputs'].cuda() # [bs, res, res, res]

        predicted_shape_codes_batch = self.shape_encoder(inputs)

        if split == "train":
            gt_shape_codes_batch = self.shape_codes_train[identity_ids, ...] # [bs, 1, C]
        elif split == "val":
            gt_shape_codes_batch = self.shape_codes_val[identity_ids, ...] # [bs, 1, C]
        
        gt_shape_codes_batch.requires_grad = False

        loss = F.mse_loss(predicted_shape_codes_batch, gt_shape_codes_batch)

        # Prepare dict of losses
        loss_dict = {
            'total': loss.item(),
        }

        return loss, loss_dict

    def train_model(self, epochs):

        for p in trainer.shape_encoder.parameters():
            p.requires_grad = True

        start = 0

        if not self.debug:
            start = self.load_latest_checkpoint()

        # Multi-GPU
        if torch.cuda.device_count() > 1:
            print()
            print(f"########## Using {torch.cuda.device_count()} GPUs")
            print()
            self.shape_encoder = torch.nn.DataParallel(self.shape_encoder)

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
        path = os.path.join(self.checkpoints_dir, f'shape_encoder_{epoch}.tar')

        if isinstance(self.shape_encoder, torch.nn.DataParallel):
            shape_encoder_state_dict = self.shape_encoder.module.state_dict()
        else:
            shape_encoder_state_dict = self.shape_encoder.state_dict()

        if not os.path.exists(path):
            torch.save(
                {
                    'epoch':epoch,
                    'model_state_dict_shape_encoder':shape_encoder_state_dict,
                    'optimizer_state_dict': self.optimizer.state_dict()
                }, 
                path
            )

    def save_special_checkpoint(self, epoch, special_name):
        path = os.path.join(self.checkpoints_dir, f'shape_encoder_{special_name}.tar')

        if isinstance(self.shape_encoder, torch.nn.DataParallel):
            shape_encoder_state_dict = self.shape_encoder.module.state_dict()
        else:
            shape_encoder_state_dict = self.shape_encoder.state_dict()

        torch.save(
            {
                'epoch':epoch,
                'model_state_dict_shape_encoder': shape_encoder_state_dict,
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
        latest_checkpoint_path = os.path.join(self.checkpoints_dir, "shape_encoder_latest.tar")

        if not os.path.exists(latest_checkpoint_path):
            raise Exception(f'Latest checkpoint {latest_checkpoint_path} does not exist!')

        checkpoint = torch.load(latest_checkpoint_path)

        self.shape_encoder.load_state_dict(checkpoint['model_state_dict_shape_encoder'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        epoch = checkpoint['epoch']
        
        print()
        print('Loaded checkpoint from: {}'.format(latest_checkpoint_path))
        
        return epoch

    def compute_val_loss(self):
        self.shape_encoder.eval()

        assert not self.shape_encoder.training

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

    torch.backends.cudnn.benchmark = True

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

    if viz_before_optim or viz_after_optim:
        input("Viz?")

    # Reconstruction resolution
    mult = 2
    reconstruction_res = mult * 256
    max_batch = (mult * 32)**3

    ########################################################################################################################
    # Experiment and data dir
    ########################################################################################################################
    print()
    print("#"*100)

    exp_version = "npms"
    exps_dir  = os.path.join(cfg.exp_dir, exp_version)
    print("exps dir", exps_dir)

    # data_dir = f"/cluster/lothlann/ppalafox/datasets"
    data_dir = f"/cluster_HDD/lothlann/ppalafox/datasets_SSD@20_04_2021"
    print("data dir", data_dir)
    
    print("#"*100)

    #######################################################################################################
    # Checkpoint
    #######################################################################################################

    # Default
    shape_codes_dim = 256
    fs = 512
    positional_enc = False

    dropout = None #[0, 1, 2, 3, 4, 5, 6, 7]
    dropout_prob = 0.2
    norm_layers = None # [0, 1, 2, 3, 4, 5, 6, 7]
    weight_norm = False

    # ----------------------------------------------------------------------------------------------------

    exp_name = "2021-06-11__NPMs__SHAPE_nss0.7_uni0.3__bs8__lr-0.0005-0.0005-0.001-0.001_intvl500__s256-512-8l__p256-1024-8l__woSE3__woShapePosEnc__woPosePosEnc__woDroutS__woDroutP__wWNormS__wWNormP__ON__MIX-SHAPE__CAPE_MIXAMO-250id__AMASS-50id"
    checkpoint = 3900
    positional_enc = False
    shape_codes_dim = 256
    fs = 512
    norm_layers = [0, 1, 2, 3, 4, 5, 6, 7]
    weight_norm = True

    # ----------------------------------------------------------------------------------------------------

    print("#"*60)
    print("exp_name        ", exp_name)
    print("checkpoint      ", checkpoint)
    print("positional_enc  ", positional_enc)
    print("shape_codes_dim ", shape_codes_dim)
    print("fs              ", fs)
    print("#"*60)

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

    lr = 5e-6
    adjust_lr_every = 5
    factor = 0.5 # 0.5  

    shape_network_specs = {
        "dims" : [fs] * 8,
        "dropout" : dropout, # [0, 1, 2, 3, 4, 5, 6, 7],
        "dropout_prob" : dropout_prob,
        "norm_layers" : norm_layers,
        "latent_in" : [4],
        "xyz_in_all" : False,
        "latent_dropout" : False,
        "weight_norm" : weight_norm, # True
        "positional_enc": positional_enc,
        "n_positional_freqs": 8,
    }

    learning_rate_schedules = {
        "shape_encoder": {
            "type": "step",
            "initial": lr,
            "interval": adjust_lr_every,
            "factor": factor,
        }
    }

    # Data
    res = 256
    batch_size = 24
    num_workers = 8
    target_num_val_batches = 8
    cache_data = False
    print("res         ", res)
    print("batch_size  ", batch_size)
    print("num_workers ", num_workers)
    
    num_to_eval = 10
    load_every = 5000
    from_frame_id = 4000

    ########################################################################################################################
    ########################################################################################################################
    
    #####################################################################################################
    ## ORIGINAL dataset (the one we used to train the shape MLP)
    #####################################################################################################
    if "amass" in exp_name.lower() or "mixamo" in exp_name.lower() or "cape" in exp_name.lower():
        # dataset_orig_name = "SHAPE_MIX__A-amass-419id__B-mixamo_trans_all-205id"
        dataset_orig_name = "MIX-SHAPE__CAPE_MIXAMO-250id__AMASS-50id"
    elif "mano" in exp_name.lower():
        dataset_orig_name = "SHAPE_MANO-train-200ts-200seqs"
    
    #####################################################################################################
    ## TRAIN dataset of poses to train the shape encoder, which maps from a pose to the corresponding t-pose
    #####################################################################################################
    if "amass" in exp_name.lower() or "mixamo" in exp_name.lower() or "cape" in exp_name.lower():
        # dataset_train_name = "MIX-POSE__AMASS-50id-4309__MIXAMO-165id-10065__CAPE-35id-10119"
        dataset_train_name = "MIX-POSE__AMASS-50id-10349__MIXAMO-165id-40000__CAPE-35id-20533"
    elif "mano" in exp_name.lower():
        dataset_train_name = "MANO-POSE-TRAIN-200id-40000ts-40000seqs"

    print("dataset_orig_name ", dataset_orig_name)
    print("dataset_train_name", dataset_train_name)
    # input("Train on above datasets?")
    
    ########################################################################################################################
    ########################################################################################################################
    
    from utils.parsing_utils import get_dataset_type_from_dataset_name
    dataset_type = get_dataset_type_from_dataset_name(dataset_train_name)
    splits_dir = f"{cfg.splits_dir}_{dataset_type}"

    labels_train, labels_tpose_train, labels_val, labels_tpose_val = compute_labels_shape_encoder_training(
        data_dir, dataset_train_name, splits_dir
    )

    #######################################################################################################
    # Experiment name
    #######################################################################################################
    
    exp_name_encoder = f"SHAPE_ENCODER_lr{lr}__cpt{checkpoint}_p{shape_codes_dim}_fs{fs}_res{res}_adj{factor}every{adjust_lr_every}"
    exp_name_encoder += "_wPosEnc" if positional_enc else "_woPosEnc"
    exp_name_encoder += f"__ON__{dataset_train_name}"
    if len(args.extra_name) > 0:
        exp_name_encoder += f"__{args.extra_name}"
    # input(f"Exp name: {exp_name_encoder}")
    print(f"Exp name: {exp_name_encoder}")

    #######################################################################################################
    # Data
    #######################################################################################################

    # Compute mapping from current identities to original identities in the "unsplit" dataset
    train_to_orig = compute_dataset_mapping(os.path.join(data_dir, splits_dir), dataset_orig_name, labels_tpose_train)
    val_to_orig   = compute_dataset_mapping(os.path.join(data_dir, splits_dir), dataset_orig_name, labels_tpose_val)

    batch_size_train = min(batch_size, len(train_to_orig))
    batch_size_val   = min(batch_size, len(val_to_orig))

    train_dataset = data_sdflow.VoxelsDataset(
        data_dir=data_dir,
        labels_json={'labels': labels_train, 'labels_tpose': labels_tpose_train,},
        batch_size=batch_size_train, 
        num_workers=num_workers,
        res=res,
        cache_data=cache_data
    )

    val_dataset = data_sdflow.VoxelsDataset(
        data_dir=data_dir,
        labels_json={'labels': labels_val, 'labels_tpose': labels_tpose_val,},
        batch_size=batch_size_val, 
        num_workers=num_workers,
        res=res,
        cache_data=cache_data
    )

    print("Train dataset", len(train_dataset))
    print("Val dataset  ", len(val_dataset))

    ########################
    # Shape decoder
    ########################
    if viz_after_optim or viz_before_optim:
        shape_decoder = ShapeDecoder(shape_codes_dim, **shape_network_specs).cuda()
        shape_decoder.load_state_dict(loaded_checkpoint['model_state_dict_shape_decoder'])
        for param in shape_decoder.parameters():
            param.requires_grad = False
        shape_decoder.eval()
        nnutils.print_num_parameters(shape_decoder)

    ########################
    # Shape encoder
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
        from viz.viz_shape import ViewerShape

        with torch.no_grad():
            print("#"*60)
            print("Loading viewer for train frames")
            print("#"*60)
            viewer_shape = ViewerShape(
                labels_train, data_dir,
                shape_decoder, trainer.shape_codes_train,
                num_to_eval,
                load_every,
                from_frame_id,
                res=reconstruction_res,
                max_batch=max_batch
            )
            viewer_shape.run()

            print("#"*60)
            print("Loading viewer for val frames")
            print("#"*60)
            viewer_shape = ViewerShape(
                labels_val, data_dir,
                shape_decoder, trainer.shape_codes_val,
                num_to_eval,
                load_every,
                from_frame_id,
                res=reconstruction_res,
                max_batch=max_batch
            )
            viewer_shape.run()

    #################################################################
    # Train
    #################################################################    
    trainer.train_model(epochs=epochs)
    #################################################################    
    #################################################################    

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
        from viz.viz_shape import ViewerShape
        
        #################################################################
        # Predict pose codes for train
        #################################################################
        print("#"*60)
        print("Predicting train codes after optimization")
        print("#"*60)
        shape_codes = predict_shape_codes(
            res, data_dir, labels_train, trainer.shape_encoder, trainer.shape_codes_train.shape,
            only_once_per_identity=True
        )

        print("#"*60)
        print("Loading viewer for train frames")
        print("#"*60)
        viewer_shape = ViewerShape(
            labels_train, data_dir,
            shape_decoder, shape_codes,
            num_to_eval,
            load_every,
            from_frame_id,
            res=reconstruction_res,
            max_batch=max_batch
        )
        viewer_shape.run()

        #################################################################
        # Predict pose codes for val
        #################################################################
        print("#"*60)
        print("Predicting val codes after optimization")
        print("#"*60)
        shape_codes = predict_shape_codes(
            res, data_dir, labels_val, trainer.shape_encoder, trainer.shape_codes_val.shape,
            only_once_per_identity=True
        )

        print("#"*60)
        print("Loading viewer for val frames")
        print("#"*60)
        viewer_shape = ViewerShape(
            labels_val, data_dir,
            shape_decoder, shape_codes,
            num_to_eval,
            load_every,
            from_frame_id,
            res=reconstruction_res,
            max_batch=max_batch
        )
        viewer_shape.run()


