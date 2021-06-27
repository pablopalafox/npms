import sys,os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import os
import json
import torch
import argparse

from models.pose_decoder import PoseDecoder, PoseDecoderSE3
from models.shape_decoder import ShapeDecoder

import utils.nnutils as nnutils

from viz.viz_shape import ViewerShape
from viz.viz_optim import ViewerOptim

import config as cfg


if __name__ == "__main__":

    torch.backends.cudnn.benchmark = True
    
    parser = argparse.ArgumentParser(
        description='Run Model'
    )
    parser.add_argument('-d', '--dataset_type', choices=['HUMAN', 'MANO'], required=True)

    try:
        args = parser.parse_args()
    except:
        args = parser.parse_known_args()[0]
    
    dataset_type = args.dataset_type

    # ------------------------------------------- #
    print("#"*30)
    print(f"dataset_type: {dataset_type}")
    print("#"*30)
    # input("Continue?")
    # ------------------------------------------- #

    if dataset_type == "HUMAN":
        from configs_eval.config_eval_HUMAN import *
    elif dataset_type == "MANO":
        from configs_eval.config_eval_MANO import *

    # ------------------------------------------------------------------------

    data_dir = f"{ROOT}/datasets_mix"
    
    # Extract dataset name
    tmp = exp_name.split('__ON__')
    dataset_name = tmp[-1]

    
    from utils.parsing_utils import get_dataset_type_from_dataset_name
    dataset_type = get_dataset_type_from_dataset_name(dataset_name)
    splits_dir = f"{cfg.splits_dir}_{dataset_type}"

    labels_json       = os.path.join(data_dir, splits_dir, dataset_name, "labels.json")
    labels_tpose_json = os.path.join(data_dir, splits_dir, dataset_name, "labels_tpose.json")

    print("Reading from:")
    print(labels_json)
    print("Dataset name:")
    print(dataset_name)
    print()
    
    ########################################################################################################################
    ########################################################################################################################

    #######################################################################################################
    # Data
    #######################################################################################################
    with open(labels_json, "r") as f:
        labels = json.loads(f.read())

    with open(labels_tpose_json, "r") as f:
        labels_tpose = json.loads(f.read())

    num_identities = len(labels_tpose)
    num_poses = len(labels)

    print()
    print("Num identities", num_identities)
    print("Num poses", num_poses)

    ###############################################################################################################
    ###############################################################################################################

    # Load checkpoint
    exp_dir = os.path.join(exps_dir, exp_name)
    checkpoint = nnutils.load_checkpoint(exp_dir, checkpoint_epoch)    

    ########################
    # Shape decoder
    ########################
    shape_decoder = ShapeDecoder(shape_codes_dim, **shape_network_specs).cuda()
    shape_decoder.load_state_dict(checkpoint['model_state_dict_shape_decoder'])
    
    for param in shape_decoder.parameters():
        param.requires_grad = False
    
    shape_decoder.eval()
    
    nnutils.print_num_parameters(shape_decoder)

    ########################
    # Pose decoder
    ########################
    if use_se3:
        print()
        print("Using SE(3) formulation for the PoseDecoder")
        pose_decoder = PoseDecoderSE3(
            pose_codes_dim + shape_codes_dim, **pose_network_specs
        ).cuda()
    else:
        print()
        print("Using normal (translation) formulation for the PoseDecoder")
        pose_decoder = PoseDecoder(
            pose_codes_dim + shape_codes_dim, **pose_network_specs
        ).cuda()
    pose_decoder.load_state_dict(checkpoint['model_state_dict_pose_decoder'])
    
    for param in pose_decoder.parameters():
        param.requires_grad = False
    
    pose_decoder.eval()
    
    nnutils.print_num_parameters(pose_decoder)

    ########################
    # Shape codes
    ########################
    shape_codes = checkpoint['shape_codes'].cuda().detach().clone()
    shape_codes.requires_grad = False
    print("Initialized shape codes from train codes", shape_codes.shape, shape_codes.mean().item())

    assert shape_codes.shape[0] == num_identities, f"{ shape_codes.shape[0]} vs {num_identities}"

    ##################################################################
    # Pose codes
    ##################################################################

    pose_codes = checkpoint['pose_codes'].cuda().detach().clone()
    pose_codes.requires_grad = False
    print("Initialized pose codes from train codes", pose_codes.shape, pose_codes.mean().item())

    assert pose_codes.shape[0] == num_poses, f"{ pose_codes.shape[0]} vs {num_poses}"

    ##################################################################################################################
    ##################################################################################################################

    print()
    print()
    print("#######################################################################")
    print("Final visualization")
    print("#######################################################################")

    num_to_eval = 5
    reconstruction_res = 512

    with torch.no_grad():

        viewer_pose = ViewerOptim(
            labels, data_dir,
            shape_decoder, pose_decoder,
            shape_codes, pose_codes,
            num_to_eval=num_to_eval,
            load_every=load_every,
            from_frame_id=from_frame_id,
            reconstruction_res=reconstruction_res,
            input_res=encoder_res,
            max_batch=max_batch,
            warp_reconstructed_mesh=warp_reconstructed_mesh,
            exp_dir=None,
            use_pred_vertices=True,
            viz_mesh=True,
            cache_for_viz=True
        )
        viewer_pose.run()

        viewer_shape = ViewerShape(
            labels_tpose, data_dir,
            shape_decoder, shape_codes,
            num_to_eval,
            load_every,
            from_frame_id,
        )
        viewer_shape.run()
