import sys,os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import os
import numpy as np
import json
import open3d as o3d
import torch
from tqdm import tqdm
from timeit import default_timer as timer
import argparse
from datetime import datetime

from models.pose_decoder import PoseDecoder, PoseDecoderSE3
from models.shape_decoder import ShapeDecoder
import models.inference_encoder as encoders
import datasets.sdf_singleview_dataset as sdf_singleview_dataset
import utils.inference_utils as inference_utils
import utils.nnutils as nnutils
from viz.viz_shape import ViewerShape
from viz.viz_optim import ViewerOptim
import config as cfg


if __name__ == "__main__":

    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser(
        description='Run Model'
    )
    parser.add_argument('-o', '--optimize_codes', action='store_true')
    parser.add_argument('-e', '--extra_name', default="")
    parser.add_argument('-v', '--viz', action='store_true')
    parser.add_argument('-n', '--optim_name', default=None)
    parser.add_argument('-d', '--dataset_type', choices=['HUMAN', 'MANO'], required=True)

    try:
        args = parser.parse_args()
    except:
        args = parser.parse_known_args()[0]
    
    viz = False

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
    
    train_to_augmented_json = os.path.join(data_dir, splits_dir, dataset_name, "train_to_augmented.json")

    ########################################################################################################################
    ########################################################################################################################

    #######################################################################################################
    # Data
    #######################################################################################################
    with open(labels_json, "r") as f:
        labels = json.loads(f.read())

    with open(labels_tpose_json, "r") as f:
        labels_tpose = json.loads(f.read())

    train_to_augmented = None
    if os.path.isfile(train_to_augmented_json):
        with open(train_to_augmented_json, "r") as f:
            train_to_augmented = json.loads(f.read())

    batch_size = min(batch_size, len(labels))
    print("batch_size", batch_size)

    print("clamping distance", clamping_distance)

    num_identities = len(labels_tpose)
    num_frames = len(labels)

    print()
    print("#"*60)
    print("Num identities", num_identities)
    print("Num frames    ", num_frames)
    print()
    print('Interval', interval)
    print('Factor', factor)
    print()
    print("SHAPE ENCODER:", use_shape_encoder)
    print("POSE ENCODER: ", use_pose_encoder)
    print("#"*60)
    print()

    # input("Continue?")

    # Pose MLP
    exp_dir = os.path.join(exps_dir, exp_name)
    checkpoint = nnutils.load_checkpoint(exp_dir, checkpoint_epoch)    

    if use_shape_encoder:
        # Pose Encoder
        exp_dir_shape_encoder = os.path.join(exps_dir, exp_name_shape_encoder)
        checkpoint_path_shape_encoder = os.path.join(exp_dir_shape_encoder, "checkpoints", f"shape_encoder_{checkpoint_shape_encoder}.tar")
        checkpoint_shape_encoder = torch.load(checkpoint_path_shape_encoder)

    if use_pose_encoder:
        # Pose Encoder
        exp_dir_pose_encoder = os.path.join(exps_dir, exp_name_pose_encoder)
        checkpoint_path_pose_encoder = os.path.join(exp_dir_pose_encoder, "checkpoints", f"pose_encoder_{checkpoint_pose_encoder}.tar")
        checkpoint_pose_encoder = torch.load(checkpoint_path_pose_encoder)
        
    ###############################################################################################################
    ###############################################################################################################

    ########################
    # Shape decoder
    ########################
    shape_decoder = ShapeDecoder(shape_codes_dim, **shape_network_specs).cuda()
    shape_decoder.load_state_dict(checkpoint['model_state_dict_shape_decoder'])
    
    ########################
    # Pose decoder
    ########################
    if use_se3:
        print("\nUsing SE(3) formulation for the PoseDecoder")
        pose_decoder = PoseDecoderSE3(
            pose_codes_dim + shape_codes_dim, **pose_network_specs
        ).cuda()
    else:
        print("\nUsing normal (translation) formulation for the PoseDecoder")
        pose_decoder = PoseDecoder(
            pose_codes_dim + shape_codes_dim, **pose_network_specs
        ).cuda()
    pose_decoder.load_state_dict(checkpoint['model_state_dict_pose_decoder'])

    ###########################################################################################################
    # To multi-gpu
    ###########################################################################################################
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        shape_decoder = torch.nn.DataParallel(shape_decoder)
        pose_decoder  = torch.nn.DataParallel(pose_decoder)
    ###########################################################################################################
    
    ###########################################################################################################
    # Freeze decoders
    ###########################################################################################################
    for param in shape_decoder.parameters():
        param.requires_grad = False
    shape_decoder.eval()
    nnutils.print_num_parameters(shape_decoder)

    for param in pose_decoder.parameters():
        param.requires_grad = False
    pose_decoder.eval()
    nnutils.print_num_parameters(pose_decoder)
    ###########################################################################################################

    ###########################################################################################################
    ###########################################################################################################
    ###########################################################################################################
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
    ###########################################################################################################
    ###########################################################################################################
    ###########################################################################################################

    ###########################################################################################################
    ###########################################################################################################
    # Optimization 
    ###########################################################################################################
    ###########################################################################################################
    if args.optimize_codes:

        ########################
        # Shape and Pose std
        ########################
        pretrained_shape_codes = checkpoint['shape_codes'].cuda().detach().clone()
        pretrained_shape_codes.requires_grad = False
        shape_mean_channel = torch.mean(pretrained_shape_codes, dim=0)[0]
        shape_std_channel  = torch.std(pretrained_shape_codes, dim=0)[0]

        pretrained_pose_codes = checkpoint['pose_codes'].cuda().detach().clone()
        pretrained_pose_codes.requires_grad = False
        pose_mean_channel = torch.mean(pretrained_pose_codes, dim=0)[0]
        pose_std_channel  = torch.std(pretrained_pose_codes, dim=0)[0]

        train_stats = {
            's': {
                'mean': shape_mean_channel,
                'std':  shape_std_channel,
                'scale': shape_code_reg_scale,
            },
            'p': {
                'mean': pose_mean_channel,
                'std':  pose_std_channel,
            }
        }

        ########################
        # Shape Encoder
        ########################
        if use_shape_encoder:
            print()
            print("####### SHAPE ENCODER ####### ")
            shape_encoder = encoders.ShapeEncoder(
                code_dim=shape_codes_dim,
                res=encoder_res
            ).cuda()
            
            shape_encoder.load_state_dict(checkpoint_shape_encoder['model_state_dict_shape_encoder'])
            nnutils.print_num_parameters(shape_encoder)

            shape_encoder.eval()
            for param in shape_encoder.parameters():
                param.requires_grad = False

        ########################
        # Pose Encoder
        ########################
        if use_pose_encoder:
            print()
            print("####### POSE ENCODER ####### ")
            print("Using POSE encoder v3")
            pose_encoder = encoders.PoseEncoder(
                code_dim=pose_codes_dim,
                res=encoder_res
            ).cuda()
            pose_encoder.load_state_dict(checkpoint_pose_encoder['model_state_dict_pose_encoder'])
            nnutils.print_num_parameters(pose_encoder)

            pose_encoder.eval()
            for param in pose_encoder.parameters():
                param.requires_grad = False

        # print()
        # input('Continue?')

        #######################################################################################################
        # Shape Encoder
        #######################################################################################################
        shape_codes = torch.ones(len(labels_tpose), 1, shape_codes_dim).normal_(0, 1.0 / shape_codes_dim).cuda()
        assert shape_codes.shape[0] == 1, f"shape_codes.shape = {shape_codes.shape}. Cannot process multiple identities at the same time!"

        print("######")
        print("shape codes:", shape_codes.shape, shape_codes.mean().item(), shape_codes.std().item())
        print("######")
        
        if use_shape_encoder:

            #################################
            # Init codes with shape encoder
            #################################
            print()
            print("Initializing shape codes with the ShapeEncoder")

            for p in shape_encoder.parameters():
                p.requires_grad = False

            num_shape_codes_init = 0
            shape_codes_init = torch.zeros_like(shape_codes)

            for frame_i in tqdm(range(len(labels))):

                label = labels[frame_i]

                dataset, identity_name, animation_name, sample_id = \
                    label['dataset'], label['identity_name'], label['animation_name'], label['sample_id']

                # Path to ref and cur
                cur_shape_path = os.path.join(data_dir, dataset, identity_name, animation_name, sample_id)

                cur_inputs_path = os.path.join(cur_shape_path, f'partial_views/voxelized_view0_{encoder_res}res.npz')
                occupancies_cur = np.unpackbits(np.load(cur_inputs_path)['compressed_occupancies'])
                inputs_cur_np = np.reshape(occupancies_cur, (encoder_res,)*3).astype(np.float32)
                inputs_cur = torch.from_numpy(inputs_cur_np)[None, :].cuda()

                #######################################################
                # voxels_trimesh = VoxelGrid(inputs_cur_np).to_mesh()
                # voxels_mesh = o3d.geometry.TriangleMesh(
                #     o3d.utility.Vector3dVector(voxels_trimesh.vertices),
                #     o3d.utility.Vector3iVector(voxels_trimesh.faces)
                # )
                # voxels_mesh.compute_vertex_normals()
                # voxels_mesh.paint_uniform_color(Colors.green)
                # o3d.visualization.draw_geometries([voxels_mesh])
                #######################################################

                predicted_code = shape_encoder(inputs_cur).squeeze(0)
                predicted_code = predicted_code.detach().clone()
                shape_codes_init += predicted_code
                num_shape_codes_init += 1

            shape_codes = shape_codes_init / num_shape_codes_init

        if viz:
            with torch.no_grad():
                viewer_shape = ViewerShape(
                    labels, data_dir,
                    shape_decoder, shape_codes,
                    num_to_eval=1,
                    load_every=1,
                    from_frame_id=0,
                    res=reconstruction_res,
                    max_batch=max_batch,
                )
                viewer_shape.run()

        print("######")
        print("shape codes:", shape_codes.shape, shape_codes.mean().item(), shape_codes.std().item())
        print("######")

        #######################################################################################################
        # Sample points randomly around the predicted shape (we'll pass it on to SingleViewSDFDataset)
        #######################################################################################################
        samples = inference_utils.sample_points_around_tpose(
            shape_decoder, shape_codes, identity_id=0, total_sample_num=total_sample_num, sigma=sigma, viz=False
        )

        ref_sample_info = {
            'points': samples,
            'num_samples': num_samples,
            'num_samples_cur': num_samples,
        }

        #######################################################################################################
        # Flow
        #######################################################################################################

        # Data
        print("TEST DATASET...")

        test_dataset = sdf_singleview_dataset.SingleViewSDFDataset(
            data_dir=data_dir,
            labels_json=labels_json,
            batch_size=batch_size, 
            num_workers=num_workers,
            res=res_sdf[0],
            clamping_distance=clamping_distance,
            radius=radius,
            ref_sample_info=ref_sample_info,
            cache_data=cache_data,
            use_partial_input=use_partial_input,
            use_sdf_from_ifnet=use_sdf_from_ifnet,
        )
            
        shape_codes_optim_video = None
        pose_codes_optim_video = None

        ##################################################################
        # Optimize over codes
        ##################################################################

        start = timer()
        print()
        print("Optimizing over new pose codes")
        print()

        # Init pose codes randomly
        pose_codes = torch.ones(len(test_dataset), 1, pose_codes_dim).normal_(0, 0.1).cuda()

        print("Pose codes:", pose_codes.shape)
        
        #################################
        # Pose code INITIALIZATION
        #################################
        if use_pose_encoder:

            #################################
            # Init codes with pose encoder
            #################################

            print()
            print("Initializing pose codes with the PoseEncoder")

            for p in pose_encoder.parameters():
                p.requires_grad = False

            for frame_i in tqdm(range(len(labels))):

                label = labels[frame_i]

                dataset, identity_name, animation_name, sample_id = \
                    label['dataset'], label['identity_name'], label['animation_name'], label['sample_id']

                # Path to ref and cur
                cur_shape_path = os.path.join(data_dir, dataset, identity_name, animation_name, sample_id)

                cur_inputs_path = os.path.join(cur_shape_path, f'partial_views/voxelized_view0_{encoder_res}res.npz')
                occupancies_cur = np.unpackbits(np.load(cur_inputs_path)['compressed_occupancies'])
                inputs_cur = np.reshape(occupancies_cur, (encoder_res,)*3).astype(np.float32)
                inputs_cur = torch.from_numpy(inputs_cur)[None, :].cuda()

                predicted_code = pose_encoder(inputs_cur).squeeze(0)
                predicted_code = predicted_code.detach().clone()
                pose_codes[frame_i, ...] = predicted_code

        
        elif init_from_pretrained_pose_codes:

            #################################
            # Init from training codes
            #################################
            
            pretrained_pose_codes = checkpoint['pose_codes'].cuda()

            if init_from_mean:
                # Init with mean
                mean_pose_code = pretrained_pose_codes.mean(dim=0).unsqueeze(0)
                mean_pose_code = mean_pose_code.expand(pose_codes.shape[0], -1, -1)
                print(f"Init from mean pose code channels (before: mean -> {torch.mean(pose_codes)} | std -> {torch.std(pose_codes)})")
                pose_codes = mean_pose_code.clone().detach()
                print(f"Init from mean pose code channels (after: mean -> {torch.mean(pose_codes)} | std -> {torch.std(pose_codes)})")

            else:
                print("Init from train pose codes themselves")
                if pose_codes.shape[0] != pretrained_pose_codes.shape[0]:
                    pose_codes = pretrained_pose_codes[:len(test_dataset)].detach().clone()
                    if len(pose_codes.shape) == 2:
                        pose_codes = pose_codes.unsqueeze(0)
                else:
                    pose_codes = pretrained_pose_codes.detach().clone()

            if add_noise_to_initialization:
                print()
                print(f"Adding noise to pose codes (before: mean -> {torch.mean(pose_codes)} | std -> {torch.std(pose_codes)})")
                pose_codes = pose_codes + sigma_noise * torch.randn_like(pose_codes)
                print(f"Adding noise to pose codes (after:  mean -> {torch.mean(pose_codes)} | std -> {torch.std(pose_codes)})")

            assert pose_codes.shape[1] == 1 and pose_codes.shape[2] == pose_codes_dim
        
        print("Codes initialized -", pose_codes.shape)
        pose_codes.requires_grad = True
        assert pose_codes.requires_grad

        pose_codes_optim_video = pose_codes

        if viz:
            viewer = ViewerOptim(
                labels, data_dir,
                shape_decoder, pose_decoder,
                shape_codes, pose_codes_optim_video,
                num_to_eval=num_to_eval,
                load_every=load_every,
                from_frame_id=from_frame_id,
                reconstruction_res=reconstruction_res,
                input_res=encoder_res,
                max_batch=max_batch,
                warp_reconstructed_mesh=warp_reconstructed_mesh,
                viz_mesh=True,
                cache_for_viz=True
            )
            viewer.run()

        # exit()

        #################################
        # Find codes
        #################################

        if do_optim:

            print()
            print("num_iterations", num_iterations)
            print("code_snapshot_every", code_snapshot_every)
            # if batch_size != 4: input("batch size is not 4 - try to use 4 or higher if possible")
            print()

            ########################################################################################################################
            # Create output dir
            ########################################################################################################################
            date_optim = datetime.now().strftime('%Y-%m-%d')
            optim_name = f"{date_optim}__{dataset_name}"
            optim_name = f"{optim_name}__bs{batch_size}"
            optim_name = f"{optim_name}__icp{optim_dict['icp']['lambda']}-{optim_dict['icp']['iters']}"
            optim_name = f"{optim_name}__itrs{num_iterations}"
            optim_name = f"{optim_name}__sreg{code_reg_lambdas['shape']}_preg{code_reg_lambdas['pose']}"
            optim_name = f"{optim_name}__slr{lr_dict['shape_codes']}_plr{lr_dict['pose_codes']}"
            optim_name = f"{optim_name}__interv{interval}_factr{factor}"
            optim_name = f"{optim_name}__clamp{clamping_distance}"
            optim_name = f"{optim_name}__sigma{sigma}"
            optim_name = f"{optim_name}__tmpreg{temporal_reg_lambda}"
            optim_name = f"{optim_name}__codecon{code_consistency_lambda}"
            optim_name = f"{optim_name}__cpt{checkpoint_epoch}"
            # optim_name = f"{optim_name}__wALT" if alternate_shape_pose else f"{optim_name}__woALT"
            # optim_name = f"{optim_name}__wClampRed" if reduce_clamping else f"{optim_name}__woClampRed"
            # optim_name = f"{optim_name}__part" if use_partial_input else f"{optim_name}__comp"
            # optim_name = f"{optim_name}__nSE" if not use_shape_encoder else optim_name
            # optim_name = f"{optim_name}__nPE" if not use_pose_encoder else optim_name

            if not use_partial_input:
                optim_name = f"{optim_name}IFNet" if use_sdf_from_ifnet else f"{optim_name}GT"

            if args.extra_name != "":
                optim_name = f"{optim_name}__{args.extra_name}"
            
            print()
            print(optim_name)
            # input("CONTINUE?")

            optimization_dir = os.path.join(exp_dir, "optimization", optim_name)
            if not os.path.isdir(optimization_dir):
                os.makedirs(optimization_dir)

            # Save the options file
            import shutil
            if dataset_type == "HUMAN":
                import configs_eval.config_eval_HUMAN as eval_cfg
                eval_cfg_path = eval_cfg.__file__
                shutil.copy(eval_cfg_path, os.path.join(optimization_dir, "human_config.py"))
            elif dataset_type == "MANO":
                import configs_eval.config_eval_MANO as eval_cfg
                eval_cfg_path = eval_cfg.__file__
                shutil.copy(eval_cfg_path, os.path.join(optimization_dir, "mano_config.py"))

            ########################################################################################################################

            if freeze_shape_codes:
                shape_codes.requires_grad = False
            else:
                shape_codes.requires_grad = True

            print("shape_codes mean", shape_codes.mean().item(), shape_codes.std().item())
            print(shape_codes.requires_grad)
            
            # Optimize
            optimized_codes = inference_utils.optimize_over_all_codes(
                ref_sample_info,
                shape_decoder, pose_decoder,
                shape_codes, pose_codes,
                test_dataset=test_dataset,
                res_sdf=res_sdf,
                clamping_distance=clamping_distance,
                reduce_clamping=reduce_clamping,
                num_iterations=num_iterations,
                pose_bootstrap_iterations=pose_bootstrap_iterations,
                alternate_shape_pose=alternate_shape_pose,
                code_snapshot_every=code_snapshot_every,
                learning_rate_schedules=learning_rate_schedules,
                optim_dict=optim_dict,
                train_stats=train_stats,
                temporal_reg_lambda=temporal_reg_lambda,
                code_consistency_lambda=code_consistency_lambda,
                use_partial_input=use_partial_input,
                shuffle=shuffle
            )

            shape_codes, shape_codes_optim_video, pose_codes, pose_codes_optim_video = optimized_codes

            print("shape_codes mean", shape_codes.mean().item(), shape_codes.std().item())
            print(shape_codes.requires_grad)

            assert shape_codes_optim_video.shape[1] == pose_codes_optim_video.shape[1]

        #######################################################################################################
        # Save codes
        #######################################################################################################
        print()
        print("Saving codes at:", optimization_dir)
        print()
        print("optim_name:", optim_name)

        optimized_codes_path = os.path.join(optimization_dir, "codes.npz")
        np.savez_compressed(optimized_codes_path, shape=shape_codes_optim_video, pose=pose_codes_optim_video)

        shape_codes_optim_video = torch.from_numpy(shape_codes_optim_video).cuda()
        pose_codes_optim_video  = torch.from_numpy(pose_codes_optim_video).cuda()

        # Generate meshes for later error computation
        with torch.no_grad():
            viewer = ViewerOptim(
                labels, data_dir,
                shape_decoder, pose_decoder,
                shape_codes_optim_video, pose_codes_optim_video,
                num_to_eval=num_to_eval,
                load_every=load_every,
                from_frame_id=from_frame_id,
                reconstruction_res=reconstruction_res,
                input_res=encoder_res,
                max_batch=max_batch,
                warp_reconstructed_mesh=warp_reconstructed_mesh,
                exp_dir=optimization_dir,
                viz_mesh=True,
                cache_for_viz=False
            )
        
        ##################################################################################################################
        ##################################################################################################################

    elif args.optim_name is not None:

        optim_name = args.optim_name

        print("###############################################")
        print("Reading from:")
        print("exp_dir", exp_dir)
        print("run_dir", args.optim_name)
        print("###############################################")

        assert dataset_name in optim_name, "make sure you're evaluating on the same dataset you optimized over!"

        optimization_dir = os.path.join(exp_dir, "optimization", optim_name)
        optimized_codes_path = os.path.join(optimization_dir, "codes.npz")
        
        assert os.path.isfile(optimized_codes_path), f"File {optimized_codes_path} does not exist!"

        # Maybe visualize results
        codes_npz = np.load(optimized_codes_path)
        shape_codes_optim_video = torch.from_numpy(codes_npz['shape']).cuda()
        pose_codes_optim_video  = torch.from_numpy(codes_npz['pose']).cuda()
        
        print()
        print()
        print("#######################################################################")
        print("Final visualization")
        print("#######################################################################")

        with torch.no_grad():

            viewer = ViewerOptim(
                labels, data_dir,
                shape_decoder, pose_decoder,
                shape_codes_optim_video, pose_codes_optim_video,
                num_to_eval=num_to_eval,
                load_every=load_every,
                from_frame_id=from_frame_id,
                reconstruction_res=reconstruction_res,
                input_res=encoder_res,
                max_batch=max_batch,
                warp_reconstructed_mesh=warp_reconstructed_mesh,
                exp_dir=optimization_dir,
                cache_for_viz=args.viz
            )
        
        if args.viz:
            viewer.run()

            # viewer_shape = ViewerShape(
            #     labels_tpose, data_dir,
            #     shape_decoder, shape_codes,
            #     num_to_eval,
            #     load_every,
            #     from_frame_id,
            # )
            # viewer_shape.run()

    else:
        raise Exception("Specify either -o or -n")


    print()
    print("optim_name")
    print(optim_name)
    print()

    ######################################################################################################
    # Compute metrics
    ######################################################################################################
    from compute_errors import compute_tracking_error, compute_reconstruction_error 
    
    predicted_meshes_dir = os.path.join(optimization_dir, f"predicted_meshes_{reconstruction_res}res")
    assert os.path.isdir(predicted_meshes_dir), predicted_meshes_dir

    sample_num = 100000
    knn = 1
    coverage = 0.005 # np.max(dist)
    keyframe_every = min(50, len(labels))

    # Tracking error
    epe3d_avg = compute_tracking_error(
        data_dir, labels, predicted_meshes_dir,
        sample_num, knn, coverage,
        keyframe_every
    )
    # Reconstruction error
    iou_avg, chamfer_avg, accuracy_avg, completeness_avg, normal_consis_avg = compute_reconstruction_error(
        data_dir, labels, predicted_meshes_dir
    )

    print()
    print("#"*60)
    print("iou_avg:           {}".format(iou_avg))
    print("chamfer_avg:       {}".format(chamfer_avg))
    print("accuracy_avg:      {}".format(accuracy_avg))
    print("completeness_avg:  {}".format(completeness_avg))
    print("normal_consis_avg: {}".format(normal_consis_avg))
    print("epe3d_avg:         {}".format(epe3d_avg))
    print("#"*60)

    print()
    print("exp_dir")
    print(exp_dir)
    print()
    print("optim_name")
    print(optim_name)
    print()

    # Write results
    import csv
    results_summary_path = os.path.join(optimization_dir, f"results_summary__{sample_num}samples__{knn}knn__{coverage}coverage__{keyframe_every}kf_every.csv")
    with open(results_summary_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["iou_avg", iou_avg])
        csvwriter.writerow(["chamfer_avg", chamfer_avg])
        csvwriter.writerow(["normal_consis_avg", normal_consis_avg])
        csvwriter.writerow(["epe3d_avg", epe3d_avg])