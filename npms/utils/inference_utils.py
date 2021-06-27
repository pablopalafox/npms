import os, sys
import trimesh
import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm
from timeit import default_timer as timer
import open3d as o3d

# from chamferdist import ChamferDistance
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import external.ChamferDistancePytorch.chamfer3D.dist_chamfer_3D as dist_chamfer_3D

import data_processing.implicit_waterproofing as iw
import utils.deepsdf_utils as deepsdf_utils
import utils.nnutils as nnutils

import config as cfg

"""
Optimize over shape codes
"""
def optimize_over_shape_codes(
    sample_info,
    shape_decoder,
    shape_codes,
    test_dataset,
    num_iterations=800,
    lr=1e-2,
    decreased_by=2,
    adjust_lr_every=10,
    code_reg_lambda=1e-4,
    code_bound=None,
    train_stats={},
    shuffle=True
):

    def adjust_learning_rate(initial_lr, optimizer, num_iterations, decreased_by, adjust_lr_every):
        lr = initial_lr * ((1 / decreased_by) ** (num_iterations // adjust_lr_every))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        return lr

    for p in shape_decoder.parameters():
        p.requires_grad = False

    assert not shape_decoder.training
    assert shape_codes.requires_grad

    print()
    print(f"Will decrease lr by {decreased_by} every {adjust_lr_every} iterations")

    learnable_params = []
    learnable_params.append(shape_codes)
    optimizer = torch.optim.Adam(learnable_params, lr=lr)

    # Code dims    
    shape_codes_dim = shape_codes.shape[-1]

    batch_size = test_dataset.get_batch_size()

    enforce_minmax = True
    min_vec = torch.ones(sample_info['sdf']['num_points'] * batch_size, 1).cuda() * (-cfg.clamping_distance)
    max_vec = torch.ones(sample_info['sdf']['num_points'] * batch_size, 1).cuda() * (cfg.clamping_distance)

    loss_l1 = deepsdf_utils.SoftL1()

    #####################################################################################
    # Optimization
    #####################################################################################

    for e in tqdm(range(num_iterations)):

        # Update learning rate
        current_lr = adjust_learning_rate(lr, optimizer, e, decreased_by, adjust_lr_every)

        test_data_loader = test_dataset.get_loader(shuffle=shuffle)

        loss_recon_batch = []
        code_reg_batch = []

        for _, batch in enumerate(test_data_loader):

            # print("batch_i", batch_i)

            ################################################
            # Set gradients to None
            ################################################
            # Latent codes
            shape_codes.grad = None
            ################################################

            #####################################################################################
            # Prepare data
            #####################################################################################
            ref = batch.get('ref')
            identity_ids = batch.get('identity_id')

            p_ref = ref['points_sdf'].cuda() # [bs, N, 3]
    
            batch_size = p_ref.shape[0]
            num_points = p_ref.shape[1]

            sdf_data = p_ref.reshape(num_points * batch_size, 4)
            p_ref_flat  = sdf_data[:, 0:3]
            sdf_gt      = sdf_data[:, 3].unsqueeze(1)

            #####################################################################################
            # Forward pass
            #####################################################################################
            assert torch.all(identity_ids < shape_codes.shape[0]), f"{identity_ids} vs {shape_codes.shape[0]}"
            shape_codes_batch = shape_codes[identity_ids, ...] # [bs, 1, C]

            # Extent latent code to all sampled points
            shape_codes_repeat = shape_codes_batch.expand(-1, num_points, -1) # [bs, N, C]
            shape_codes_inputs = shape_codes_repeat.reshape(-1, shape_codes_dim) # [bs*N, C]

            shape_inputs = torch.cat([shape_codes_inputs, p_ref_flat], 1)

            # Truncate groundtruth sdf
            if enforce_minmax:
                sdf_gt = deepsdf_utils.threshold_min_max(sdf_gt, min_vec, max_vec)

            # Predict SDF
            pred_sdf = shape_decoder(shape_inputs)  # [bs*N, 1]

            # Truncate groundtruth sdf
            if enforce_minmax:
                pred_sdf = deepsdf_utils.threshold_min_max(pred_sdf, min_vec, max_vec)

            # Loss
            loss, _ = loss_l1(pred_sdf, sdf_gt)
            loss *= (1 + 0.5 * torch.sign(sdf_gt) * torch.sign(sdf_gt - pred_sdf))
            loss_ref = torch.mean(loss)

            # Add to sum of losses
            loss = 0.0
            loss += loss_ref

            loss_recon_batch.append(loss_ref.item())

            # Code regularization
            if code_reg_lambda is not None:
                # print("Normal reg")
                l2_shape_loss = deepsdf_utils.latent_size_regul(shape_codes, identity_ids.numpy())
                loss += code_reg_lambda * l2_shape_loss
                
                code_reg_batch.append(code_reg_lambda * l2_shape_loss.item())
            else:
                # print("Train reg")
                l2_shape_loss = deepsdf_utils.latent_size_regul(
                    shape_codes, identity_ids.numpy(),
                    train_stats['s']['mean'], train_stats['s']['std']
                )
                l2_shape_loss = train_stats['s']['scale'] * l2_shape_loss
                loss += l2_shape_loss
                code_reg_batch.append(l2_shape_loss.item())

            # Backward pass
            loss.backward()
            optimizer.step()

            # Project code to sphere
            if code_bound is not None:
                deepsdf_utils.project_latent_codes_onto_sphere(shape_codes, code_bound)

        if e % 50 == 0:
            loss_recon_avg   = sum(loss_recon_batch) / len(loss_recon_batch) if len(loss_recon_batch) > 0 else -1
            code_reg_avg     = sum(code_reg_batch) / len(code_reg_batch) if len(code_reg_batch) > 0 else -1
            print(f"iter {e}: {loss_recon_avg:.8f} | {code_reg_avg:.8f} | lr: {current_lr}")

    loss_recon_avg   = sum(loss_recon_batch) / len(loss_recon_batch) if len(loss_recon_batch) > 0 else -1
    code_reg_avg     = sum(code_reg_batch) / len(code_reg_batch) if len(code_reg_batch) > 0 else -1
    print(f"iter {e}: {loss_recon_avg:.8f} | {code_reg_avg:.8f} | lr: {current_lr}")
    print()
    print("Optimization done!")
    print()

    return shape_codes

"""
Reconstruct shape SDF
"""
def reconstruct_shape_sdf(
    reconstruct_resolution, batch_points,
    shape_decoder, shape_codes, identity_ids, 
    shape_codes_dim
):
    import mcubes

    min = -0.5
    max = 0.5

    shape_codes.requires_grad = False

    # Get shape codes for batch samples
    shape_codes_batch = shape_codes[identity_ids, ...] # [bs, 1, C]
    assert shape_codes_batch.shape[1] == 1, shape_codes_batch.shape

    grid_points = iw.create_grid_points_from_bounds(min, max, reconstruct_resolution)
    grid_points = torch.from_numpy(grid_points.astype(np.float32)).cuda()
    grid_points = grid_points[None, ...]
    # print(torch.max(grid_points), torch.min(grid_points))

    # print(grid_points)
    grid_points_split = torch.split(grid_points, batch_points, dim=1)

    # Logits
    sdf_list = []
    for points in grid_points_split:

        points = points.squeeze(0)
        num_points = points.shape[0]

        with torch.no_grad():
            # Extent latent code to all sampled points
            shape_codes_repeat = shape_codes_batch.expand(-1, num_points, -1) # [bs, N, C]
            shape_codes_inputs = shape_codes_repeat.reshape(-1, shape_codes_dim) # [bs*N, C]

            shape_inputs = torch.cat([shape_codes_inputs, points], 1)

            pred_sdf = shape_decoder(shape_inputs)        # [bs*N, 1]

        sdf_list.append(pred_sdf.squeeze(0).detach().cpu())

    sdfs = torch.cat(sdf_list, dim=0)
    sdfs = sdfs.numpy()

    # Mesh
    sdfs = np.reshape(sdfs, (reconstruct_resolution,) * 3)

    threshold = 0.0
    vertices, faces = mcubes.marching_cubes(sdfs, threshold)

    # remove translation due to padding
    # vertices -= 1

    # rescale to original scale
    step = (max - min) / (reconstruct_resolution - 1)
    vertices = np.multiply(vertices, step)
    vertices += [min, min, min]

    return trimesh.Trimesh(vertices, faces)


"""
Optimize over all codes in a sequence
"""
def optimize_over_all_codes(
    sample_info,
    shape_decoder, pose_decoder,
    shape_codes_tmp, pose_codes_tmp,
    test_dataset,
    res_sdf,
    clamping_distance=0.1,
    reduce_clamping=False,
    num_iterations=800,
    pose_bootstrap_iterations=None,
    alternate_shape_pose=False,
    code_snapshot_every=10,
    learning_rate_schedules={},
    optim_dict={},
    train_stats={},
    temporal_reg_lambda=1e2,
    code_consistency_lambda=1e2,
    use_partial_input=True,
    shuffle=True
):
    print_loss_every = min(num_iterations // 10, 50)

    print()

    lr_schedule_shape = nnutils.StepLearningRateSchedule(learning_rate_schedules['shape_codes'])
    lr_schedule_pose  = nnutils.StepLearningRateSchedule(learning_rate_schedules['pose_codes'])
    
    lr_schedules = [
        lr_schedule_shape,
        lr_schedule_pose,
    ]

    for p in shape_decoder.parameters():
        p.requires_grad = False

    for p in pose_decoder.parameters():
        p.requires_grad = False

    assert not shape_decoder.training
    assert not pose_decoder.training

    # Shape codes
    shape_codes = torch.ones_like(shape_codes_tmp).cuda()
    shape_codes = shape_codes_tmp.detach().clone()
    if shape_codes_tmp.requires_grad:
        shape_codes.requires_grad = True

    # Pose codes
    pose_codes = torch.ones_like(pose_codes_tmp).cuda()
    pose_codes = pose_codes_tmp.detach().clone()
    if pose_codes_tmp.requires_grad:
        pose_codes.requires_grad = True

    learnable_params = []
    learnable_params.append(shape_codes)
    learnable_params.append(pose_codes)

    learnable_params = [
        {
            "params": shape_codes,
            "lr": lr_schedule_shape.get_learning_rate(0),
        },
        {
            "params": pose_codes,
            "lr": lr_schedule_pose.get_learning_rate(0),
        },
    ]

    optimizer = torch.optim.Adam(learnable_params)

    # Code dims    
    shape_codes_dim = shape_codes.shape[-1]
    pose_codes_dim  = pose_codes.shape[-1]

    if not shape_codes.requires_grad:
        print()
        # input("Shape codes don't require grad. Continue?")
        print("Shape codes don't require grad. Continue?")
    if not pose_codes.requires_grad:
        print()
        # input("Pose codes don't require grad. Continue?")
        print("Pose codes don't require grad. Continue?")

    # Store codes for each iteration for later visualization
    shape_codes_optim_video = []
    pose_codes_optim_video  = []

    #####################################################################################
    # Optimization
    #####################################################################################
    loss_l1 = deepsdf_utils.SoftL1()
    chamfer_dist = dist_chamfer_3D.chamfer_3DDist()

    batch_size = test_dataset.get_batch_size()

    enforce_minmax = True # only for sdf
    clamping_distance = clamping_distance
    min_vec = torch.ones(sample_info['num_samples'] * batch_size, 1).cuda() * (-clamping_distance)
    max_vec = torch.ones(sample_info['num_samples'] * batch_size, 1).cuda() * (clamping_distance)
    
    print()
    print("#"*60)
    print("clamping distance", clamping_distance)
    print("#"*60)
    print()

    # Coarse-to-fine sections
    interval = int(num_iterations / len(res_sdf)) 
    coarse2fine_epochs = [interval * (i+1) for i in range(len(res_sdf))]
    res = res_sdf[0]

    ################################################################################################
    # Go over epochs
    ################################################################################################
    for e in tqdm(range(num_iterations)):

        if pose_bootstrap_iterations is not None:
            if e < pose_bootstrap_iterations:
                print("Bootstrapping pose...")
                shape_codes.requires_grad = False
            else: 
                shape_codes.requires_grad = True

        if e > (num_iterations // 1.5) and alternate_shape_pose:
            print("Alternating shape and pose optim!")
            
            if e % 2 == 0:
                shape_codes.requires_grad = False
                pose_codes.requires_grad  = True
            else:
                shape_codes.requires_grad = True
                pose_codes.requires_grad  = False

        if e > (num_iterations // 2) and reduce_clamping:
            clamping_distance = clamping_distance / 2
            min_vec = torch.ones(sample_info['num_samples'] * batch_size, 1).cuda() * (-clamping_distance)
            max_vec = torch.ones(sample_info['num_samples'] * batch_size, 1).cuda() * (clamping_distance)
            print("clamp", clamping_distance)

        # Update learning rate
        nnutils.adjust_learning_rate(lr_schedules, optimizer, e)

        # Update resolution of the input SDF grid
        if e < coarse2fine_epochs[0]:
            res = res_sdf[0]
        elif e < coarse2fine_epochs[1]:
            res = res_sdf[1]
        elif e < coarse2fine_epochs[2]:
            res = res_sdf[2]
        elif e < coarse2fine_epochs[3]:
            res = res_sdf[3]

        # Get data loader
        test_data_loader = test_dataset.get_loader(shuffle=shuffle, res=res)

        loss_recon_batch = []
        loss_recon_batch_partial = []
        loss_icp_batch = []
        shape_code_reg_batch = []
        pose_code_reg_batch = []
        temporal_reg_batch = []
        code_consistency_batch = []

        # Go over every batch in the dataset
        for batch_i, batch in enumerate(test_data_loader):

            #####################################################################################
            # Freeze codes
            #####################################################################################
            if e < (num_iterations // 2) and alternate_shape_pose:
            # if alternate_shape_pose:
                if batch_i % 2 == 0:
                    shape_codes.requires_grad = False
                    pose_codes.requires_grad  = True
                else:
                    shape_codes.requires_grad = True
                    pose_codes.requires_grad  = False
                # print(f"Alternating!!! shape {shape_codes.requires_grad} - pose {pose_codes.requires_grad}")
            else:
                shape_codes.requires_grad = True
                pose_codes.requires_grad  = True
            
            # print("batch_i", batch_i)

            #####################################################################################
            # Prepare data
            #####################################################################################
            ref = batch.get('ref')
            curs = batch.get('curs')
            indices_list = batch.get('idxs')
            identity_ids = batch.get('identity_id')

            # The points in the reference frame
            p_ref = ref['points'].cuda() # [bs, N, 3]

            # Reshape inputs
            p_ref_flat = p_ref.reshape(-1, 3) # [bs*N, 3]

            batch_size = p_ref.shape[0]
            num_samples_per_shape = p_ref.shape[1]

            ################################################
            # Prepare shape codes
            ################################################
            # Get shape codes for batch samples
            assert torch.all(identity_ids < shape_codes.shape[0]), f"{identity_ids} vs {shape_codes.shape[0]}"
            shape_codes_batch = shape_codes[identity_ids, ...] # [bs, 1, C]
            
            # Extent latent code to all sampled points
            shape_codes_repeat = shape_codes_batch.expand(-1, num_samples_per_shape, -1) # [bs, N, C]
            shape_codes_inputs = shape_codes_repeat.reshape(-1, shape_codes_dim) # [bs*N, C]

            ################################################
            # Predict sdf of p_ref using the shape codes
            ################################################
            shape_inputs = torch.cat([shape_codes_inputs, p_ref_flat], 1)
            sdf_ref = shape_decoder(shape_inputs)  # [bs*N, 1]

            ################################################
            # For doing ICP later, compute a mask for near the surface points (ns == near surface)
            ################################################
            if optim_dict['icp']['lambda'] > 0:
                ns_mask_flat = (sdf_ref < optim_dict['icp']['ns_eps']) & (sdf_ref > -optim_dict['icp']['ns_eps'])
                ns_mask = ns_mask_flat.reshape(batch_size, -1)

            # The input SDF grids of the sequence - we'll optimize our shape / pose codes such that they fit the SDF grid
            inputs_complete_curs = []
            inputs_partial_curs = []
            mask_partial_curs = []
            for cur in curs:
                # print(cur['inputs'].shape)
                mask_partial_curs.append(cur['grid_mask'].unsqueeze(1).cuda())
                
                if use_partial_input:
                    inputs_partial_curs.append(cur['inputs_psdf'].unsqueeze(1).cuda())
                else:
                    inputs_complete_curs.append(cur['inputs_sdf'].unsqueeze(1).cuda())

            # The input point cloud of the sequence
            p_curs = []
            for cur in curs:
                p_curs.append(cur['points_cur'])
                # print(cur['points_cur'].shape)

            ################################################
            # Set gradients to None
            ################################################
            # Latent codes
            if shape_codes.requires_grad:
                shape_codes.grad = None
            if shape_codes.requires_grad:
                pose_codes.grad  = None
            ################################################

            # Loss
            loss = 0.0

            ################################################
            # Go over each frame in the list of pairs of the current batch element
            # batch:
            #     batch_sample: [frame_i, frame_i + 1]
            #     batch_sample: [frame_j, frame_j + 1]
            ################################################
            deltas = []
            for frame_i, indices in enumerate(indices_list):

                # print("\t", frame_i, indices)

                # Input point cloud of the current frames in the batch
                p_cur = p_curs[frame_i].cuda()
                p_cur.requires_grad = False

                ############################
                # Prepare pose codes
                ############################
                # Get shape codes for batch samples
                assert torch.all(indices < pose_codes.shape[0]), f"{indices} vs {pose_codes.shape[0]}"
                pose_codes_batch = pose_codes[indices, ...] # [bs, 1, C]
                # Extent latent code to all sampled points
                pose_codes_repeat = pose_codes_batch.expand(-1, num_samples_per_shape, -1) # [bs, N, C]
                pose_codes_inputs = pose_codes_repeat.reshape(-1, pose_codes_dim) # [bs*N, C]

                # Concatenate pose and shape codes
                shape_pose_codes_inputs = torch.cat([shape_codes_inputs, pose_codes_inputs], 1)

                # Concatenate (for each sample point), the corresponding code and the p_cur coords
                pose_inputs = torch.cat([shape_pose_codes_inputs, p_ref_flat], 1) 

                ########################################################
                # Warp points in ref to cur
                ########################################################
                p_ref_warped_flat, delta_ref_to_cur = pose_decoder(pose_inputs) # [bs*N, 3]
                p_ref_warped = p_ref_warped_flat.reshape(batch_size, -1, 3)
                deltas.append(delta_ref_to_cur) # store for regularization

                ########################################################
                # Get the SDF values for p_ref_warped
                ########################################################
                # First, prepare the coordinates such that they are in the form required by the 'grid_sample' function
                p_ref_warped_coords = p_ref_warped.clone()
                p_ref_warped_coords[..., 0], p_ref_warped_coords[..., 2] = p_ref_warped[..., 2], p_ref_warped[..., 0]
                p_ref_warped_coords = 2.0 * p_ref_warped_coords

                p_ref_warped_coords_grd = p_ref_warped_coords.unsqueeze(1).unsqueeze(1)
                
                # Mask
                mask_partial_cur = mask_partial_curs[frame_i]
                sdf_cur_mask = F.grid_sample(mask_partial_cur, p_ref_warped_coords_grd, padding_mode='border', align_corners=False)    # [bs, 1, 1, 1, N]
                sdf_cur_mask = sdf_cur_mask.squeeze(1).squeeze(1).squeeze(1) # bs, N
                sdf_cur_mask = (sdf_cur_mask > 0.999999).type(torch.float32)

                # SDF
                if use_partial_input:
                    inputs_partial_cur = inputs_partial_curs[frame_i]
                    sdf_cur = F.grid_sample(inputs_partial_cur, p_ref_warped_coords_grd, padding_mode='border', align_corners=False)    # [bs, 1, 1, 1, N]
                    sdf_cur = sdf_cur.squeeze(1).squeeze(1).squeeze(1) # bs, N
                else:
                    inputs_complete_cur = inputs_complete_curs[frame_i]
                    sdf_cur = F.grid_sample(inputs_complete_cur, p_ref_warped_coords_grd, padding_mode='border', align_corners=False)    # [bs, 1, 1, 1, N]

                sdf_cur      = sdf_cur.reshape(-1, 1)
                sdf_cur_mask = sdf_cur_mask.reshape(-1, 1)

                ############################
                # Reconstruction Loss
                ############################
                # Truncate groundtruth sdf
                if enforce_minmax:
                    sdf_ref = deepsdf_utils.threshold_min_max(sdf_ref, min_vec, max_vec)

                # Loss
                loss_recon, _ = loss_l1(sdf_cur, sdf_ref)
                assert torch.all(torch.isfinite(loss_recon))
                loss_recon *= (1 + 0.5 * torch.sign(sdf_ref) * torch.sign(sdf_ref - sdf_cur))

                loss_recon_partial = sdf_cur_mask * loss_recon
                loss_recon_partial = loss_recon_partial.sum() / sdf_cur_mask.sum()
                loss_recon_batch_partial.append(loss_recon_partial)

                if use_partial_input:
                    loss_recon_mean = loss_recon_partial
                else:
                    loss_recon_mean = loss_recon.mean()
                
                loss += loss_recon_mean

                loss_recon_batch.append(loss_recon_mean)

                ############################
                # ICP Loss (go over each frame in the batch separately)
                ############################
                loss_icp = 0
                if optim_dict['icp']['lambda'] > 0 and e < optim_dict['icp']['iters']:
                    for frame_j in range(batch_size):
                        
                        p_cur_j        = p_cur[frame_j]
                        p_ref_warped_j = p_ref_warped[frame_j]
                        ns_mask_j      = ns_mask[frame_j]

                        # Filter those transformed points that were near the surface in the tpose
                        p_ref_warped_ns_j = p_ref_warped_j[ns_mask_j] # [NS, 3], with NS the number of near surface points

                        # For each point in the input point cloud, find its nn in p_ref_warped_ns_flat
                        dist, _, _, _ = chamfer_dist(p_cur_j.unsqueeze(0), p_ref_warped_ns_j.unsqueeze(0))

                        loss_icp += torch.mean(dist)

                    loss_icp_mean = loss_icp / batch_size
                    loss_icp_mean = optim_dict['icp']['lambda'] * loss_icp_mean
                    loss += loss_icp_mean
                    loss_icp_batch.append(loss_icp_mean)

                ############################
                # Code Reg 
                ############################
                if shape_codes.requires_grad:
                    shape_code_reg_lambda = optim_dict['s']['code_reg_lambda']
                    
                    # Code regularization
                    if shape_code_reg_lambda is not None:
                        # print("Normal reg")
                        l2_shape_loss = deepsdf_utils.latent_size_regul(shape_codes, identity_ids.numpy())
                        loss += shape_code_reg_lambda * l2_shape_loss
                        
                        shape_code_reg_batch.append(shape_code_reg_lambda * l2_shape_loss.item())
                    else:
                        # print("Train reg")
                        l2_shape_loss = deepsdf_utils.latent_size_regul(
                            shape_codes, identity_ids.numpy(),
                            train_stats['s']['mean'], train_stats['s']['std']
                        )
                        l2_shape_loss = train_stats['s']['scale'] * l2_shape_loss
                        loss += l2_shape_loss
                        shape_code_reg_batch.append(l2_shape_loss.item())

                pose_code_reg_lambda = optim_dict['p']['code_reg_lambda']
                if pose_code_reg_lambda is not None and pose_codes.requires_grad:
                    l2_pose_loss = deepsdf_utils.latent_size_regul(pose_codes, indices.numpy())
                    loss += pose_code_reg_lambda * l2_pose_loss
                    pose_code_reg_batch.append(pose_code_reg_lambda * l2_pose_loss.item())

            ############################
            # Temporal Reg 
            ############################
            if temporal_reg_lambda > 0.0:
                assert len(deltas) == 2, "For temporal regularization we need more that 1 frame. Set radius=1"
                temporal_reg = F.smooth_l1_loss(deltas[0], deltas[1])
                temporal_reg_batch.append(temporal_reg_lambda * temporal_reg.item())
                loss += temporal_reg_lambda * temporal_reg

            ############################
            # Temporal Reg (Code Consistency)
            ############################
            if code_consistency_lambda > 0.0:
                assert len(indices_list) == 2, "For temporal regularization we need more that 1 frame. Set radius=1 when creating the dataset"
                
                pose_codes_current = pose_codes[indices_list[0]]
                pose_codes_next    = pose_codes[indices_list[1]]

                code_consistency = F.smooth_l1_loss(pose_codes_current, pose_codes_next)
                code_consistency_batch.append(code_consistency_lambda * code_consistency.item())
                loss += code_consistency_lambda * code_consistency

            # Backward pass
            loss.backward()

            assert torch.isfinite(loss), loss

            if True:
                grad_sum = 0.0
                if pose_codes.requires_grad:
                    grad_sum += torch.sum(torch.abs(pose_codes.grad))
                if shape_codes.requires_grad:
                    grad_sum += torch.sum(torch.abs(shape_codes.grad))
                assert torch.isfinite(grad_sum), grad_sum

            # Take step
            optimizer.step()

            # Project code to sphere
            shape_code_bound = optim_dict['s']['code_bound']
            if shape_code_bound is not None:
                deepsdf_utils.project_latent_codes_onto_sphere(shape_codes, shape_code_bound)

            pose_code_bound = optim_dict['p']['code_bound']
            if pose_code_bound is not None:
                deepsdf_utils.project_latent_codes_onto_sphere(shape_codes, pose_code_bound)

            # print("batch done")

        # Store pose codes for optimization video
        if e % code_snapshot_every == 0:
            print()
            print("Storing code snapshot!")
            # shape_codes_optim_video[:, [e // code_snapshot_every], :] = shape_codes.detach().clone().cpu().numpy()
            # pose_codes_optim_video[:, [e // code_snapshot_every], :] = pose_codes.detach().clone().cpu().numpy()
            
            shape_codes_tmp = np.moveaxis(shape_codes.detach().clone().cpu().numpy(), 0, 1)
            pose_codes_tmp  = np.moveaxis(pose_codes.detach().clone().cpu().numpy(), 0, 1)

            shape_codes_optim_video.append(shape_codes_tmp)
            pose_codes_optim_video.append(pose_codes_tmp)

        # if e % 50 == 0:
        if e % print_loss_every == 0:
            lr_list = nnutils.get_learning_rates(optimizer)
            loss_recon_avg     = sum(loss_recon_batch) / len(loss_recon_batch) if len(loss_recon_batch) > 0 else -1
            loss_recon_p_avg   = sum(loss_recon_batch_partial) / len(loss_recon_batch_partial) if len(loss_recon_batch_partial) > 0 else -1
            loss_icp_avg       = sum(loss_icp_batch) / len(loss_icp_batch) if len(loss_icp_batch) > 0 else -1
            shape_code_reg_avg = sum(shape_code_reg_batch) / len(shape_code_reg_batch) if len(shape_code_reg_batch) > 0 else -1
            pose_code_reg_avg  = sum(pose_code_reg_batch) / len(pose_code_reg_batch) if len(pose_code_reg_batch) > 0 else -1
            temporal_reg_avg   = sum(temporal_reg_batch) / len(temporal_reg_batch) if len(temporal_reg_batch) > 0 else -1
            code_consist_avg   = sum(code_consistency_batch) / len(code_consistency_batch) if len(code_consistency_batch) > 0 else -1
            print(f"iter {e} (res {res}): recon {loss_recon_avg:.8f} | recon_p {loss_recon_p_avg:.8f} | icp {loss_icp_avg:.8f}| shape reg {shape_code_reg_avg:.8f} | pose reg {pose_code_reg_avg:.8f} | tmp reg {temporal_reg_avg:.8f} | codeconsist {code_consist_avg:.8f} | lr_shape: {lr_list[0]} | lr_pose: {lr_list[1]}")

        # print("iter done")

    lr_list = nnutils.get_learning_rates(optimizer)
    loss_recon_avg     = sum(loss_recon_batch) / len(loss_recon_batch) if len(loss_recon_batch) > 0 else -1
    loss_recon_p_avg   = sum(loss_recon_batch_partial) / len(loss_recon_batch_partial) if len(loss_recon_batch_partial) > 0 else -1
    loss_icp_avg       = sum(loss_icp_batch) / len(loss_icp_batch) if len(loss_icp_batch) > 0 else -1
    shape_code_reg_avg = sum(shape_code_reg_batch) / len(shape_code_reg_batch) if len(shape_code_reg_batch) > 0 else -1
    pose_code_reg_avg  = sum(pose_code_reg_batch) / len(pose_code_reg_batch) if len(pose_code_reg_batch) > 0 else -1
    temporal_reg_avg   = sum(temporal_reg_batch) / len(temporal_reg_batch) if len(temporal_reg_batch) > 0 else -1
    print(f"iter {e} (res {res}): recon {loss_recon_avg:.8f} | recon_p {loss_recon_p_avg:.8f} | icp {loss_icp_avg:.8f} | shape reg {shape_code_reg_avg:.8f} | pose reg {pose_code_reg_avg:.8f} | tmp reg {temporal_reg_avg:.8f} | codeconsist {code_consist_avg:.8f} | lr_shape: {lr_list[0]} | lr_pose: {lr_list[1]}")
    print()
    print("Optimization done!")
    print()

    shape_codes_optim_video = np.moveaxis(np.vstack(shape_codes_optim_video), 0, 1)
    pose_codes_optim_video = np.moveaxis(np.vstack(pose_codes_optim_video), 0, 1)

    return shape_codes, shape_codes_optim_video, pose_codes, pose_codes_optim_video


"""
Sample points around tpose
"""
def sample_points_around_tpose(shape_decoder, shape_codes, identity_id, total_sample_num, sigma, viz=False):
    
    # Generate mesh from shape code
    mesh_pred_trimesh = deepsdf_utils.create_mesh(
        shape_decoder, shape_codes, 
        identity_ids=[identity_id], shape_codes_dim=shape_codes.shape[-1]
    )

    # Sample points on the mesh
    points, _ = trimesh.sample.sample_surface_even(mesh_pred_trimesh, total_sample_num)
    
    # Now, add some noise to them
    points = points + sigma * np.random.randn(total_sample_num, 3)

    if viz:
        # Mesh
        mesh_pred = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(mesh_pred_trimesh.vertices),
            o3d.utility.Vector3iVector(mesh_pred_trimesh.faces)
        )
        mesh_pred.compute_vertex_normals()
        mesh_pred.paint_uniform_color([0, 0, 1])

        # Points
        points_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))

        o3d.visualization.draw_geometries([mesh_pred, points_pcd])

    return points