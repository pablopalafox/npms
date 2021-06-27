import torch
import torch.utils.data as data_utils
import signal
import sys
import os
import logging
import math
import numpy as np
import json
import time
import open3d as o3d
import trimesh

from utils.sdf_utils import sample_grid_points, sample_scaled_grid_points
import marching_cubes as mcubes
# import mcubes


class SoftL1(torch.nn.Module):
    def __init__(self):
        super(SoftL1, self).__init__()

    def forward(self, input, target, eps=0.0):
        l1 = torch.abs(input - target)
        ret = l1 - eps
        ret = torch.clamp(ret, min=0.0, max=100.0)
        return ret, torch.mean(l1.detach())


class LearningRateSchedule:
    def get_learning_rate(self, epoch):
        pass


class StepLearningRateSchedule(LearningRateSchedule):
    def __init__(self, specs):
        print(specs)
        self.initial = specs['initial']
        self.interval = specs['interval']
        self.factor = specs['factor']

    def get_learning_rate(self, epoch):
        return self.initial * (self.factor ** (epoch // self.interval))


def adjust_learning_rate(lr_schedules, optimizer, epoch):
    for i, param_group in enumerate(optimizer.param_groups):
        param_group["lr"] = lr_schedules[i].get_learning_rate(epoch)


def latent_size_regul(latent_codes, indices, component_mean=None, component_std=None):
    # OneCodePerFrame
    latent_codes_squared = latent_codes[indices, ...].pow(2) # [batch_size, 1, code_dim]
    if component_std is not None:
        latent_codes_squared = latent_codes_squared / component_std.pow(2)
    latent_loss = torch.mean(latent_codes_squared, dim=-1)   # [batch_size, 1]
    
    latent_loss = torch.mean(latent_loss)                       
    return latent_loss

def latent_size_regul_bis(latent_codes):
    # OneCodePerFrame
    latent_codes_squared = latent_codes.pow(2) # [batch_size, 1, code_dim]
    latent_loss = torch.mean(latent_codes_squared, dim=-1)   # [batch_size, 1]
    
    latent_loss = torch.mean(latent_loss)                       
    return latent_loss

def empirical_stat(latent_vecs, indices):
    lat_mat = torch.zeros(0).cuda()
    for ind in indices:
        lat_mat = torch.cat([lat_mat, latent_vecs[ind]], 0)
    mean = torch.mean(lat_mat, 0)
    var = torch.var(lat_mat, 0)
    return mean, var


def get_mean_latent_vector_magnitude_old(latent_codes):
    host_vectors = np.array(
        [vec.detach().cpu().numpy().squeeze() for vec in latent_codes]
    )
    return np.mean(np.linalg.norm(host_vectors, axis=1))


def get_mean_latent_code_magnitude(latent_codes):
    host_latent_codes = latent_codes.detach().cpu().numpy()
    assert len(host_latent_codes.shape) == 3
    return np.mean(np.linalg.norm(host_latent_codes, axis=2))


def threshold_min_max(tensor, min_vec, max_vec):
    return torch.min(max_vec, torch.max(tensor, min_vec))


def project_latent_codes_onto_sphere(latent_codes, radius):
    length = torch.norm(latent_codes, dim=-1, keepdim=True).detach()
    latent_codes.data = latent_codes.mul(radius / length)


####################################################################################
####################################################################################

"""
def create_mesh_ours_supersampling(
    decoder, latent_code, identity_ids, shape_codes_dim, dim=256, n_edge_samples=0
):
    print("Super sampling")
    latent_code.requires_grad = False

    # Get shape codes for batch samples
    shape_codes_batch = latent_code[identity_ids, ...] # [bs, 1, C]
    assert shape_codes_batch.shape[1] == 1, shape_codes_batch.shape
    
    # start = timer()

    for param in decoder.parameters():
        param.requires_grad = False

    assert not decoder.training

    num_chunks = 32
    
    # if dim == 128:
    #     num_chunks = 16
    # elif dim == 256:
    #     num_chunks = 128
    #     n_edge_samples = 5
    # elif dim == 512:
    #     num_chunks = 128

    # Extract the 0-isosurface with super res.
    edges = dim - 1
    supersamples = dim + edges * n_edge_samples

    dims_x = [supersamples, dim, dim]
    dims_y = [dim, supersamples, dim]
    dims_z = [dim, dim, supersamples]

    dims = [dims_x, dims_y, dims_z]

    points_all = sample_scaled_grid_points(dims)

    # We'll store the sdf here
    sdf = []

    for g, pts in enumerate(points_all):

        points = torch.from_numpy(pts).permute(1, 0)
        assert points.shape[1] == 3

        num_points = points.shape[0]
        assert num_points % num_chunks == 0, "The number of points in the grid must be divisible by the number of chunks"
        points_per_chunk = int(num_points / num_chunks)

        # print(points.shape)

        sdf_pred = np.empty((num_points), dtype=np.float32)
        
        for i in range(num_chunks):
            points_i = points[i*points_per_chunk:(i+1)*points_per_chunk, :].cuda()
            points_i.requires_grad = False

            num_points_i = points_i.shape[0]

            # Run forward pass.
            # Extent latent code to all sampled points
            shape_codes_repeat = shape_codes_batch.expand(-1, num_points_i, -1) # [bs, N, C]
            shape_codes_inputs = shape_codes_repeat.reshape(-1, shape_codes_dim) # [bs*N, C]

            shape_inputs = torch.cat([shape_codes_inputs, points_i], 1)

            sdf_pred_i = decoder(shape_inputs).squeeze(1).detach().cpu().numpy()

            # Concatenate
            sdf_pred[..., i*points_per_chunk:(i+1)*points_per_chunk] = sdf_pred_i

        sdf_pred = sdf_pred.reshape(dims[g][0], dims[g][1], dims[g][2])

        sdf.append(sdf_pred)

    vertices, triangles = mcubes.marching_cubes_super_sampling(sdf[0], sdf[1], sdf[2], 0)

    vertices = 1.0 * (vertices / (dim - 1)) - 0.5

    return trimesh.Trimesh(vertices, triangles)


def create_mesh_ours(decoder, latent_code, identity_ids, shape_codes_dim, dim=512, num_chunks=128):

    latent_code.requires_grad = False

    # Get shape codes for batch samples
    shape_codes_batch = latent_code[identity_ids, ...] # [bs, 1, C]
    assert shape_codes_batch.shape[1] == 1, shape_codes_batch.shape
    
    # start = timer()

    for param in decoder.parameters():
        param.requires_grad = False

    assert not decoder.training

    # Sample grid points
    points = np.moveaxis(sample_grid_points(dim), 0, 1)
    # visualize_grid([points], colors=None, exit_after=True)
    points = torch.from_numpy(points)#.cuda() # Move to device

    num_points = points.shape[0]
    assert num_points % num_chunks == 0, "The number of points in the grid must be divisible by the number of chunks"
    points_per_chunk = int(num_points / num_chunks)

    sdf_pred = np.empty((num_points), dtype=np.float32)
    for i in range(num_chunks):
        points_i = points.clone()[i*points_per_chunk:(i+1)*points_per_chunk, :].cuda()

        num_points_i = points_i.shape[0]

        # Run forward pass.
        # Extent latent code to all sampled points
        shape_codes_repeat = shape_codes_batch.expand(-1, num_points_i, -1) # [bs, N, C]
        shape_codes_inputs = shape_codes_repeat.reshape(-1, shape_codes_dim) # [bs*N, C]

        shape_inputs = torch.cat([shape_codes_inputs, points_i], 1)

        # print(shape_inputs.shape)

        sdf_pred_i = decoder(shape_inputs).squeeze(1).detach().cpu().numpy()
        # sdf_pred_i = decode_sdf(decoder, latent_code, points_i).squeeze(1).detach().cpu().numpy()

        # Concatenate
        sdf_pred[..., i*points_per_chunk:(i+1)*points_per_chunk] = sdf_pred_i

    sdf_pred = sdf_pred.reshape(dim, dim, dim)

    # Extract mesh with Marching cubes.
    vertices, triangles = mcubes.marching_cubes(sdf_pred, 0)

    # Normalize vertices to be in [-1, 1]
    # vertices = 2.0 * (vertices / (dim - 1)) - 1.0
    vertices = 1.0 * (vertices / (dim - 1)) - 0.5

    return trimesh.Trimesh(vertices, triangles)
"""

def create_mesh_from_code(decoder, latent_code, shape_codes_dim, N=256, max_batch=32 ** 3):
    latent_code.requires_grad = False

    # Get shape codes for batch samples
    shape_codes_batch = latent_code
    assert shape_codes_batch.shape[1] == 1, shape_codes_batch.shape

    decoder.eval()

    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    # voxel_origin = [-1, -1, -1]
    bbox_min = -0.5
    bbox_max = 0.5

    voxel_origin = [bbox_min] * 3
    voxel_size = (bbox_max - bbox_min) / (N - 1)
    
    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 4)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.long() // N) % N
    samples[:, 0] = ((overall_index.long() // N) // N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3

    samples.requires_grad = False

    head = 0

    while head < num_samples:
        sample_subset = samples[head : min(head + max_batch, num_samples), 0:3].cuda()

        # Run forward pass.
        # Extent latent code to all sampled points
        shape_codes_repeat = shape_codes_batch.expand(-1, sample_subset.shape[0], -1) # [bs, N, C]
        shape_codes_inputs = shape_codes_repeat.reshape(-1, shape_codes_dim) # [bs*N, C]

        shape_inputs = torch.cat([shape_codes_inputs, sample_subset], 1)

        sdf_pred_i = decoder(shape_inputs).squeeze(1).detach().cpu()

        samples[head : min(head + max_batch, num_samples), 3] = (
            sdf_pred_i
        )
        head += max_batch

    sdf_values = samples[:, 3]
    sdf_values = sdf_values.reshape(N, N, N)

    # Extract mesh with Marching cubes.
    vertices, triangles = mcubes.marching_cubes(sdf_values.numpy(), 0)

    # Normalize vertices to be in [-1, 1]
    step = (bbox_max - bbox_min) / (N - 1)
    vertices = np.multiply(vertices, step)
    vertices += [bbox_min, bbox_min, bbox_min]

    return trimesh.Trimesh(vertices, triangles)


def create_mesh(decoder, latent_code, identity_ids, shape_codes_dim, N=256, max_batch=32 ** 3):

    latent_code.requires_grad = False

    # Get shape codes for batch samples
    assert len(identity_ids) == 1 and identity_ids[0] < latent_code.shape[0], f"Identity id {identity_ids[0]} is out of range of latent code of shape {latent_code.shape}" 
    shape_codes_batch = latent_code[identity_ids, ...] # [bs, 1, C]
    assert shape_codes_batch.shape[1] == 1, shape_codes_batch.shape

    decoder.eval()

    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    # voxel_origin = [-1, -1, -1]
    bbox_min = -0.5
    bbox_max = 0.5

    voxel_origin = [bbox_min] * 3
    voxel_size = (bbox_max - bbox_min) / (N - 1)
    
    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 4)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.long() // N) % N
    samples[:, 0] = ((overall_index.long() // N) // N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3

    samples.requires_grad = False

    head = 0

    while head < num_samples:
        sample_subset = samples[head : min(head + max_batch, num_samples), 0:3].cuda()

        # Run forward pass.
        # Extent latent code to all sampled points
        shape_codes_repeat = shape_codes_batch.expand(-1, sample_subset.shape[0], -1) # [bs, N, C]
        shape_codes_inputs = shape_codes_repeat.reshape(-1, shape_codes_dim) # [bs*N, C]

        shape_inputs = torch.cat([shape_codes_inputs, sample_subset], 1)

        sdf_pred_i = decoder(shape_inputs).squeeze(1).detach().cpu()

        samples[head : min(head + max_batch, num_samples), 3] = (
            sdf_pred_i
        )
        head += max_batch

    sdf_values = samples[:, 3]
    sdf_values = sdf_values.reshape(N, N, N)

    # Extract mesh with Marching cubes.
    vertices, triangles = mcubes.marching_cubes(sdf_values.numpy(), 0)

    # Normalize vertices to be in [-1, 1]
    step = (bbox_max - bbox_min) / (N - 1)
    vertices = np.multiply(vertices, step)
    vertices += [bbox_min, bbox_min, bbox_min]

    return trimesh.Trimesh(vertices, triangles)


def compute_trimesh_chamfer(
    gt_points, gen_points
):
    from scipy.spatial import cKDTree as KDTree
    import trimesh
    
    """
	This function computes a symmetric chamfer distance, i.e. the sum of both chamfers.
	"""

    gen_points_kd_tree = KDTree(gen_points)
    one_distances, one_vertex_ids = gen_points_kd_tree.query(gt_points)
    gt_to_gen_temp = np.square(one_distances)
    gt_to_gen_chamfer = np.mean(gt_to_gen_temp)
    
    gt_points_kd_tree = KDTree(gt_points)
    two_distances, two_vertex_ids = gt_points_kd_tree.query(gen_points)
    gen_to_gt_temp = np.square(two_distances)
    gen_to_gt_chamfer = np.mean(gen_to_gt_temp)

    squared_chamfer = gt_to_gen_chamfer + gen_to_gt_chamfer

    # For easier understanding, compute the "unsquared" version
    unsquared_chamfer = np.mean(one_distances) + np.mean(two_distances)
    
    return {
        'squared_chamfer': squared_chamfer, 
        'unsquared_chamfer': unsquared_chamfer
    }
