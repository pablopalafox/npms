from __future__ import division
import sys
from torch.utils.data import Dataset
import os
import numpy as np
import pickle
import imp
import trimesh
import torch
import json
from tqdm import tqdm
from timeit import default_timer as timer

from utils.gaps_utils import read_pts_file


class SDFDataset(Dataset):

    def __init__(
        self, 
        data_dir='data', 
        labels_json='labels.json',
        batch_size=64, 
        num_workers=12, 
        sample_info={}, 
        cache_data=True,
        **kwargs
    ):

        ###################################################################################################
        # SDF
        ###################################################################################################
        sdf_samples_info = sample_info['sdf']
        self.sdf_samples_types = sdf_samples_info['types']
        self.num_points_sdf    = sdf_samples_info['num_points']
        percentages = [p for p in self.sdf_samples_types.values()]

        assert sum(percentages) > 0.999 and sum(percentages) <= 1.0, sum(percentages)
        
        if self.num_points_sdf > 0:
            self.sdf_samples_types = {k: int(v * self.num_points_sdf) for k, v in self.sdf_samples_types.items()}
            assert sum(self.sdf_samples_types.values()) == self.num_points_sdf

            print()
            print("num sdf samples", self.num_points_sdf)
            print()
        
        ###################################################################################################
        # Flow
        ###################################################################################################
        sample_flow_info = sample_info['flow']
        self.num_points_flow    = np.array(sample_flow_info['num_points'])

        self.num_flow_samples_list = []
        if self.num_points_flow > 0:
            self.sample_flow_dist   = np.array(sample_flow_info['dist']) / len(sample_flow_info['dist'])
            self.sample_flow_sigmas = np.array(sample_flow_info['sigmas'])
            
            assert np.sum(self.sample_flow_dist) == 1
            assert np.any(self.sample_flow_dist < 0) == False
            assert len(self.sample_flow_dist) == len(self.sample_flow_sigmas)

            self.num_flow_samples_list = np.rint(self.sample_flow_dist * self.num_points_flow).astype(np.uint32)
            assert np.all(self.num_flow_samples_list == self.num_flow_samples_list[0]), f"num_samples: {self.num_flow_samples_list}"
            self.num_flow_samples_per_sigma = self.num_flow_samples_list[0]

            print()
            print("num_samples", self.num_flow_samples_list)
            print("num_samples per sigma", self.num_flow_samples_per_sigma)
            print()

        self.max_samples_per_sigma = 100000

        self.num_samples_per_shape = {
            'sdf': self.num_points_sdf,
            'flow': self.num_points_flow,
        }
            
        self.data_dir = data_dir
        self.labels_json = labels_json
        self.labels_tpose_json = os.path.join(os.path.dirname(labels_json), "labels_tpose.json")

        self.cache_data = cache_data
        self.cache = []
        self.cache_tpose = []

        # Load labels
        self._load()

        self.batch_size = batch_size
        self.num_workers = num_workers

        # Preload data
        if self.cache_data:
            print("Preloading cached data ...")

            for index in tqdm(range(len(self.labels))):
                data = self.labels[index]
                data_dict = self._load_sample(data, is_tpose=False)
                self.cache.append(data_dict)

            # T-Poses
            for index in range(len(self.labels_tpose)):
                data = self.labels_tpose[index]
                data_dict = self._load_sample(data, is_tpose=True)
                self.cache_tpose.append(data_dict)

            print("Loaded cached data ...")

    def _load(self):
        with open(self.labels_json) as f:
            self.labels = json.loads(f.read())

        with open(self.labels_tpose_json) as f:
            self.labels_tpose = json.loads(f.read())
        
        self.num_identities = len(self.labels_tpose)

    def __len__(self):
        return len(self.labels)

    def get_num_identities(self):
        return self.num_identities

    def get_num_samples_per_shape(self):
        return self.num_samples_per_shape

    def _load_sample(self, data, is_tpose):
        if 'dataset' in data:
            shape_path = os.path.join(self.data_dir, data['dataset'], data['identity_name'], data['animation_name'], data['sample_id'])
        else:
            shape_path = os.path.join(self.data_dir, data['identity_name'], data['animation_name'], data['sample_id'])

        # BOUNDARY
        points_sdf_dict = {}

        if is_tpose and self.num_points_sdf > 0:
            for i, sdf_samples_type in enumerate(self.sdf_samples_types):
                ext = 'pts' if sdf_samples_type == "surface" else 'sdf' 
                sdf_samples_path = shape_path + f'/samples_{sdf_samples_type}.{ext}'

                # Load data from disk
                sdf_points = read_pts_file(sdf_samples_path)

                if ext == 'pts':
                    sdf_points = sdf_points[:, :3]
                
                points_sdf_dict[sdf_samples_type] = sdf_points

        # FLOW
        points_flow = None

        for i in range(len(self.num_flow_samples_list)):
            # points
            flow_samples_path = shape_path + '/flow_samples_{}.npz'.format(self.sample_flow_sigmas[i])
            flow_samples_npz = np.load(flow_samples_path)
            flow_sample_points = flow_samples_npz['points'][None, ...]

            if points_flow is None:
                points_flow = flow_sample_points
            else:
                points_flow = np.concatenate((points_flow, flow_sample_points), axis=0) # factor, 100k, 3

        return {
            'points_sdf_dict': points_sdf_dict,
            'points_flow':     np.array(points_flow, dtype=np.float32), 
            'path':            shape_path,
            'identity_id':     data['identity_id']
        }

    def _subsample(self, data_dict, subsample_indices_list, is_tpose):
        points_sdf = []
        points_flow = []

        # SDF samples
        if is_tpose and self.num_points_sdf > 0:
            points_sdf_dict = data_dict['points_sdf_dict']

            points_sdf = prepare_samples(points_sdf_dict, self.sdf_samples_types)
            
            assert points_sdf.shape[0] == self.num_points_sdf, f"{points_sdf.shape[0]} vs {self.num_points_sdf}"

        # Flow samples
        for i in range(len(self.num_flow_samples_list)): # sample type

            flow_sample_points  = data_dict['points_flow'][i]

            subsample_indices = subsample_indices_list[i]

            points_flow.extend(flow_sample_points[subsample_indices])

        assert len(points_flow) == self.num_points_flow, f"{len(points_flow)} vs {self.num_points_flow}"

        return {
            'points_sdf':  np.array(points_sdf, dtype=np.float32),
            'points_flow': np.array(points_flow, dtype=np.float32),
            'path':        data_dict['path'],
            'identity_id': data_dict['identity_id']
        }

    def _get_identity_id(self, d):
        identity_id = d['identity_id']
        assert identity_id < self.num_identities, f"Identity {identity_id} is not defined in labels_tpose.json"
        return identity_id

    def __getitem__(self, idx):

        if self.cache_data:
            data_dict     = self.cache[idx]
            data_ref_dict = self.cache_tpose[self._get_identity_id(data_dict)]    

        else:
            data     = self.labels[idx]
            data_ref = self.labels_tpose[self._get_identity_id(data)]

            # Load samples
            data_dict     = self._load_sample(data, is_tpose=False)
            data_ref_dict = self._load_sample(data_ref, is_tpose=True)
            
        # Sample random indices for each sample type
        subsample_flow_indices_list = []
        for i, num in enumerate(self.num_flow_samples_list): # sample type
            subsample_indices = np.random.randint(0, self.max_samples_per_sigma, num)
            subsample_flow_indices_list.append(subsample_indices)

        # Subsample
        data_ref_dict = self._subsample(data_ref_dict, subsample_flow_indices_list, is_tpose=True)
        data_dict     = self._subsample(data_dict,     subsample_flow_indices_list, is_tpose=False)

        # print(data_ref_dict['path'])
        # print(data_dict['path'])
        # print()

        return {
            'ref': data_ref_dict,
            'cur': data_dict,
            'idx': idx,
            'identity_id': data_dict['identity_id'],
        }

    def get_loader(self, shuffle=True):

        assert self.batch_size <= len(self), f"batch size ({self.batch_size}) > len dataset ({len(self)})" 

        return torch.utils.data.DataLoader(
            self, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            shuffle=shuffle,
            worker_init_fn=self.worker_init_fn,
            pin_memory=True,
            drop_last=True
        )

    def get_batch_size(self):
        return self.batch_size

    def worker_init_fn(self, worker_id):
        random_data = os.urandom(4)
        base_seed = int.from_bytes(random_data, byteorder="big")
        np.random.seed(base_seed + worker_id)


def prepare_samples(sample_data, N_target_dict):

    ##################################################################################################
    # Surface
    ##################################################################################################
    surface_sdf_samples = np.empty((0, 4))

    # Subsample
    if N_target_dict['surface'] > 0:
        surface_sdf_samples = sample_data['surface']
        N = N_target_dict['surface']
        assert surface_sdf_samples.shape[0] > N
        surface_idxs = np.random.permutation(surface_sdf_samples.shape[0])[:N]
        surface_sdf_samples = surface_sdf_samples[surface_idxs, :]
        assert surface_sdf_samples.shape[1] == 3
    
        # Generate gt sdf for surface points (all have 0 sdf values)
        surface_sdf = np.zeros((surface_sdf_samples.shape[0], 1), dtype=np.float32)
        surface_sdf_samples = np.concatenate((surface_sdf_samples, surface_sdf), axis=1)

    # print(surface_sdf_samples.shape)

    ##################################################################################################
    # Near Surface Sampled
    ##################################################################################################
    nss_sdf_samples = np.empty((0, 4))
    
    # Subsample
    if N_target_dict['near'] > 0:
        nss_sdf_samples = sample_data['near']
        N = N_target_dict['near']
        assert nss_sdf_samples.shape[0] > N
        nss_idxs = np.random.permutation(nss_sdf_samples.shape[0])[:N]
        nss_sdf_samples = nss_sdf_samples[nss_idxs, :]
        assert nss_sdf_samples.shape[1] == 4

    # print(nss_sdf_samples.shape)
    # import open3d as o3d
    # pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(nss_sdf_samples[:, :3]))
    # pcd.paint_uniform_color([1, 0, 0])
    # o3d.visualization.draw_geometries([pcd])

    ##################################################################################################
    # Uniform
    ##################################################################################################
    uniform_sdf_samples = np.empty((0, 4))
    
    # Subsample
    if N_target_dict['uniform'] > 0:
        uniform_sdf_samples = sample_data['uniform']
        N = N_target_dict['uniform']
        assert uniform_sdf_samples.shape[0] > N
        uniform_idxs = np.random.permutation(uniform_sdf_samples.shape[0])[:N]
        uniform_sdf_samples = uniform_sdf_samples[uniform_idxs, :]
        assert uniform_sdf_samples.shape[1] == 4

    # print(uniform_sdf_samples.shape)
    # pcd2 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(uniform_sdf_samples[:, :3]))
    # pcd2.paint_uniform_color([0, 1, 0])
    # o3d.visualization.draw_geometries([pcd, pcd2])

    ##################################################################################################
    ##################################################################################################
    # Concatenate
    ##################################################################################################
    ##################################################################################################
    all_sdf_samples = np.concatenate((nss_sdf_samples, surface_sdf_samples, uniform_sdf_samples), axis=0)

    return all_sdf_samples


# class VoxelsDataset(Dataset):

#     def __init__(
#         self, 
#         data_dir='data', 
#         labels_json='labels.json',
#         batch_size=64, 
#         num_workers=12,
#         res=64,
#         cache_data=True,
#         **kwargs
#     ):

#         self.res = res

#         self.data_dir = data_dir
#         self.labels_json = labels_json
#         self.labels_tpose_json = os.path.join(os.path.dirname(labels_json), "labels_tpose.json")

#         self.cache_data = cache_data
#         self.cache = []
#         self.cache_tpose = []

#         # Load labels
#         self._load()

#         self.batch_size = batch_size
#         self.num_workers = num_workers

#         # Preload data
#         if self.cache_data:
#             print("Preloading cached data ...")

#             for index in tqdm(range(len(self.labels))):
#                 data = self.labels[index]
#                 data_dict = self._load_sample(data)
#                 self.cache.append(data_dict)

#             # T-Poses
#             for index in range(len(self.labels_tpose)):
#                 data = self.labels_tpose[index]
#                 data_dict = self._load_sample(data)
#                 self.cache_tpose.append(data_dict)

#     def _load(self):
#         with open(self.labels_json) as f:
#             self.labels = json.loads(f.read())

#         with open(self.labels_tpose_json) as f:
#             self.labels_tpose = json.loads(f.read())
        
#         self.num_identities = len(self.labels_tpose)

#     def __len__(self):
#         return len(self.labels)

#     def get_num_identities(self):
#         return self.num_identities

#     def _load_sample(self, data):
#         shape_path = os.path.join(self.data_dir, data['identity_name'], data['animation_name'], data['sample_id'])

#         # Inputs
#         voxel_path = shape_path + f'/partial_views/voxelized_view0_{self.res}res.npz'
#         occupancies = np.unpackbits(np.load(voxel_path)['compressed_occupancies'])
#         inputs = np.reshape(occupancies, (self.res,)*3)

#         return {
#             'inputs':      np.array(inputs, dtype=np.float32), 
#             'path':        shape_path,
#             'identity_id': data['identity_id']
#         }

#     def _get_identity_id(self, d):
#         identity_id = d['identity_id']
#         assert identity_id < self.num_identities, f"Identity {identity_id} is not defined in labels_tpose.json"
#         return identity_id

#     def __getitem__(self, idx):

#         if self.cache_data:
#             data_dict = self.cache[idx]

#         else:
#             data = self.labels[idx]
#             # Load samples
#             data_dict = self._load_sample(data)
            
#         # print(data_ref_dict['path'])
#         # print(data_dict['path'])
#         # print()

#         return {
#             'data': data_dict,
#             'idx': idx,
#             'identity_id': data_dict['identity_id'],
#         }

#     def get_loader(self, shuffle=True):

#         assert self.batch_size <= len(self), f"batch size ({self.batch_size}) > len dataset ({len(self)})" 

#         return torch.utils.data.DataLoader(
#             self, 
#             batch_size=self.batch_size, 
#             num_workers=self.num_workers, 
#             shuffle=shuffle,
#             worker_init_fn=self.worker_init_fn
#         )

#     def worker_init_fn(self, worker_id):
#         random_data = os.urandom(4)
#         base_seed = int.from_bytes(random_data, byteorder="big")
#         np.random.seed(base_seed + worker_id)
