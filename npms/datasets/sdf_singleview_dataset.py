import sys
from torch.utils.data import Dataset
import os
import numpy as np
import trimesh
import torch
import json
from tqdm import tqdm
import open3d as o3d
from timeit import default_timer as timer

from utils import gaps_utils

from utils.viz_utils import occupancy_to_o3d
import data_processing.implicit_waterproofing as iw


class SingleViewSDFDataset(Dataset):
    def __init__(
        self, 
        data_dir='data', 
        labels_json='labels.json',
        batch_size=4,
        num_workers=4, 
        res=64,
        radius=1,
        ref_sample_info={}, 
        cache_data=True,
        use_partial_input=True,
        use_sdf_from_ifnet=True,
        **kwargs
    ):
        self.use_partial_input = use_partial_input
        self.use_sdf_from_ifnet = use_sdf_from_ifnet

        self.res = res
        self.radius = radius

        self.points = ref_sample_info['points']
        self.total_num_samples = self.points.shape[0]
        self.num_samples = ref_sample_info['num_samples']
        self.num_samples_cur = ref_sample_info['num_samples_cur']

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
                data_dict = self._load_sample(data, False)
                self.cache.append(data_dict)

            # T-Poses
            for index in range(len(self.labels_tpose)):
                data = self.labels_tpose[index]
                data_dict = self._load_sample(data, True)
                self.cache_tpose.append(data_dict)

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

    def get_res(self):
        return self.res

    def get_batch_size(self):
        return self.batch_size

    def _load_sample(self, data, is_tpose):
        shape_path = os.path.join(self.data_dir, data['dataset'], data['identity_name'], data['animation_name'], data['sample_id'])

        points_world = None
        inputs_sdf = None
        inputs_psdf = None
        grid_mask = None

        if not is_tpose:
            # Partial sdf grid
            partial_sdf_grid_path = shape_path + f'/partial_sdf_grd/partial_sdf_grd_{self.res}.npz'
            partial_sdf_grid_npz = np.load(partial_sdf_grid_path)
            points_world = partial_sdf_grid_npz['points_world']
            inputs_psdf  = partial_sdf_grid_npz['sdf_grid']
            grid_mask    = partial_sdf_grid_npz['validity_mask']

            # Set invalid voxels to zero (we'll mask them out later on when computing the loss)
            # inputs_psdf[~grid_mask] = np.nan

            inputs_psdf = np.reshape(inputs_psdf, (self.res,)*3)
            grid_mask   = np.reshape(grid_mask,   (self.res,)*3)

            # Complete sdf grid
            inputs_sdf = []
            if not self.use_partial_input:

                if self.use_sdf_from_ifnet:
                    # IFNet-based sdf
                    sdf_grid_path = shape_path + f'/ifnet_sdf_grd_{self.res}.npz'
                    sdf_grd_npz = np.load(sdf_grid_path)
                    # print("LOADING IFNET SDF")
                    sdf_grd_tmp_flat = sdf_grd_npz['sdf_grd_flat']
                    inputs_sdf = np.reshape(sdf_grd_tmp_flat, (self.res,)*3)

                else:
                    # GT sdf
                    sdf_grid_path = shape_path + f'/sdf_{self.res}.grd'
                    _, grd = gaps_utils.read_grd(sdf_grid_path)
                    inputs_sdf = np.transpose(grd, (2, 1, 0)) # current format is z y x, so convert to x y z

            #######################################################
            if False:
                grid_points = iw.create_grid_points_from_bounds(-0.5, 0.5, self.res)

                invalid_mask_grid_tmp = ~grid_mask
                invalid_mask_grid_tmp = invalid_mask_grid_tmp.astype(np.int8)
                invalid_voxels_mesh = occupancy_to_o3d(invalid_mask_grid_tmp)

                # Partial grid
                in_flat_mask = np.reshape(inputs_psdf, (-1,)) < 0
                pgrid_pcd_in = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(grid_points[in_flat_mask]))
                pgrid_pcd_in.paint_uniform_color([0, 0, 1])

                out_flat_mask = np.reshape(inputs_psdf, (-1,)) > 0
                out_flat_mask = out_flat_mask & (np.reshape(inputs_psdf, (-1,)) < 0.01)
                pgrid_pcd_out = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(grid_points[out_flat_mask]))
                pgrid_pcd_out.paint_uniform_color([1, 0, 0])

                # Grid points
                invalid_flat_mask = np.reshape(invalid_mask_grid_tmp, (-1,)).astype(np.bool)
                grid_pcd_invalid = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(grid_points[invalid_flat_mask]))
                grid_pcd_invalid.paint_uniform_color([1, 0, 1])

                # invalid and in
                invalid_in = invalid_flat_mask & in_flat_mask
                grid_pcd_invalid_in = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(grid_points[invalid_in]))
                grid_pcd_invalid_in.paint_uniform_color([1, 1, 0])

                # invalid and in
                valid_in = ~invalid_flat_mask & in_flat_mask
                grid_pcd_valid_in = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(grid_points[valid_in]))
                grid_pcd_valid_in.paint_uniform_color([0, 1, 1])

                o3d.visualization.draw_geometries([invalid_voxels_mesh, pgrid_pcd_in, pgrid_pcd_out])
                o3d.visualization.draw_geometries([pgrid_pcd_in, pgrid_pcd_out])
                # o3d.visualization.draw_geometries([pgrid_pcd_in, pgrid_pcd_out])
                o3d.visualization.draw_geometries([pgrid_pcd_in])
                o3d.visualization.draw_geometries([grid_pcd_invalid])
                o3d.visualization.draw_geometries([grid_pcd_valid_in, grid_pcd_invalid_in])
                exit()
            #######################################################

        return {
            'points':      np.array(self.points, dtype=np.float32), 
            'points_cur':  np.array(points_world, dtype=np.float32), 
            'inputs_sdf':  np.array(inputs_sdf, dtype=np.float32), 
            'inputs_psdf': np.array(inputs_psdf, dtype=np.float32), 
            'grid_mask':   np.array(grid_mask, dtype=np.float32), 
            'path':        shape_path,
            'identity_id': data['identity_id']
        }

    def _subsample(self, data_dict, is_tpose):
        points = []
        points_cur = []

        if is_tpose:
            boundary_sample_points = data_dict['points']
            subsample_indices = np.random.randint(0, self.total_num_samples, self.num_samples)
            points.extend(boundary_sample_points[subsample_indices])
            assert len(points) == self.num_samples, f"{len(points)} vs {self.num_samples}"

        else:
            boundary_sample_points_cur = data_dict['points_cur']
            subsample_indices = np.random.randint(0, boundary_sample_points_cur.shape[0], self.num_samples_cur)
            points_cur.extend(boundary_sample_points_cur[subsample_indices])
            assert len(points_cur) == self.num_samples_cur, f"{len(points_cur)} vs {self.num_samples_cur}"

        return {
            'points':      np.array(points, dtype=np.float32),
            'points_cur':  np.array(points_cur, dtype=np.float32),
            'inputs_sdf':  data_dict['inputs_sdf'], 
            'inputs_psdf': data_dict['inputs_psdf'], 
            'grid_mask':   data_dict['grid_mask'],
            'path':        data_dict['path'],
            'identity_id': data_dict['identity_id']
        }

    def _get_identity_id(self, d):
        identity_id = d['identity_id']
        assert identity_id < self.num_identities, f"Identity {identity_id} is not defined in labels_tpose.json"
        return identity_id

    def __getitem__(self, idx):

        # Neighboring indices around the current idx
        segment_idxs = [idx]
        for delta in range(1, self.radius + 1):
            if idx + delta == len(self):
                delta = -delta
            segment_idxs.append(idx + delta)

        # List of data dictionaries
        data_dicts = []

        if self.cache_data:
            for sidx in segment_idxs:
                data_dict = self.cache[sidx]
                data_dicts.append(data_dict)
            data_ref_dict = self.cache_tpose[self._get_identity_id(data_dict)]    

        else:
            for sidx in segment_idxs:
                data = self.labels[sidx]
                data_dict = self._load_sample(data, False)
                data_dicts.append(data_dict)

            data_ref = self.labels_tpose[self._get_identity_id(data)]
            data_ref_dict = self._load_sample(data_ref, True)
            
        # Subsample
        data_dicts_subsampled = []

        for data_dict in data_dicts:
            data_dict = self._subsample(data_dict, False)
            data_dicts_subsampled.append(data_dict)
        
        data_ref_dict = self._subsample(data_ref_dict, True)

        return {
            'ref': data_ref_dict,
            'curs': data_dicts_subsampled,
            'idxs': segment_idxs,
            'identity_id': data_dict['identity_id'],
        }

    def get_loader(self, shuffle=True, res=None):

        if res is not None:
            self.res = res

        assert self.batch_size <= len(self), f"batch size ({self.batch_size}) > len dataset ({len(self)})" 

        return torch.utils.data.DataLoader(
            self, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            shuffle=shuffle,
            worker_init_fn=self.worker_init_fn,
            drop_last=True
        )

    def worker_init_fn(self, worker_id):
        random_data = os.urandom(4)
        base_seed = int.from_bytes(random_data, byteorder="big")
        np.random.seed(base_seed + worker_id)