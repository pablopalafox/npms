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


class VoxelsDataset(Dataset):

    def __init__(
        self, 
        data_dir='data', 
        labels_json='labels.json',
        batch_size=64, 
        num_workers=12,
        res=64,
        cache_data=True,
        **kwargs
    ):

        self.res = res

        self.data_dir = data_dir
        self.labels_json = labels_json

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
                data_dict = self._load_sample(data)
                self.cache.append(data_dict)

            # T-Poses
            for index in range(len(self.labels_tpose)):
                data = self.labels_tpose[index]
                data_dict = self._load_sample(data)
                self.cache_tpose.append(data_dict)

    def _load(self):

        if isinstance(self.labels_json, dict):
            self.labels       = self.labels_json['labels']
            self.labels_tpose = self.labels_json['labels_tpose']
        else:
            assert os.path.isfile(self.labels_json), self.labels_json
            
            with open(self.labels_json) as f:
                self.labels = json.loads(f.read())
            
            self.labels_tpose_json = os.path.join(os.path.dirname(self.labels_json), "labels_tpose.json")
            
            with open(self.labels_tpose_json) as f:
                self.labels_tpose = json.loads(f.read())
        
        self.num_identities = len(self.labels_tpose)

    def __len__(self):
        return len(self.labels)

    def get_num_identities(self):
        return self.num_identities

    def _load_sample(self, data):
        shape_path = os.path.join(
            self.data_dir, data['dataset'], data['identity_name'], data['animation_name'], data['sample_id']
        )

        # Inputs
        voxel_path = shape_path + f'/partial_views/voxelized_view0_{self.res}res.npz'
        occupancies = np.unpackbits(np.load(voxel_path)['compressed_occupancies'])
        inputs = np.reshape(occupancies, (self.res,)*3)

        return {
            'inputs':      np.array(inputs, dtype=np.float32), 
            'path':        shape_path,
            'identity_id': data['identity_id']
        }

    def _get_identity_id(self, d):
        identity_id = d['identity_id']
        assert identity_id < self.num_identities, f"Identity {identity_id} is not defined in labels_tpose.json"
        return identity_id

    def __getitem__(self, idx):

        if self.cache_data:
            data_dict = self.cache[idx]

        else:
            data = self.labels[idx]
            data_dict = self._load_sample(data)
            
        return {
            'data': data_dict,
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
            worker_init_fn=self.worker_init_fn
        )

    def worker_init_fn(self, worker_id):
        random_data = os.urandom(4)
        base_seed = int.from_bytes(random_data, byteorder="big")
        np.random.seed(base_seed + worker_id)