import os
import json
from tqdm import tqdm 

import config as cfg


if __name__ == "__main__":

    ROOT = f'/cluster/lothlann/ppalafox/datasets'

    dataset_name = "MIX-POSE__AMASS-50id-10349__MIXAMO-165id-40000__CAPE-35id-20533"

    from utils.parsing_utils import get_dataset_type_from_dataset_name
    dataset_type = get_dataset_type_from_dataset_name(dataset_name)
    splits_dir = f"{cfg.splits_dir}_{dataset_type}"

    labels_json       = os.path.join(ROOT, splits_dir, dataset_name, "labels.json")
    labels_tpose_json = os.path.join(ROOT, splits_dir, dataset_name, "labels_tpose.json")
    
    with open(labels_json, "r") as f:
        labels = json.loads(f.read())

    with open(labels_tpose_json, "r") as f:
        labels_tpose = json.loads(f.read())

    input(f"labels_tpose: {len(labels_tpose)}")
    input(f"labels:       {len(labels)}")

    # Verify tpose
    for label_tpose in tqdm(labels_tpose):
        sample_dir = os.path.join(ROOT, label_tpose['dataset'], label_tpose['identity_name'], label_tpose['animation_name'], label_tpose['sample_id'])
        assert os.path.isdir(sample_dir), sample_dir

        samples_near    = os.path.join(sample_dir, 'samples_near.sdf')
        samples_surface = os.path.join(sample_dir, 'samples_surface.pts')
        samples_uniform = os.path.join(sample_dir, 'samples_uniform.sdf')
        assert os.path.isfile(samples_near), samples_near
        assert os.path.isfile(samples_surface), samples_surface
        assert os.path.isfile(samples_uniform), samples_uniform

        flow_path_002 = os.path.join(sample_dir, 'flow_samples_0.002.npz')
        flow_path_01  = os.path.join(sample_dir, 'flow_samples_0.01.npz')
        assert os.path.isfile(flow_path_002), flow_path_002
        assert os.path.isfile(flow_path_01), flow_path_01

    # Verify flow
    for label in tqdm(labels):
        sample_dir = os.path.join(ROOT, label['dataset'], label['identity_name'], label['animation_name'], label['sample_id'])
        assert os.path.isdir(sample_dir), sample_dir

        flow_path_002 = os.path.join(sample_dir, 'flow_samples_0.002.npz')
        flow_path_01  = os.path.join(sample_dir, 'flow_samples_0.01.npz')
        assert os.path.isfile(flow_path_002), flow_path_002
        assert os.path.isfile(flow_path_01), flow_path_01