import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import numpy as np
import shutil
import config as cfg


def compute_labels_pose_encoder_training(data_base_dir, dataset, splits_dir, verbose=False, return_split_dir=False):
    input_labels_json       = os.path.join(data_base_dir, splits_dir, dataset, "labels.json")
    input_labels_tpose_json = os.path.join(data_base_dir, splits_dir, dataset, "labels_tpose.json")

    assert os.path.isfile(input_labels_json), input_labels_json
    assert os.path.isfile(input_labels_tpose_json), input_labels_tpose_json

    with open(input_labels_json, "r") as f:
        labels = json.loads(f.read())

    with open(input_labels_tpose_json, "r") as f:
        labels_tpose = json.loads(f.read())

    input_num_timesteps = len(labels)
    train_frames = int(0.9 * input_num_timesteps)
    val_frames = input_num_timesteps - train_frames
    
    if verbose:
        print(input_num_timesteps, train_frames, val_frames)

    labels_train = labels[:train_frames]
    labels_val = labels[train_frames:]

    if return_split_dir:
        return labels_train, labels_val, labels_tpose, os.path.join(data_base_dir, cfg.splits_dir)

    return labels_train, labels_val, labels_tpose
    

if __name__ == "__main__":

    ##################################################################################
    data_base_dir = "/cluster/lothlann/ppalafox/datasets"

    dataset = "POSE_MIX__amass-20000ts__mixamo_trans_all-20000ts"
    ##################################################################################

    labels_train, labels_val, labels_tpose, splits_root_dir = prepare_labels_pose_encoder_training(data_base_dir, dataset, verbose=True, return_split_dir=True)

    ########
    # TRAIN
    ########
    output_train_name = f"{dataset}-POSE-train"
    output_train_dir = os.path.join(splits_root_dir, output_train_name)
    if not os.path.isdir(output_train_dir):
        os.mkdir(output_train_dir)

    # Save json
    output_train_labels_json = os.path.join(output_train_dir, "labels.json")
    with open(output_train_labels_json, 'w') as f:
        json.dump(labels_train, f, indent=4)

    # Copy tpose
    output_train_tpose_json = os.path.join(output_train_dir, "labels_tpose.json")
    with open(output_train_tpose_json, 'w') as f:
        json.dump(labels_tpose, f, indent=4)

    print(output_train_tpose_json)

    ########
    # VAL
    ########
    output_val_name = f"{dataset}-POSE-val"
    output_val_dir = os.path.join(splits_root_dir, output_val_name)
    if not os.path.isdir(output_val_dir):
        os.mkdir(output_val_dir)

    # Save json
    output_val_labels_json = os.path.join(output_val_dir, "labels.json")
    with open(output_val_labels_json, 'w') as f:
        json.dump(labels_val, f, indent=4)

    # Copy tpose
    output_val_tpose_json = os.path.join(output_val_dir, "labels_tpose.json")
    with open(output_val_tpose_json, 'w') as f:
        json.dump(labels_tpose, f, indent=4)

    print(output_val_tpose_json)
