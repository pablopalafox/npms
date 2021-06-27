import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import random
import numpy as np
import shutil
import config as cfg
import data_scripts.config_data as cfg_data
from utils.utils import filter_identities


def compute_labels_shape_encoder_training(data_base_dir, dataset, splits_dir, verbose=False):

    input_labels_json       = os.path.join(data_base_dir, splits_dir, dataset, "labels.json")
    input_labels_tpose_json = os.path.join(data_base_dir, splits_dir, dataset, "labels_tpose.json")

    assert os.path.isfile(input_labels_json), input_labels_json
    assert os.path.isfile(input_labels_tpose_json), input_labels_tpose_json

    with open(input_labels_json, "r") as f:
        labels = json.loads(f.read())

    with open(input_labels_tpose_json, "r") as f:
        labels_tpose = json.loads(f.read())

    ##########################################################################################
    # Set number of train / val identities
    ##########################################################################################
    input_num_identities = len(labels_tpose)

    num_train_identities = int(0.95 * input_num_identities) - 1
    num_val_identities = input_num_identities - num_train_identities

    if verbose:
        print(input_num_identities, num_train_identities, num_val_identities)

    ##########################################################################################
    # Prepare all identities that we can actually train on, ie, those from the training set
    ##########################################################################################
    all_valid_identities = [
        l['identity_name'] for l in labels_tpose
    ]

    # Just in case, filter test identities and identities that don't have pose training data (identities_augmented)
    all_valid_identities = filter_identities(all_valid_identities, cfg_data.test_identities_mixamo)
    all_valid_identities = filter_identities(all_valid_identities, cfg_data.identities_augmented)
    all_valid_identities = filter_identities(all_valid_identities, cfg_data.test_identities_amass)

    ##########################################################################################
    # Now separate those training identities into a train and val set to train the encoder 
    ##########################################################################################
    all_valid_identities = {ti: i for i, ti in enumerate(all_valid_identities)}

    train_identities = all_valid_identities.copy()
    val_identities = {}
    for k in range(num_val_identities):
        val_identity_name, val_identity_id = train_identities.popitem()
        val_identities[val_identity_name] = val_identity_id

    assert len(train_identities) + len(val_identities) == len(all_valid_identities), f"{len(train_identities) + len(val_identities)} vs {len(all_valid_identities)}"

    if verbose:
        for a in train_identities:
            print(a)

        print()
        for a in val_identities:
            print(a)
    
        print("num train identities", len(train_identities))
        print("num val identities  ", len(val_identities))

        input("Continue?")

    def get_sample_dataset(identity_name):
        for ltp in labels_tpose:
            if identity_name == ltp['identity_name']:
                return ltp['dataset']

    ##################################################################################
    # T-Pose labels
    ##################################################################################

    # tpose train
    labels_tpose_train = []
    for identity_id, train_identity_name in enumerate(train_identities):
        
        sample_dataset = get_sample_dataset(train_identity_name)

        sample_tpose = {
            "dataset": sample_dataset,
            "identity_id": identity_id,
            "identity_name": train_identity_name,
            "animation_name": "a_t_pose",
            "sample_id": "000000"
        }
        labels_tpose_train.append(sample_tpose)

    # tpose val
    labels_tpose_val = []
    for identity_id, val_identity_name in enumerate(val_identities):
        
        sample_dataset = get_sample_dataset(train_identity_name)
        
        sample_tpose = {
            "dataset": sample_dataset,
            "identity_id": identity_id,
            "identity_name": val_identity_name,
            "animation_name": "a_t_pose",
            "sample_id": "000000"
        }
        labels_tpose_val.append(sample_tpose)

    ##################################################################################
    # Labels
    ##################################################################################

    # Go over the labels and organize samples into train and val
    labels_train = []
    labels_val = []

    for label in labels:
        if label['identity_name'] in train_identities.keys():
            # Update identity_id based on the new ordering
            train_identity_id = list(train_identities.keys()).index(label['identity_name'])
            label['identity_id'] = train_identity_id
            labels_train.append(label)

        else:
            # Update identity_id based on the new ordering
            val_identity_id = list(val_identities.keys()).index(label['identity_name'])
            label['identity_id'] = val_identity_id
            labels_val.append(label)

    return labels_train, labels_tpose_train, labels_val, labels_tpose_val
    

if __name__ == "__main__":

    data_base_dir = f"/cluster/lothlann/ppalafox/datasets"
    
    ##########################################################################################
    dataset_name = "AUG_filtered_subsampled-train-20000ts"
    ##########################################################################################

    labels_train, labels_tpose_train, labels_val, labels_tpose_val = compute_labels_shape_encoder_training(
        data_base_dir, dataset_name
    )

    ###########
    # TRAIN
    ###########
    output_train_name = f"{dataset_name}-SHAPE-train"
    output_train_dir = os.path.join(data_base_dir, cfg.splits_dir, output_train_name)
    if not os.path.isdir(output_train_dir):
        os.mkdir(output_train_dir)

    # Save json
    output_train_labels_json = os.path.join(output_train_dir, "labels.json")
    with open(output_train_labels_json, 'w') as f:
        json.dump(labels_train, f, indent=4)

    # Save json
    output_train_labels_tpose_json = os.path.join(output_train_dir, "labels_tpose.json")
    with open(output_train_labels_tpose_json, 'w') as f:
        json.dump(labels_tpose_train, f, indent=4)
   
    ###########
    # VAL
    ###########
    # output_val_name = f"{input_split_name}-SHAPE-val{len(labels_val)}ts"
    output_val_name = f"{dataset_name}-SHAPE-val"
    output_val_dir = os.path.join(data_base_dir, cfg.splits_dir, output_val_name)
    if not os.path.isdir(output_val_dir):
        os.mkdir(output_val_dir)

    # Save json
    output_val_labels_json = os.path.join(output_val_dir, "labels.json")
    with open(output_val_labels_json, 'w') as f:
        json.dump(labels_val, f, indent=4)

    # Save json
    output_val_labels_tpose_json = os.path.join(output_val_dir, "labels_tpose.json")
    with open(output_val_labels_tpose_json, 'w') as f:
        json.dump(labels_tpose_val, f, indent=4)
