import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import numpy as np
import shutil
import random
from tqdm import tqdm
import math

import config as cfg


def subsample_even_json(input_labels_json, input_labels_tpose_json, starting_name, target_num_timesteps):

    with open(input_labels_json, "r") as f:
        labels = json.loads(f.read())

    with open(input_labels_tpose_json, "r") as f:
        labels_tpose = json.loads(f.read())

    num_poses_for_identity = math.ceil(target_num_timesteps / len(labels_tpose))

    print(num_poses_for_identity)

    subsampled_labels = []
    for identity_dict in tqdm(labels_tpose):
        count_current_identity = 0
        identity_id = identity_dict['identity_id']
        
        # Shuffle labels
        random.shuffle(labels)

        # Go over labels and find the first 'num_poses_for_identity' poses for the current identity
        for sample_dict in labels:
            sample_identity_id = sample_dict['identity_id']
            
            if sample_identity_id == identity_id:
                subsampled_labels.append(sample_dict)
                count_current_identity += 1
                
                if count_current_identity == num_poses_for_identity:
                    break

    identity_count_dict = {}
    for sl in subsampled_labels:
        if sl['identity_name'] in identity_count_dict:
            identity_count_dict[sl['identity_name']] += 1
        else:
            identity_count_dict[sl['identity_name']] = 1

    for identity_name, count in identity_count_dict.items():
        if count < 50:
            print("\t\t", identity_name, count)
        else:
            print(identity_name, count)
        # assert count > 50

    print()
    print("Total frames", len(subsampled_labels))
    print()

    output_dataset_name = f"{starting_name}-subsampled-{len(subsampled_labels)}ts"
    output_split_dir = os.path.join(splits_root_dir, output_dataset_name)

    # Save json
    os.makedirs(output_split_dir, exist_ok=True)
    output_labels_json = os.path.join(output_split_dir, "labels.json")
    with open(output_labels_json, 'w') as f:
        json.dump(subsampled_labels, f, indent=4)

    return output_dataset_name


def subsample_json(input_labels_json, starting_name, target_num_timesteps):

    output_dataset_name = f"{starting_name}-subsampled-{TARGET_NUM_TIMESTEPS}ts"
    output_split_dir = os.path.join(splits_root_dir, output_dataset_name)

    os.makedirs(output_split_dir, exist_ok=True)

    input(f"output_dataset_name: {output_dataset_name}")

    with open(input_labels_json, "r") as f:
        labels = json.loads(f.read())

    input_number_timesteps = len(labels)

    # Get some random indices
    random_idices = np.random.permutation(input_number_timesteps)[:target_num_timesteps]

    # Concert list of labels to a numpy array
    labels_np = np.array(labels)

    # Subsample the array
    subsampled_labels_np = labels_np[random_idices]
    
    # Convert back to python list
    subsampled_labels = list(subsampled_labels_np)

    identity_count_dict = {}
    for sl in subsampled_labels:
        if sl['identity_name'] in identity_count_dict:
            identity_count_dict[sl['identity_name']] += 1
        else:
            identity_count_dict[sl['identity_name']] = 0

    for identity_name, count in identity_count_dict.items():
        if count < 50:
            print("\t\t", identity_name, count)
        else:
            print(identity_name, count)
        # assert count > 50

    # Save json
    output_labels_json = os.path.join(output_split_dir, "labels.json")
    with open(output_labels_json, 'w') as f:
        json.dump(subsampled_labels, f, indent=4)
    
    return output_dataset_name
    

if __name__ == "__main__":

    ##############################################################################
    TARGET_NUM_TIMESTEPS = 2000
    
    cluster = "cluster_HDD"
    dataset_name = "AMASS-POSE-TRAIN-50id-10349ts-1319seqs"
    # dataset_name = "MIXAMO_TRANS_ALL-POSE-TRAIN-165id-201115ts-1025seqs"
    # dataset_name = "CAPE-POSE-TRAIN-35id-20533ts-543seqs"

    starting_name = dataset_name.split('-')
    starting_name = f"{starting_name[0]}-{starting_name[1]}-{starting_name[2]}-{starting_name[3]}"

    ##############################################################################

    from utils.parsing_utils import get_dataset_type_from_dataset_name
    splits_dir_name = cfg.splits_dir + f"_{get_dataset_type_from_dataset_name(dataset_name)}"
    
    splits_root_dir = f"/{cluster}/lothlann/ppalafox/datasets/{splits_dir_name}"

    input_labels_json       = os.path.join(os.path.join(splits_root_dir, dataset_name, "labels.json"))
    input_labels_tpose_json = os.path.join(os.path.join(splits_root_dir, dataset_name, "labels_tpose.json"))

    # Subsample the input json
    if True:
        output_dataset_name = subsample_even_json(
            input_labels_json, input_labels_tpose_json, starting_name, TARGET_NUM_TIMESTEPS
        )
    else:
        output_dataset_name = subsample_json(
            input_labels_json, output_split_dir, starting_name, TARGET_NUM_TIMESTEPS
        )

    # Copy the labels_tpose.json
    input_animation_to_seq_json = os.path.join(
        os.path.join(splits_root_dir, dataset_name, "labels_tpose.json")
    )

    output_animation_to_seq_json = os.path.join(
        os.path.join(splits_root_dir, output_dataset_name, "labels_tpose.json")
    )

    shutil.copyfile(input_animation_to_seq_json, output_animation_to_seq_json)
    
    print()
    print("Done!", output_dataset_name)