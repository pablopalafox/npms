import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import shutil

from data_scripts import config_data as cfg
import config as cfg_general
from utils.utils import filter_identities


if __name__ == "__main__":

    cluster_name = "cluster"
    dataset = "smal" # "amass"

    data_base_dir = f"/{cluster_name}/lothlann/ppalafox/datasets"
    data_dir = f"{data_base_dir}/{dataset}"

    OVERWRITE = True

    train_labels_tpose = []
    train_labels = []

    ################################################################################################

    split = "train"

    if split == "train":

        target_identities = [
            d for d in os.listdir(data_dir) 
            if "ZSPLITS" not in d
            and not d.endswith("json")
            and not d.endswith("txt")
        ]
        target_identities = sorted(target_identities)

        if "mixamo" in dataset:
            target_identities = filter_identities(target_identities, cfg.test_identities_mixamo)
        elif "amass" in dataset:
            target_identities = filter_identities(target_identities, cfg.test_identities_amass)
        elif "mano" in dataset:
            target_identities = filter_identities(target_identities, cfg.test_identities_mano)
        elif "cape" in dataset:
            target_identities = filter_identities(target_identities, cfg.test_identities_cape)
        elif "smal" in dataset:
            target_identities = filter_identities(target_identities, cfg.test_identities_smal)
        else:
            raise Exception("Dataset not implemented")
        
    elif split == "test":
        if "mixamo" in dataset:
            target_identities = cfg.test_identities_mixamo
        elif "amass" in dataset:
            target_identities = cfg.test_identities_amass
        else:
            raise Exception("Dataset not implemented")
    
    for a in target_identities:
        print(a)

    print(len(target_identities))

    #####################################################################################

    # Name we give our resulting dataset
    dataset_name = f"{dataset.upper()}-SHAPE-{split.upper()}"

    ################################################################################################

    num_identities = len(target_identities)
    num_train_samples = 0
    num_train_seqs    = 0

    for identity_id, identity_name in enumerate(sorted(target_identities)):

        sample_tpose = {
            "dataset": dataset,
            "identity_id": identity_id,
            "identity_name": identity_name,
            "animation_name": "a_t_pose",
            "sample_id": "000000"
        }

        train_labels_tpose.append(sample_tpose)
        
        identity_path = os.path.join(data_dir, identity_name)

        all_animation_names = [
            m for m in sorted(os.listdir(identity_path))
            if m != cfg_general.splits_dir and not m.endswith('json') and not m.endswith('npz')
        ]

        # Go over all animations for our character
        for animation_name in all_animation_names:

            if "a_t_pose" not in animation_name:
                continue

            assert animation_name != cfg_general.splits_dir

            animation_dir = os.path.join(identity_path, animation_name)

            assert os.path.isdir(animation_dir), animation_dir

            # Go over all samples in current animation
            for sample_id in sorted(os.listdir(animation_dir)):

                sample_dir = os.path.join(animation_dir, sample_id)
                
                if not os.path.isdir(sample_dir):
                    continue

                try:
                    sample_id_int = int(sample_id)
                except:
                    print(f"Skipping {sample_id}")
                    continue

                sample = {
                    "dataset": dataset,
                    'identity_id': identity_id,
                    'identity_name': identity_name,
                    'animation_name': animation_name,
                    'sample_id': sample_id 
                }

                # TRAIN set
                num_train_samples += 1
                
                train_labels.append(sample)

            num_train_seqs += 1
            

    ###############################################################################################################
    ###############################################################################################################
    from utils.parsing_utils import get_dataset_type_from_dataset_name
    splits_dir = cfg_general.splits_dir + f"_{get_dataset_type_from_dataset_name(dataset_name)}"
    train_labels_dir = os.path.join(data_base_dir, splits_dir, f"{dataset_name}-{num_identities}id")
    
    if OVERWRITE or not os.path.exists(train_labels_dir) and len(train_labels) > 0:

        if os.path.exists(train_labels_dir):
            shutil.rmtree(train_labels_dir)
        
        os.makedirs(train_labels_dir)
        
        train_labels_json       = os.path.join(train_labels_dir, "labels.json")
        train_labels_tpose_json = os.path.join(train_labels_dir, "labels_tpose.json")
        
        with open(train_labels_json, 'w') as f:
            json.dump(train_labels, f, indent=4)

        with open(train_labels_tpose_json, 'w') as f:
            json.dump(train_labels_tpose, f, indent=4)

        print("Done", train_labels_json)

    else:
        print(f"Train labels {train_labels_dir} already exists! Didn't overwrite it.")

    print("Total num train samples:", num_train_samples)
    print("Label generation done!")