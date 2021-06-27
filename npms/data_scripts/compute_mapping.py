import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import config as cfg


def compute_train_to_augmented_mapping(
    data_base_dir, splits_dir,
    augmented_dataset_name,
    train_dataset_name
):

    # Augmented
    augmented_labels_json = os.path.join(data_base_dir, splits_dir, augmented_dataset_name, "labels_tpose.json")

    with open(augmented_labels_json) as f:
        augmented_labels = json.loads(f.read())

    # Train
    train_labels_json = os.path.join(data_base_dir, splits_dir, train_dataset_name, "labels_tpose.json")

    with open(train_labels_json) as f:
        train_labels = json.loads(f.read())

    train_to_augmented = {}

    for i, train_label in enumerate(train_labels):
        for j, augmented_label in enumerate(augmented_labels):
            if train_label['identity_name'] == augmented_label['identity_name']:
                train_to_augmented[i] = j

    # for train_id, augmented_id in train_to_augmented.items():
    #     print(train_id, augmented_id)

    return train_to_augmented

if __name__ == "__main__":
    print

    ##############################################################################
    # OPTIONS:
    # augmented_dataset_name = "SHAPE__amass-452id__mixamo_trans_all-225id"
    # train_dataset_name     = "POSE_MIX__amass-20000ts__mixamo_trans_all-20000ts"
    # train_dataset_name     = "POSE_MIX__amass-151698ts__mixamo_trans_all-210585ts"

    augmented_dataset_name = "SHAPE_MIX__A-amass-419id__B-mixamo_trans_all-205id"
    train_dataset_name = "POSE_MIX__amass-10000ts__mixamo_trans_all-30000ts"
    ##############################################################################

    from utils.parsing_utils import get_dataset_type_from_dataset_name
    dataset_type = get_dataset_type_from_dataset_name(train_dataset_name)
    splits_dir = f"{cfg.splits_dir}_{dataset_type}"

    train_to_augmented = compute_train_to_augmented_mapping(
        cfg.data_base_dir, splits_dir,
        augmented_dataset_name,
        train_dataset_name,
    )

    # Store
    train_to_augmented_json = os.path.join(cfg.data_base_dir, cfg.splits_dir, train_dataset_name, "train_to_augmented.json")
    with open(train_to_augmented_json, 'w') as f:
        json.dump(train_to_augmented, f, indent=4)