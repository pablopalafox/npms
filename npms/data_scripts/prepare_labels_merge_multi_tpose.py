import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import shutil

import config as cfg


if __name__ == "__main__":
    OVERWRITE = True
 
    data_base_dir = "/cluster/lothlann/ppalafox/datasets"
    
    # ------------------------------------------------------------------------ #

    # dataset_type_A = "amass"
    # dataset_name_A = "SHAPE_amass-train-419ts-419seqs"
    # labels_json_A = os.path.join(data_base_dir, cfg.splits_dir, dataset_name_A, "labels.json")

    # FIRST STEP (merge CAPE and MIXAMO)
    # dataset_type_A = "cape"
    # dataset_name_A = "CAPE-SHAPE-TRAIN-35id"
    # dataset_type_B = "mixamo_trans_all"
    # dataset_name_B = "MIXAMO_TRANS_ALL-SHAPE-TRAIN-215id"

    # SECOND STEP (merge previous MIX and AMASS)
    dataset_type_A = "cape_mixamo" # used only for the final name
    dataset_name_A = "MIX-SHAPE__CAPE-35id__MIXAMO_TRANS_ALL-215id"
    dataset_type_B = "amass" # used only for the final name
    dataset_name_B = "AMASS-SHAPE-TRAIN-50id"

    # ------------------------------------------------------------------------ #

    from utils.parsing_utils import get_dataset_type_from_dataset_name
    dataset_type = get_dataset_type_from_dataset_name(dataset_name_A)
    splits_dir = f"{cfg.splits_dir}_{dataset_type}"
    
    labels_json_A = os.path.join(data_base_dir, splits_dir, dataset_name_A, "labels.json")
    labels_json_B = os.path.join(data_base_dir, splits_dir, dataset_name_B, "labels.json")

    with open(labels_json_A, 'r') as f:
        labels_A = json.loads(f.read())

    with open(labels_json_B, 'r') as f:
        labels_B = json.loads(f.read())

    num_identities_A = len(labels_A)
    num_identities_B = len(labels_B)

    # We'll add a new field specifying the dataset type
    updated_labels = []

    identity_count = 0

    for label in labels_A:
        updated_label = {
            'dataset':        label['dataset'],
            'identity_id':    identity_count,
            'identity_name':  label['identity_name'],
            'animation_name': label['animation_name'],
            'sample_id':      label['sample_id'],
        }
        updated_labels.append(updated_label)
        
        identity_count += 1

    for label in labels_B:
        updated_label = {
            'dataset':        label['dataset'],
            'identity_id':    identity_count,
            'identity_name':  label['identity_name'],
            'animation_name': label['animation_name'],
            'sample_id':      label['sample_id'],
        }
        updated_labels.append(updated_label)
        
        identity_count += 1

    # Save
    labels_dir = os.path.join(data_base_dir, splits_dir, f"MIX-SHAPE__{dataset_type_A.upper()}-{num_identities_A}id__{dataset_type_B.upper()}-{num_identities_B}id")
    
    if OVERWRITE or not os.path.exists(labels_dir) and len(updated_labels) > 0:

        if os.path.exists(labels_dir):
            shutil.rmtree(labels_dir)
        
        os.makedirs(labels_dir)
        
        labels_json             = os.path.join(labels_dir, "labels.json")
        train_labels_tpose_json = os.path.join(labels_dir, "labels_tpose.json")
        
        with open(labels_json, 'w') as f:
            json.dump(updated_labels, f, indent=4)

        with open(train_labels_tpose_json, 'w') as f:
            json.dump(updated_labels, f, indent=4)

        print("Done", labels_json)

    else:
        print(f"Labels {labels_dir} already exists! Didn't overwrite it.")

    print("Total num train samples:", len(updated_labels))
    print("Label generation done!")