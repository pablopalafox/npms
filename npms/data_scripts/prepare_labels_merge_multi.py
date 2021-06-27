import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import shutil

import config as cfg


if __name__ == "__main__":
    OVERWRITE = True

    # --------------------------------------------------------------------------------------- #
    # dataset_name_dict = {
    #     # "amass": "AMASS-POSE-TRAIN-50id-subsampled-5000ts", 
    #     "mixamo": "MIXAMO_TRANS_ALL-POSE-TRAIN-165id-subsampled-20000ts", 
    #     "cape": "CAPE-POSE-TRAIN-35id-20533ts-543seqs", 
    # }

    # dataset_name_dict = {
    #     "amass": "AMASS-POSE-TRAIN-50id-subsampled-4309ts", 
    #     "mixamo": "MIXAMO_TRANS_ALL-POSE-TRAIN-165id-subsampled-10065ts", 
    #     "cape": "CAPE-POSE-TRAIN-35id-subsampled-10119ts", 
    # }

    # dataset_name_dict = {
    #     "amass": "AMASS-POSE-TRAIN-50id-10349ts-1319seqs", 
    #     "mixamo": "MIXAMO_TRANS_ALL-POSE-TRAIN-165id-201115ts-1025seqs", 
    #     "cape": "CAPE-POSE-TRAIN-35id-20533ts-543seqs", 
    # }

    dataset_name_dict = {
        "amass": "AMASS-POSE-TRAIN-50id-subsampled-1807ts", 
        "mixamo": "MIXAMO_TRANS_ALL-POSE-TRAIN-165id-subsampled-5115ts", 
        "cape": "CAPE-POSE-TRAIN-35id-subsampled-4807ts", 
    }
    
    
    # --------------------------------------------------------------------------------------- #

    ROOT = f"/cluster/lothlann/ppalafox/datasets"

    from utils.parsing_utils import get_dataset_type_from_dataset_name
    dataset_type = get_dataset_type_from_dataset_name(list(dataset_name_dict.keys())[0])
    splits_dir = f"{cfg.splits_dir}_{dataset_type}"
    
    info_per_dataset = {}

    ################################################################################
    # First, merge the tposes
    ################################################################################
    updated_labels_tpose = []
    for dataset_generic_name, dataset_name in dataset_name_dict.items():

        labels_tpose_json = os.path.join(ROOT, splits_dir, dataset_name, "labels_tpose.json")
        with open(labels_tpose_json, 'r') as f:
            labels_tpose = json.loads(f.read())   

        info_per_dataset[dataset_generic_name] = {'num_identities': len(labels_tpose)}

        for lt in labels_tpose:
            tmp_label = {
                'dataset':        lt['dataset'],
                'identity_id':    len(updated_labels_tpose),
                'identity_name':  lt['identity_name'],
                'animation_name': lt['animation_name'],
                'sample_id':      lt['sample_id'],
            }
            updated_labels_tpose.append(tmp_label)

    ################################################################################
    # Second, merge the poses
    ################################################################################

    updated_labels = []
    for dataset_generic_name, dataset_name in dataset_name_dict.items():

        labels_json = os.path.join(ROOT, splits_dir, dataset_name, "labels.json")
        with open(labels_json, 'r') as f:
            labels = json.loads(f.read())  

        # For generating the name later
        info_per_dataset[dataset_generic_name]['num_frames'] = len(labels)

        # Go over the labels of the current dataset and append it to the list
        for l in labels:

            # Find the identity_id of the current pose in updated_labels_tpose

            for ii, ult in enumerate(updated_labels_tpose):
                if ult['identity_name'] == l['identity_name']:
                    identity_id = ii

            tmp_label = {
                'dataset':        l['dataset'],
                'identity_id':    identity_id,
                'identity_name':  l['identity_name'],
                'animation_name': l['animation_name'],
                'sample_id':      l['sample_id'],
            }
            updated_labels.append(tmp_label)
    

    for a, info in info_per_dataset.items():
        print(a, info)

    ################################################################################
    # Store merged labels
    ################################################################################
    
    # Create dir name for the merged labels
    labels_dir = os.path.join(ROOT, splits_dir, f"MIX-POSE")
    for dn, info in info_per_dataset.items():
        num_identities = info['num_identities']
        num_frames     = info['num_frames']
        labels_dir += f"__{dn.upper()}-{num_identities}id-{num_frames}"

    if OVERWRITE or not os.path.exists(labels_dir) and len(updated_labels) > 0:

        if os.path.exists(labels_dir):
            shutil.rmtree(labels_dir)
        
        os.makedirs(labels_dir)
        
        labels_json       = os.path.join(labels_dir, "labels.json")
        labels_tpose_json = os.path.join(labels_dir, "labels_tpose.json")
        
        with open(labels_json, 'w') as f:
            json.dump(updated_labels, f, indent=4)

        with open(labels_tpose_json, 'w') as f:
            json.dump(updated_labels_tpose, f, indent=4)

        print("Done", labels_json)

    else:
        print(f"Labels {labels_dir} already exists! Didn't overwrite it.")

    print("Total num train samples:", len(updated_labels))
    print("Label generation done!")