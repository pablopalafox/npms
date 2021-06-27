import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import shutil

import config as cfg


if __name__ == "__main__":
    OVERWRITE = True
 
    data_base_dir = "/cluster_HDD/lothlann/ppalafox/datasets_mix"
    labels_dir = os.path.join(data_base_dir, "ZSPLITS_HUMAN")
    
    # ------------------------------------------------------
    dataset_name = "amass"
    dataset = "POSE_amass_subsampled-train-20000ts"
    # ------------------------------------------------------
    
    labels_json = os.path.join(labels_dir, dataset, "labels.json")
    labels_tpose_json = os.path.join(labels_dir, dataset, "labels_tpose.json")

    with open(labels_json, 'r') as f:
        labels = json.loads(f.read())  

    with open(labels_tpose_json, 'r') as f:
        labels_tpose = json.loads(f.read())  

    new_labels = []
    for label in labels:
        label["dataset"] = dataset_name
        new_labels.append(label)

    new_labels_tpose = []
    for label_tpose in labels_tpose:
        label_tpose["dataset"] = dataset_name
        new_labels_tpose.append(label_tpose)

    with open(labels_json, 'w') as f:
        json.dump(new_labels, f, indent=4)

    with open(labels_tpose_json, 'w') as f:
        json.dump(new_labels_tpose, f, indent=4)