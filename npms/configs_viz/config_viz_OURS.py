import os
import json
import glob

from utils.utils import query_yes_no 

import config as cfg


#####################################################################################################################
# SET ME!!!
ROOT = "/cluster_HDD/lothlann/ppalafox"
# SET ME!!!
#####################################################################################################################

data_base_dir = f"{ROOT}/datasets_SSD@20_04_2021"

exp_version = "npms"
exps_dir  = os.path.join(ROOT, "experiments", exp_version)


def prepare_paths():
    #############################
    # Experiment
    #############################

    # ---------------------------------------------
    # ---------------------------------------------
    # ---------------------------------------------
    # ---------------------------------------------
    # ---------------------------------------------
    exp_name = ""
    run_name = ""
    # ---------------------------------------------
    # ---------------------------------------------
    # ---------------------------------------------
    # ---------------------------------------------
    # ---------------------------------------------
    
    # Extract dataset name
    tmp = run_name.split('__')
    dataset_name = tmp[1]

    #############################
    # Groundtruth data    
    #############################
    from utils.parsing_utils import get_dataset_type_from_dataset_name
    dataset_type = get_dataset_type_from_dataset_name(dataset_name)
    splits_dir = f"{cfg.splits_dir}_{dataset_type}"

    data_dir = f"{data_base_dir}/{splits_dir}/{dataset_name}"
    assert os.path.isdir(data_dir), data_dir

    with open(os.path.join(data_dir, "labels.json"), 'r') as f:
        labels = json.loads(f.read())

    exp_dir = os.path.join(exps_dir, exp_name, "optimization", run_name)
    predicted_meshes_dir_list = os.path.join(exp_dir, f"predicted_meshes*")
    print("\nWhich one do you want?")
    predicted_meshes_dir = None
    for tmp in sorted(glob.glob(predicted_meshes_dir_list)):
        print()
        answer = query_yes_no(tmp, default="no")
        if answer:
            predicted_meshes_dir = tmp
            break
    assert predicted_meshes_dir is not None, "Please select a folder to read from!"
    assert os.path.isdir(predicted_meshes_dir), predicted_meshes_dir

    assert dataset_name in run_name, "Make sure the dataset_name matches that on which we optimized over!"

    # Video dir
    video_dir = f"/cluster_HDD/lothlann/ppalafox/videos/{dataset_name}/ours/{exp_name}/{run_name}"

    #####################################################################################################
    # Prepare the paths to gt and pred
    #####################################################################################################
    gt_path_list   = []
    pred_path_list = []

    for frame_t, label in enumerate(labels):
        
        label = labels[frame_t]
        gt_dir = os.path.join(data_base_dir, label['dataset'], label['identity_name'], label['animation_name'], label['sample_id'])
        gt_mesh_path = os.path.join(gt_dir, "mesh_normalized.ply")
        gt_path_list.append(gt_mesh_path)

        frame_dir = os.path.join(predicted_meshes_dir, label['sample_id'])
        assert os.path.isdir(frame_dir), frame_dir
        pred_mesh_path = os.path.join(frame_dir, "ref_warped.ply")
        pred_path_list.append(pred_mesh_path)

    frame_rate = 30

    if "DFAUST" in run_name:
        frame_rate = 5

    return gt_path_list, pred_path_list, video_dir, False, frame_rate, 1.0, None