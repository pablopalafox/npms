import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import subprocess as sp
import json

import multiprocessing as mp
from multiprocessing import Pool

import argparse
import os, sys
import traceback

import config as cfg_general
from utils.parsing_utils import get_dataset_type_from_dataset_name


def sample_sdf(in_path):
    try:

        if "SPLITS" in in_path or in_path.endswith("json") or in_path.endswith("txt") or in_path.endswith("npz"):
            return

        if "a_t_pose" not in in_path:
            return

        surf_file_path = os.path.join(in_path, "samples_surface.pts")

        if not OVERWRITE and os.path.isfile(surf_file_path):
            print("Skipping", in_path)
            return

        print("Processing", in_path)

        # Load mesh
        in_watertight_mesh_path = os.path.join(in_path, mesh_watertight_filename)

        if not os.path.isfile(in_watertight_mesh_path):
            print(f'Watertight mesh {in_watertight_mesh_path} does not exist')
            return

        #####################################################################################
        # Compute coarse shape
        #####################################################################################
        sp.check_output(
            f'{grid_script} {in_watertight_mesh_path} {external_root}',
            shell=True
        )

        print('Done with {}'.format(in_path))

    except:
        print('\t------------ Error with {}: {}'.format(in_path, traceback.format_exc()))


if __name__ == '__main__':

    #####################################################################
    # Set up
    #####################################################################

    OVERWRITE = False

    parser = argparse.ArgumentParser(
        description='Run boundary sampling'
    )
    parser.add_argument('-t', '-max_threads', dest='max_threads', type=int, default=-1)

    args = parser.parse_args()

    try:
        n_jobs = int(os.environ['SLURM_CPUS_ON_NODE'])
        assert args.max_threads != 0
        if args.max_threads > 0:
            n_jobs = args.max_threads
    except:
        n_jobs = 1

    print()
    print(f"Using {n_jobs} ...")
    print()

    workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    external_root = os.path.abspath(os.path.join(workspace_dir, "..", "external"))
    grid_script   = os.path.join(workspace_dir, "data_processing/sample_sdf.sh")

    ##########################################
    # Options
    dataset = "cape" #"amass"

    if "mixamo" in dataset:
        mesh_watertight_filename = "mesh_watertight_poisson.ply"
    else:
        mesh_watertight_filename = "mesh_normalized.ply"

    ##########################################

    datasets_root = os.path.join(cfg_general.ROOT, "datasets")

    # -----------------------------------------------------------------------------------------------
    
    dataset_name = "CAPE-SHAPE-TRAIN-35id"

    # -----------------------------------------------------------------------------------------------
    
    input(f"Dataset name: {dataset_name}?")
    
    dataset_type = get_dataset_type_from_dataset_name(dataset_name)
    splits_dir = f"{cfg_general.splits_dir}_{dataset_type}"

    labels_tpose_json = os.path.join(datasets_root, splits_dir, dataset_name, "labels_tpose.json")
    
    with open(labels_tpose_json, "r") as f:
        labels_tpose = json.loads(f.read())

    # -----------------------------------------------------------------------------------------------
    
    path_to_samples = []
    for label in labels_tpose:
        path_to_samples.append(
            os.path.join(datasets_root, label['dataset'], label['identity_name'], label['animation_name'], label['sample_id'])
        )

    try:
        p = Pool(n_jobs)
        p.map(sample_sdf, path_to_samples)
    finally: # To make sure processes are closed in the end, even if errors happen
        p.close()
        p.join()

