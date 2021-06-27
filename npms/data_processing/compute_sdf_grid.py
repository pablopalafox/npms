import trimesh
import numpy as np
import glob
import subprocess as sp
import shutil
import skimage
import json

import open3d as o3d

import multiprocessing as mp
from multiprocessing import Pool

# from joblib import Parallel, delayed

import argparse
import os, sys
import traceback


from utils import gaps_utils
import config as cfg


def compute_sdf_grid(in_path):
    try:

        if "SPLITS" in in_path or in_path.endswith("json") or in_path.endswith("txt") or in_path.endswith("npz"):
            return

        print("Processing", in_path)

        out_grid_path = f"{in_path}/sdf_{args.res}.grd"

        if not OVERWRITE and os.path.exists(out_grid_path):
            print('Skipping {}'.format(in_path))
            return
        
        # Load mesh
        # in_mesh_path = os.path.join(in_path, 'mesh_watertight_poisson.ply')
        in_mesh_path = os.path.join(in_path, 'mesh_normalized.ply')

        if not os.path.isfile(in_mesh_path):
            print("----------- Skipping", in_mesh_path)
            return

        #####################################################################################
        # Compute coarse shape
        #####################################################################################
        sp.check_output(
            f'{grid_script} {in_mesh_path} {out_grid_path} {external_root} {spacing}',
            shell=True
        )

        # tx, grd = gaps_utils.read_grd(tmp_grid_path)
        # grd = np.transpose(grd, (2, 1, 0)) # current format is z y x, so convert to x y z
        
        # os.remove(tmp_grid_path)

        # grd = np.reshape(grd, -1)
        
        # occ_grd = np.zeros(grd.shape[0], dtype=np.int8)
        # occ_grd[grd > 0]  = 0
        # occ_grd[grd <= 0] = 1

        # compressed_grd = np.packbits(occ_grd)

        # np.savez(
        #     out_grid_path, 
        #     compressed_occupancies=compressed_grd, 
        # )

        print('Done with {}'.format(in_path))

    except:
        print('\t------------ Error with {}: {}'.format(in_path, traceback.format_exc()))


if __name__ == '__main__':

    #####################################################################
    # Set up
    #####################################################################

    ON_SERVER = True
    OVERWRITE = False

    parser = argparse.ArgumentParser(
        description='Run boundary sampling'
    )
    parser.add_argument('-res', type=int, required=True)
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
    
    workspace_dir = "/rhome/ppalafox/workspace/neshpod/"

    external_root = os.path.join(workspace_dir, "external")
    grid_script   = os.path.join(workspace_dir, "data_processing/compute_sdf_grid_gaps.sh")

    ##########################################
    # Options
    # spacing = str(1/384)
    spacing = str(1/args.res)
    
    dataset = "dfaust"
    
    # dataset_name = "DFAUST-POSE-TEST-50021-chicken_wings-1id-17ts-1seqs"
    # dataset_name = "DFAUST-POSE-TEST-50021-knees-1id-17ts-1seqs"
    # dataset_name = "DFAUST-POSE-TEST-50021-one_leg_jump-1id-17ts-1seqs"
    # dataset_name = "DFAUST-POSE-TEST-50021-punching-1id-17ts-1seqs"
    # dataset_name = "DFAUST-POSE-TEST-50021-shake_arms-1id-17ts-1seqs"
    # dataset_name = "DFAUST-POSE-TEST-50021-shake_shoulders-1id-17ts-1seqs"
    # dataset_name = "DFAUST-POSE-TEST-50021-hips-1id-17ts-1seqs"
    # dataset_name = "DFAUST-POSE-TEST-50021-light_hopping_stiff-1id-17ts-1seqs"
    # dataset_name = "DFAUST-POSE-TEST-50021-one_leg_loose-1id-17ts-1seqs"
    # dataset_name = "DFAUST-POSE-TEST-50021-running_on_spot-1id-17ts-1seqs"
    # dataset_name = "DFAUST-POSE-TEST-50021-shake_hips-1id-17ts-1seqs"

    dataset_name = "CAPE-POSE-TEST-00032_shortlong-hips-1id-293ts-1seqs"
    # dataset_name = "CAPE-POSE-TEST-00032_shortshort-shoulders_mill-1id-207ts-1seqs"
    # dataset_name = "CAPE-POSE-TEST-03223_shortlong-shoulders_mill-1id-378ts-1seqs"
    # dataset_name = "CAPE-POSE-TEST-03223_shortshort-tilt_twist_left-1id-349ts-1seqs"
    ##########################################

    ROOT = f'/cluster/lothlann/ppalafox/datasets'

    from utils.parsing_utils import get_dataset_type_from_dataset_name
    dataset_type = get_dataset_type_from_dataset_name(dataset_name)
    splits_dir = f"{cfg.splits_dir}_{dataset_type}"
    
    labels_json = os.path.join(ROOT, splits_dir, dataset_name, "labels.json")
    
    with open(labels_json, "r") as f:
        labels = json.loads(f.read())

    paths_to_process = [os.path.join(ROOT, labels[0]['dataset'], labels[0]['identity_name'], "a_t_pose", "000000")]
    for label in labels:
        paths_to_process.append(
            os.path.join(ROOT, label['dataset'], label['identity_name'], label['animation_name'], label['sample_id'])
        )

    try:
        p = Pool(n_jobs)
        p.map(compute_sdf_grid, paths_to_process)
    finally:
        p.close()
        p.join()

