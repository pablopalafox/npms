import trimesh
import numpy as np
import glob
import subprocess as sp
import shutil
import skimage

import open3d as o3d

import multiprocessing as mp
from multiprocessing import Pool

# from joblib import Parallel, delayed

import argparse
import os, sys
import traceback
import json


from utils import gaps_utils


def compute_watertight_mesh(in_path):
    try:

        # Load mesh
        in_mesh_path = os.path.join(in_path, 'mesh_normalized.ply')
        out_mesh_path = os.path.join(in_path, 'mesh_watertight_gaps.ply')

        if not OVERWRITE and os.path.isfile(out_mesh_path):
            print("------------- Skipping", out_mesh_path)
            return

        print("Starting with", in_mesh_path)

        #####################################################################################
        # Compute coarse shape
        #####################################################################################
        grd_path = f"{in_path}/tmp.grd"
        sp.check_output(
            f'{grid_script} {in_mesh_path} {grd_path} {external_root} {spacing}',
            shell=True
        )

        # Read grid file
        tx, grd = gaps_utils.read_grd(grd_path)

        # Delete grid file
        os.remove(grd_path)

        # current format is z y x, so convert to x y z
        grd = np.transpose(grd, (2, 1, 0))

        vertices, faces, normals, _ = skimage.measure.marching_cubes_lewiner(grd, level=surface_lvl)
        # Normalize vertices to be between -0.5 and 0.5
        vertices = vertices / tx[0, 0]
        vertices = vertices - 0.5

        mesh_coarse = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
        assert mesh_coarse.is_watertight, "mesh is not watertight"
        mesh_coarse.export(out_mesh_path)

        print('Done with {}'.format(in_path))

    except:
        print('\t------------ Error with {}: {}'.format(in_path, traceback.format_exc()))


if __name__ == '__main__':

    OVERWRITE = True

    parser = argparse.ArgumentParser(
        description='Run point cloud sampling'
    )

    parser.add_argument('-res', type=int, required=True)
    parser.add_argument('-surface_lvl', type=float, default=0.001)
    parser.add_argument('-t', '-max_threads', dest='max_threads', type=int, default=-1)

    args = parser.parse_args()

    try:
        n_jobs = int(os.environ['SLURM_CPUS_ON_NODE'])
    except:
        n_jobs = -1
    assert args.max_threads != 0
    if args.max_threads > 0:
        n_jobs = args.max_threads

    print()
    print(f"Using {n_jobs} ...")
    print()

    #####################################################################
    # Set up
    #####################################################################

    surface_lvl = args.surface_lvl
    spacing = 1.0 / args.res


    workspace_dir = "/rhome/ppalafox/workspace/if-net/"
    external_root = os.path.join(workspace_dir, "external")
    # manifold_exec = os.path.join(workspace_dir, "external/ManifoldPlus/build/manifold")
    grid_script   = os.path.join(workspace_dir, "data_processing/process_mesh.sh")

    # ------------------------------------------------------------------------------------
    dataset_name = "MIXAMO_TRANS_ALL-POSE-TRAIN-165id-subsampled-10065ts"
    # ------------------------------------------------------------------------------------

    # Mesh type used to generate the data
    MESH_FILENAME = 'mesh_normalized.ply' # Select between 'mesh_normalized.ply' and 'mesh_real_scan.ply' (for CAPE)

    # -----------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------

    ROOT = f"/cluster/lothlann/ppalafox/datasets"

    from utils.parsing_utils import get_dataset_type_from_dataset_name
    dataset_type = get_dataset_type_from_dataset_name(dataset_name)
    splits_dir = f"ZSPLITS_{dataset_type}"

    labels_json = os.path.join(ROOT, splits_dir, dataset_name, "labels.json")
    
    with open(labels_json, "r") as f:
        labels = json.loads(f.read())

    # -----------------------------------------------------------------------------------------------
    
    path_to_samples = []
    for label in labels:
        path_to_samples.append(
            os.path.join(ROOT, label['dataset'], label['identity_name'], label['animation_name'], label['sample_id'])
        )

    print("Number of samples to process", len(path_to_samples))
    input("Continue?")

    ########################################################################

    try:
        p = Pool(n_jobs)
        p.map(compute_watertight_mesh, path_to_samples)
    finally: # To make sure processes are closed in the end, even if errors happen
        p.close()
        p.join()