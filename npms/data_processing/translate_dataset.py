import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import trimesh
import numpy as np
import glob
import json
import open3d as o3d

import multiprocessing as mp
from multiprocessing import Pool

import argparse
import traceback
import shutil

import utils.pcd_utils as pcd_utils
import config as cfg


def translate(mesh_path):
    assert os.path.isfile(mesh_path), mesh_path
    # Load mesh
    mesh = trimesh.load(mesh_path, process=False)
    # Apply delta
    mesh.apply_translation(delta)

    ##########################################################################################
    if viz:
        center_mass_current = np.copy(mesh.center_mass)
        
        sphere_current = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        sphere_current = sphere_current.translate(center_mass_current, relative=True)
        sphere_current.paint_uniform_color([1.0, 0.0, 1.0])

        mesh_o3d = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(mesh.vertices),
            o3d.utility.Vector3iVector(mesh.faces)
        )
        num_triangles = np.array(mesh_o3d.triangles).shape[0]
        mesh_o3d = mesh_o3d.simplify_quadric_decimation(int(num_triangles / 100))
        mesh_wireframe_o3d = o3d.geometry.LineSet.create_from_triangle_mesh(mesh_o3d)

        o3d.visualization.draw_geometries([world_frame, sphere, mesh_amass])
        o3d.visualization.draw_geometries([world_frame, sphere_current, mesh_wireframe_o3d])
        o3d.visualization.draw_geometries([world_frame, sphere, sphere_current, mesh_amass, mesh_wireframe_o3d])
    ##########################################################################################
    
    return mesh

def translate_mesh(path_dict):
    try:

        in_path  = path_dict['src']
        out_path = path_dict['tgt']

        if not os.path.isdir(out_path):
            os.makedirs(out_path)

        if "SPLITS" in in_path or in_path.endswith("json") or in_path.endswith("txt") or in_path.endswith("npz"):
            return

        #################################
        # Translate mesh
        #################################
        in_mesh_path = os.path.join(in_path, MESH_FILENAME)
        out_mesh_path = os.path.join(out_path, MESH_FILENAME)

        if not OVERWRITE and os.path.isfile(out_mesh_path):
            print("------ Skipping", in_path)
            return

        mesh = translate(in_mesh_path)
        mesh.export(out_mesh_path)

        #################################
        # Translate watertight mesh
        #################################
        in_mesh_watertight_path = os.path.join(in_path, 'mesh_watertight_poisson.ply')
        if "a_t_pose" in in_path: assert os.path.isfile(in_mesh_watertight_path)
        if os.path.isfile(in_mesh_watertight_path):
            mesh_watertight = translate(in_mesh_watertight_path)
            # Export
            out_mesh_watertight_path = os.path.join(out_path, 'mesh_watertight_poisson.ply')
            mesh_watertight.export(out_mesh_watertight_path)

        print("Processed", in_path)

    except:
        print('\t------------ Error with {}: {}'.format(in_path, traceback.format_exc()))


def compute_delta_mixamo_to_amass(mesh_path, center_mass_amass):
    mesh = trimesh.load(mesh_path, process=False)
    center_mass_mixamo = np.copy(mesh.center_mass)
    delta = center_mass_amass - center_mass_mixamo
    return delta


def get_center_mass_amass(mesh_path):
    mesh = trimesh.load(mesh_path, process=False)
    mesh_center_mass = np.copy(mesh.center_mass)
    mesh_o3d = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(mesh.vertices),
        o3d.utility.Vector3iVector(mesh.faces)
    )
    mesh_wireframe_o3d = o3d.geometry.LineSet.create_from_triangle_mesh(mesh_o3d)
    return mesh_center_mass, mesh_wireframe_o3d


if __name__ == '__main__':

    #####################################################################
    # Set up
    #####################################################################

    viz = False
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

    # Data to which we will align
    ROOT_amass  = f'/cluster/lothlann/ppalafox/datasets/amass'
    
    # -----------------------------------------------------------------------------
    
    ROOT_HDD = f'/cluster_HDD/lothlann/ppalafox/datasets'
    # ROOT_mixamo = os.path.join(ROOT_HDD, "mixamo")
    # ROOT_mixamo_trans = f'/cluster/lothlann/ppalafox/datasets/mixamo_trans_all'

    DATA_NAME = "dfaust" # mixamo

    ROOT_DATA_SRC = os.path.join(ROOT_HDD, DATA_NAME)
    ROOT_DATA_TGT = f'/cluster/lothlann/ppalafox/datasets/{DATA_NAME}'

    MESH_FILENAME = "mesh_raw.ply"
    
    # -----------------------------------------------------------------------------

    print()
    print(f"Translating {ROOT_DATA_SRC}")
    print(f"...into     {ROOT_DATA_TGT}")
    print(f"...using    {ROOT_amass} as reference.")
    print()

    if not os.path.isdir(ROOT_DATA_TGT):
        os.makedirs(ROOT_DATA_TGT)

    #####################################################################
    # Find center_mass of amass tposes 
    #####################################################################
    # mesh_path_ref = os.path.join(ROOT_amass, "ACCAD_s004", "a_t_pose", "000000", "mesh_normalized.ply")
    mesh_path_ref = os.path.join(ROOT_amass, "Transitionsmocap_s003", "a_t_pose", "000000", "mesh_raw.ply")
    center_mass_amass, mesh_amass = get_center_mass_amass(mesh_path_ref)

    if viz:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        sphere = sphere.translate(center_mass_amass, relative=True)
        sphere.paint_uniform_color([1.0, 0.0, 0.0])

        world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

    # Delta that transforms points from mixamo's frame to amass' frame
    # mesh_path_src = os.path.join(ROOT_DATA_SRC, "sophie", "a_t_pose", "000000", "mesh_watertight_poisson.ply")
    mesh_path_src = os.path.join(ROOT_DATA_SRC, "50021", "chicken_wings", "000000", "mesh_raw.ply")
    delta = compute_delta_mixamo_to_amass(mesh_path_src, center_mass_amass)

    #####################################################################
    # Characters
    #####################################################################
    if False:
        # dataset_name = "AUG_filtered_subsampled-train-20000ts"
        # dataset_name = "AUG_filtered-train-175id-210585ts-1125seqs"
        # dataset_name = "TEST-sophie-female_salsa_dancing-test-1id-90ts-1seqs"
        # dataset_name = "TEST-sophie-female_salsa_dancing-test-1id-90ts-1seqs"
        dataset_name = "MIXAMO-POSE-TEST-MOVING_FOR_TRANSLATION-2id-917ts-7seqs"
        
        from utils.parsing_utils import get_dataset_type_from_dataset_name
        dataset_type = get_dataset_type_from_dataset_name(dataset_name)
        splits_dir = f"{cfg.splits_dir}_{dataset_type}"
        
        labels_json       = os.path.join(ROOT_HDD, splits_dir, dataset_name, "labels.json")
        labels_tpose_json = os.path.join(ROOT_HDD, splits_dir, dataset_name, "labels_tpose.json")
        
        with open(labels_json, "r") as f:
            labels = json.loads(f.read())

        with open(labels_tpose_json, "r") as f:
            labels_tpose = json.loads(f.read())

        labels = labels + labels_tpose
    else:
        # dataset_name = "DFAUST-POSE-TEST-50021-chicken_wings-1id-17ts-1seqs"
        # dataset_name = "DFAUST-POSE-TEST-50021-knees-1id-664ts-1seqs"
        # dataset_name = "DFAUST-POSE-TEST-50021-one_leg_jump-1id-248ts-1seqs"
        # dataset_name = "DFAUST-POSE-TEST-50021-punching-1id-253ts-1seqs"
        # dataset_name = "DFAUST-POSE-TEST-50021-shake_arms-1id-273ts-1seqs"
        # dataset_name = "DFAUST-POSE-TEST-50021-shake_shoulders-1id-337ts-1seqs"
        # dataset_name = "DFAUST-POSE-TEST-50021-hips-1id-581ts-1seqs"
        # dataset_name = "DFAUST-POSE-TEST-50021-light_hopping_stiff-1id-216ts-1seqs"
        # dataset_name = "DFAUST-POSE-TEST-50021-one_leg_loose-1id-224ts-1seqs"
        # dataset_name = "DFAUST-POSE-TEST-50021-running_on_spot-1id-226ts-1seqs"
        dataset_name = "DFAUST-POSE-TEST-50021-shake_hips-1id-274ts-1seqs"


        from utils.parsing_utils import get_dataset_type_from_dataset_name
        dataset_type = get_dataset_type_from_dataset_name(dataset_name)
        splits_dir = f"{cfg.splits_dir}_{dataset_type}"
        
        labels_json = os.path.join(ROOT_HDD, splits_dir, dataset_name, "labels.json")
        
        with open(labels_json, "r") as f:
            labels = json.loads(f.read())

        labels = labels


    path_to_samples = []
    for label in labels:
        path_to_samples.append(
            {
                "src": os.path.join(ROOT_DATA_SRC, label['identity_name'], label['animation_name'], label['sample_id']),
                "tgt": os.path.join(ROOT_DATA_TGT, label['identity_name'], label['animation_name'], label['sample_id']),
            }
        )

    print()
    print("Number of frames to process:", len(path_to_samples))
    print()

    # #####################################################################
    # # Process
    # #####################################################################
    print()
    print("Jobs", n_jobs)
    print()

    input("Continue?")

    try:
        p = Pool(n_jobs)
        p.map(translate_mesh, path_to_samples)
    finally:
        p.close()
        p.join()
        