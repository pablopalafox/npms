import os, sys
import numpy as np
import time
import argparse
import traceback
import glob
import trimesh
import math
import shutil
import json
import open3d as o3d
from tqdm import tqdm

import ctypes
import logging
from contextlib import closing

import multiprocessing as mp
from multiprocessing import Pool

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config as cfg
from utils.pcd_utils import BBox


info = mp.get_logger().info


def compute_global_bbox():

    logger = mp.log_to_stderr()
    logger.setLevel(logging.INFO)

    # create shared array for bbox
    shared_bbox = mp.Array(ctypes.c_double, N*M)
    bbox = to_numpy_array(shared_bbox)

    # By updating bbox, we uppdate shared_bbox as well, since they share memory
    bbox = bbox.reshape((N, M))
    bbox[:, :] = np.array([[math.inf, math.inf, math.inf], [-math.inf, -math.inf, -math.inf]])

    #####################################################################################
    # Go over all animations
    #####################################################################################
    with closing(mp.Pool(processes=n_jobs, initializer=init, initargs=(shared_bbox,))) as p:
        # many processes access the same slice
        p.map_async(update_bbox, sample_dirs)
    
    p.join()
    p.close()

    print("done")
    final_bbox = to_numpy_array(shared_bbox)
    final_bbox = final_bbox.reshape((N, M))
    #####################################################################################
    #####################################################################################
        
    # assert np.all(np.isfinite(final_bbox)), final_bbox

    
    # Compute current extent
    p_min, p_max = final_bbox[0], final_bbox[1]
    non_cube_extent = p_max - p_min

    # Convert bbox to cube
    cube_extent = np.max(non_cube_extent) * np.ones_like(non_cube_extent)
    delta = cube_extent - non_cube_extent
    half_delta = delta / 2.0
    # Cube params
    p_min = p_min - half_delta
    extent = cube_extent
    # Enlarge bbox
    p_min = p_min - bbox_displacement
    extent = extent + 2.0 * bbox_displacement
    # Update bbox
    final_bbox[0] = p_min
    final_bbox[1] = p_min + extent

    # assert np.all(np.isfinite(final_bbox)), final_bbox

    # Store bbox
    print("Dumping into json file:", dataset_bbox_json)
    with open(dataset_bbox_json, 'w') as f:
        json.dump(final_bbox.tolist(), f, indent=4)

    return final_bbox


def init(shared_bbox_):
    global shared_bbox
    shared_bbox = shared_bbox_ # must be inherited, not passed as an argument


def to_numpy_array(mp_arr):
    return np.frombuffer(mp_arr.get_obj())


def update_bbox(sample_dir):

    print(sample_dir)

    """synchronized."""
    with shared_bbox.get_lock(): # synchronize access

        info(f"start {sample_dir}")

        mesh_raw_path = os.path.join(sample_dir, "mesh_raw.ply")
        assert os.path.exists(mesh_raw_path), f"update_bbox: {mesh_raw_path}"

        # Load meshes
        mesh = trimesh.load_mesh(mesh_raw_path, process=False, maintain_order=True)

        # Compute bbox of current mesh
        bbox_bounds = mesh.bounds
        bbox_min = bbox_bounds[0]
        bbox_max = bbox_bounds[1]

        # print(bbox_bounds)

        assert np.all(np.isfinite(bbox_bounds)), bbox_bounds

        # Update the total bbox after having taken into account the alignment to the origin
        bbox = to_numpy_array(shared_bbox)
        bbox = bbox.reshape((N, M))

        bbox[0] = np.minimum(bbox[0], bbox_min)
        bbox[1] = np.maximum(bbox[1], bbox_max)

        info(f"end {sample_dir}")

    
################################################################################################
################################################################################################
################################################################################################

def normalize_mesh(mesh):
    # Global normalization
    if compute_bbox:
        vertices = (mesh.vertices - p_min) / extent # now we're in [-1, 1]
        vertices = vertices - 0.5                   # now in [-0.5, 0.5] 
    else:
        vertices = mesh.vertices - trans
        vertices = scale * vertices
    
    mesh.vertices = vertices
    return mesh


def normalize_meshes(sample_dir):
    try:

        # Normal mesh
        mesh_raw_path = os.path.join(sample_dir, "mesh_raw.ply")
        if os.path.exists(mesh_raw_path):
            
            normalized_mesh_path = os.path.join(sample_dir, "mesh_normalized.ply")
            
            if OVERWRITE or not os.path.isfile(normalized_mesh_path):    
                mesh = trimesh.load_mesh(mesh_raw_path, process=False, maintain_order=True)
                mesh = normalize_mesh(mesh)    
                trimesh.Trimesh.export(mesh, normalized_mesh_path, 'ply')
                print("\tWriting mesh into:", normalized_mesh_path)

        if VIZ:
            mesh_o3d = o3d.io.read_triangle_mesh(normalized_mesh_path)
            mesh_o3d.compute_vertex_normals()
            o3d.visualization.draw_geometries([world_frame, unit_bbox, mesh_o3d])

        ###################################################################################
        # Real scan if exists
        real_scan_path = os.path.join(sample_dir, "mesh_real_scan.ply")
        if os.path.isfile(real_scan_path):
            mesh = trimesh.load_mesh(real_scan_path, process=False, maintain_order=True)
            mesh = normalize_mesh(mesh)        
            trimesh.Trimesh.export(mesh, real_scan_path, 'ply')
            print("\t\tWriting real scan  into:", real_scan_path)
        ###################################################################################

        ###################################################################################
        # Body mesh if exists
        body_mesh_raw_path = os.path.join(sample_dir, "mesh_body_raw.ply")
        if os.path.isfile(body_mesh_raw_path):
            
            body_mesh_normalized_path = os.path.join(sample_dir, "mesh_body_normalized.ply")
            
            if OVERWRITE_BODY or not os.path.isfile(body_mesh_normalized_path):
                mesh = trimesh.load_mesh(body_mesh_raw_path, process=False, maintain_order=True)
                mesh = normalize_mesh(mesh)        
                trimesh.Trimesh.export(mesh, body_mesh_normalized_path, 'ply')
                print("\t\tWriting body mesh into:", body_mesh_normalized_path)
        ###################################################################################

    except:
        print('\t------------ Error with {}: {}'.format(sample_dir, traceback.format_exc()))


if __name__ == '__main__':

    try:
        n_jobs = int(os.environ['SLURM_CPUS_ON_NODE'])
    except:
        n_jobs = 4

    print()
    print(f"Using {n_jobs} jobs")

    mp.freeze_support()

    p_min = -0.5
    p_max =  0.5

    # Flag to visualize meshes for debugging
    VIZ = False

    if VIZ:
        unit_bbox = BBox.compute_bbox_from_min_point_and_max_point(
            np.array([p_min]*3), np.array([p_max]*3)
        )

        world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.5, origin=[0, 0, 0]
        )

    #####################################################################################
    #####################################################################################
    dataset = 'cape'
    #####################################################################################
    #####################################################################################

    OVERWRITE_BBOX_COMPUTATION = False
    OVERWRITE                  = False # for the general mesh
    OVERWRITE_BODY             = False # for the body mesh, in case the dataset has such meshes

    #####################################################################################
    compute_bbox = False # Keep it to False, unless you wanna play with the normalization etc.
    #####################################################################################
    
    if not compute_bbox:
        input("Using predefined bbox and scale to normalize - Recommened!")
    else:
        input("Computing dataset-specific bbox to normalize")

    target_animations = []
        
    if compute_bbox:
        # bbox array dimensions ([[bbox_min], [bbox_max]])
        N, M = 2, 3

        # bbox displacement
        bbox_displacement = 0.01
    else:
        # scale
        scale = 1.0
        trans = 0.0
    
        if 'mano' in dataset:
            scale = 0.75
            bbox_displacement = 0.0
        
        elif 'cape' in dataset:
            scale = 0.4
            bbox_displacement = 0.0

            # Load our precomputed bbox to normalize the dataset to reside within a unit cube
            predefined_bbox_json_path = os.path.join("bbox.json")
            assert os.path.isfile(predefined_bbox_json_path)
            
            with open(predefined_bbox_json_path, 'r') as f:
                predefined_bbox = json.loads(f.read())
                predefined_bbox = np.array(predefined_bbox)
            
            trans = (predefined_bbox[0] + predefined_bbox[1]) / 2.
        
        else:
            print("dataset is not implemented")
            exit()
    ####################

    dataset_dir = os.path.join(cfg.ROOT, "datasets", dataset)
    assert os.path.isdir(dataset_dir), dataset_dir

    print("dataset_dir:", dataset_dir)

    # Prepare the list of all sample dirs
    sample_dirs = sorted(glob.glob(dataset_dir + "/*/*/*"))

    ########################################################################################################
    # 1. Compute global bbox
    ########################################################################################################
    if compute_bbox:
        dataset_bbox_json = os.path.join(dataset_dir, "bbox.json")

        if OVERWRITE_BBOX_COMPUTATION or not os.path.isfile(dataset_bbox_json):
            print()
            input("Need to compute bbox. Do I go ahead?")
            bbox = compute_global_bbox()
        
        else:
            print()
            input("Already had bbox - Load it?")
            with open(dataset_bbox_json, 'r') as f:
                bbox = json.loads(f.read())
                bbox = np.array(bbox)
        
        print("bbox ready!")
        print(bbox)

    ########################################################################################################
    # 2. Normalize meshes to lie within a common bbox
    ########################################################################################################
    print()
    print("#"*60)
    print(f"Will normalize {len(sample_dirs)} meshes!")
    print("#"*60)
    input("Continue?")

    if compute_bbox:
        p_min = bbox[0]
        p_max = bbox[1]
        extent = p_max - p_min

    p_norm = Pool(n_jobs)
    p_norm.map(normalize_meshes, sample_dirs)

    print("Done!")    
    