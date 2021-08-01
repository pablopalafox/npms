import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import open3d as o3d

os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
# os.environ['PYOPENGL_PLATFORM'] = 'egl'
# os.environ['EGL_DEVICE_ID'] = os.environ['SLURM_STEP_GPUS']

import numpy as np
from scipy.linalg import expm, norm
from scipy.spatial import cKDTree as KDTree
import time
from tqdm import tqdm
import math
import json
import trimesh
import argparse
import traceback
import pyrender
import glob

import multiprocessing as mp
from multiprocessing import Pool

import utils.render_utils as render_utils
import utils.utils as utils
from utils.image_proc import (
    backproject_depth, 
    compute_validity_and_sign_mask
)
import data_processing.implicit_waterproofing as iw
from utils.voxels import VoxelGrid
from utils.pcd_utils import normalize
import config as cfg
import utils.gaps_utils as gaps_utils


class PartialSDFGenerator:
    def __init__(self, num_cameras, mesh, w, h, fx, fy, cx, cy, znear=0.05, zfar=5.0):

        self.num_cameras = num_cameras

        self.w = w
        self.h = h
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

        self.scene = pyrender.Scene(ambient_light=np.array([1., 1., 1., 1.]))

        self.camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, znear=znear, zfar=zfar) 
        self.scene.add(self.camera, pose=np.eye(4), name="camera")

        # Create a light and add it to the scene.
        light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=1.0)
        nl = pyrender.Node(light=light, matrix=np.eye(4))
        self.scene.add_node(nl)

        # Most of the times meshes are given in OpenGL coordinate system.
        # We convert to vision CS first.
        mesh = mesh.apply_transform(render_utils.T_opengl_cv)
        
        # Create pyrender mesh and add it to scene.
        self.mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
        self.scene.add(self.mesh, name="mesh") 

    def compute_camera_to_world(self, i):
        # Compute the transformation that brings a point in camera space to world space
        # Basically, the camera pose wrt to the world coordinates

        angle = (1.0 * i / self.num_cameras) * 2 * math.pi
        rotation = render_utils.convert_axis_angle_to_rotation_matrix(np.array([0.0, -1.0, 0.0]), angle)

        unit_vec = np.matmul(rotation, np.array([0, 0, 1]))
        unit_vec /= norm(unit_vec)

        radius_vec = 1.4 * unit_vec
        eye = -radius_vec # Our mesh is centered around [0, 0, 0]
        look_dir = radius_vec
        camera_up = np.array([0.0, -1.0, 0.0])

        T_world_to_camera = render_utils.compute_view_matrix(eye, look_dir, camera_up)

        # The rotation around x-axis needs to be applied for OpenGL CS.
        T_world_to_camera = np.matmul(render_utils.T_opengl_cv, T_world_to_camera)
        
        T_camera_to_world = np.linalg.inv(T_world_to_camera)

        return T_camera_to_world

    def compute_valid_grid_mask(self):

        views = {}

        for camera_id in range(self.num_cameras):

            # Compute current camera pose.
            T_camera_to_world = self.compute_camera_to_world(camera_id)

            # Adapt the camera pose.
            camera_node = list(self.scene.get_nodes(name="camera"))[0]
            self.scene.set_pose(camera_node, pose=T_camera_to_world)

            ###################################################################################
            # Render color and depth.
            ###################################################################################
            renderer = pyrender.OffscreenRenderer(viewport_width=self.w, viewport_height=self.h)
            _, depth_image = renderer.render(self.scene)  
            renderer.delete()

            # Mask of valid pixel
            valid_depth_mask_image = depth_image > 0
            valid_depth_mask = np.reshape(valid_depth_mask_image, -1)

            # Backproject points
            points_cam = backproject_depth(
                depth_image, self.fx, self.fy, self.cx, self.cy, default="NaN", normalizer=1.0
            ) # (3, h, w)

            ###################################################################################
            # Compute points in the world
            ###################################################################################
            valid_points_mask = valid_depth_mask

            # Reshape
            points_cam = np.reshape(points_cam, (3, -1)) # [3, N]
            # Get rid of invalid points
            points_cam = points_cam[:, valid_points_mask]
            # Transform to world space
            R_camera_to_world = T_camera_to_world[:3, :3]          # [3, 3]
            t_camera_to_world = T_camera_to_world[:3, -1][:, None] # [3, 1]

            # To vision
            points_world = np.matmul(render_utils.T_opengl_cv_3x3, points_cam)
            # To world space
            points_world = np.matmul(R_camera_to_world, points_world) + t_camera_to_world
            # To opengl
            points_world = np.matmul(render_utils.T_opengl_cv_3x3, points_world)
            # Swap axis
            points_world = np.moveaxis(points_world, -1, 0)       # [N, 3]
            
            num_points = points_world.shape[0]
            assert points_world.shape == (num_points, 3) 
            
            ###################################################################################
            # Rotate grid to current view and project 
            ###################################################################################

            # Transform grid points to camera space
            T_world_to_camera = np.linalg.inv(T_camera_to_world)
            R_world_to_camera = T_world_to_camera[:3, :3]          # [3, 3]
            t_world_to_camera = T_world_to_camera[:3, -1][:, None] # [3, 1]

            grid_points_tmp = np.moveaxis(grid_points, 0, -1)
            grid_points_cam = np.matmul(render_utils.T_opengl_cv_3x3, grid_points_tmp)
            grid_points_cam = np.matmul(R_world_to_camera, grid_points_cam) + t_world_to_camera
            grid_points_cam = np.matmul(render_utils.T_opengl_cv_3x3, grid_points_cam)
            grid_points_cam = np.moveaxis(grid_points_cam, 0, -1)

            ################################################################
            # pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(grid_points_cam))
            # pcd.paint_uniform_color([0, 1, 0]) # blue are source
            # world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            #         size=0.1, origin=[0, 0, 0]
            # )
            # o3d.visualization.draw_geometries([pcd, world_frame])
            ################################################################

            start = time.perf_counter()
            validity_grid_mask, sign_grid_mask = compute_validity_and_sign_mask(
                grid_points_cam, 
                valid_depth_mask_image,
                depth_image,
                self.fx, self.fy, 
                self.cx, self.cy, 
                self.w, self.h,
                truncation
            )
            print("validity and sign", time.perf_counter() - start)

            views[camera_id] = {
                'points_world': points_world,
                'validity_grid_mask': validity_grid_mask,
                'sign_grid_mask': sign_grid_mask,
            }

        return views


def render_views(in_path):
    try:
        
        if "SPLITS" in in_path or in_path.endswith("json") or in_path.endswith("txt")or in_path.endswith("npz"):
            return

        mesh_path = os.path.join(in_path, MESH_FILENAME)

        if not os.path.isfile(mesh_path):
            print("--------------------------------- Skipping", mesh_path)
            return

        print('Processing', in_path)
        
        # Read mesh
        mesh = trimesh.load(
            mesh_path, 
            process=False
        )

        # Prepare scene with mesh
        scene = PartialSDFGenerator(num_cameras, mesh, w=w, h=h, fx=fx, fy=fy, cx=cx, cy=cy, znear=znear, zfar=zfar) 
        
        # Render 4 views
        views = scene.compute_valid_grid_mask()

        for view_id, geom_dict in views.items():

            points_world       = geom_dict['points_world']
            validity_grid_mask = geom_dict['validity_grid_mask']
            sign_grid_mask     = geom_dict['sign_grid_mask']

            out_dir = os.path.join(in_path, "partial_sdf_grd")
            if not os.path.isdir(out_dir):
                os.makedirs(out_dir)

            out_file = os.path.join(
                out_dir, 
                f"partial_sdf_grd_{args.res}.npz"
            )

            # Remove old files if they exist
            for old_file in glob.glob(in_path + '/partial_sdf_grd_*'):
                os.remove(old_file)

            if not OVERWRITE and os.path.exists(out_file):
                print('File exists. Done.')
                return

            # Query
            start = time.perf_counter()
            kdtree = KDTree(points_world)
            sdf_grid, _ = kdtree.query(grid_points)
            sdf_grid = sdf_grid.astype(np.float32)
            print("Querying done!", start - time.perf_counter())

            # Apply the sign
            sdf_grid = sdf_grid * sign_grid_mask

            # Clamp too big distances
            if False:
                sdf_grid[sdf_grid > max_distance] = max_distance

            # Set invalid voxels to nan 
            # sdf_grid[~validity_grid_mask] = -np.inf
            
            # Save            
            np.savez_compressed(
                out_file, 
                points_world=points_world, 
                sdf_grid=sdf_grid, 
                validity_mask=validity_grid_mask
            )

            ################################################################
            # Visualize
            ################################################################
            if False:
                print("Visualizing...")

                _, sdf_grid_read = gaps_utils.read_grd(out_file)
                sdf_grid = np.reshape(sdf_grid_read, (-1,))

                invalid_mask_grid_tmp = ~validity_grid_mask
                invalid_mask_grid_tmp = invalid_mask_grid_tmp.astype(np.int8)
                voxels = np.reshape(invalid_mask_grid_tmp, (args.res,) * 3)
                voxels_trimesh = VoxelGrid(voxels).to_mesh()
                invalid_voxels_mesh = o3d.geometry.TriangleMesh(
                    o3d.utility.Vector3dVector(voxels_trimesh.vertices),
                    o3d.utility.Vector3iVector(voxels_trimesh.faces)
                )
                invalid_voxels_mesh.compute_vertex_normals()
                invalid_voxels_mesh.paint_uniform_color([0, 1, 0])

                # Grid points
                grid_pcd_in = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(grid_points[sdf_grid < 0]))
                grid_pcd_in.paint_uniform_color([0, 0, 1])

                grid_pcd_out = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(grid_points[sdf_grid > 0]))
                grid_pcd_out.paint_uniform_color([1, 0, 0])

                # Grid points
                grid_pcd_far = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(grid_points[sdf_grid > max_distance]))
                grid_pcd_far.paint_uniform_color([1, 1, 0])

                grid_pcd_near_valid = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(grid_points[(sdf_grid < max_distance) & validity_grid_mask]))
                grid_pcd_near_valid.paint_uniform_color([0, 1, 1])

                grid_pcd_invalid = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(grid_points[invalid_mask_grid_tmp.astype(np.bool)]))
                grid_pcd_invalid.paint_uniform_color([1, 0, 1])

                # Points
                pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_world))
                pcd.paint_uniform_color([0, 1, 1]) # blue are source
                
                o3d.visualization.draw_geometries([pcd])
                o3d.visualization.draw_geometries([invalid_voxels_mesh, grid_pcd_in, grid_pcd_out, pcd])
                o3d.visualization.draw_geometries([invalid_voxels_mesh])
                o3d.visualization.draw_geometries([grid_pcd_in, grid_pcd_out])
                o3d.visualization.draw_geometries([grid_pcd_in])
                o3d.visualization.draw_geometries([grid_pcd_far])
                o3d.visualization.draw_geometries([grid_pcd_invalid])
                o3d.visualization.draw_geometries([grid_pcd_far, grid_pcd_invalid])
                o3d.visualization.draw_geometries([grid_pcd_near_valid])

                return
            ################################################################

        print('Finished {}'.format(in_path))

    except Exception as err:
        print('Error with {}: {}'.format(in_path, traceback.format_exc()))


if __name__ == "__main__":

    OVERWRITE = False

    parser = argparse.ArgumentParser(
        description='Run point cloud sampling'
    )

    parser.add_argument('-res', type=int, required=True)
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

    ##########################################################################################

    bb_min = -0.5
    bb_max = 0.5
    
    # TRUNCATION
    max_distance = 0.25
    truncation = 0.01

    # VoxelGrid on which to sample our voxels from the depth maps
    grid_points = iw.create_grid_points_from_bounds(bb_min, bb_max, args.res)

    num_cameras = 1

    # General params for the camera
    fx = 573.353
    fy = 576.057
    cx = 319.85
    cy = 240.632
    w = 640
    h = 480
    znear=0.05
    zfar=5.0

    ROOT = f'/cluster/lothlann/ppalafox/datasets'
    
    # -----------------------------------------------------------------------------------------------
    ####################################
    # MIXAMO
    ####################################
    dataset_name = "CAPE-POSE-TRAIN-35id-subsampled-10119ts"

    # Mesh type used to generate the data
    MESH_FILENAME = 'mesh_normalized.ply' # Select between 'mesh_normalized.ply' and 'mesh_real_scan.ply' (for CAPE)
    
    assert MESH_FILENAME == 'mesh_normalized.ply' or MESH_FILENAME == 'mesh_real_scan.ply'
    
    print()
    print(f"Using {MESH_FILENAME}!")
    print()

    if "cape" in dataset_name.lower() and MESH_FILENAME != 'mesh_real_scan.ply':
        print("Use real scans for CAPE!")
        exit()

    # -----------------------------------------------------------------------------------------------

    from utils.parsing_utils import get_dataset_type_from_dataset_name
    dataset_type = get_dataset_type_from_dataset_name(dataset_name)
    splits_dir = f"{cfg.splits_dir}_{dataset_type}"

    labels_json       = os.path.join(ROOT, splits_dir, dataset_name, "labels.json")
    labels_tpose_json = os.path.join(ROOT, splits_dir, dataset_name, "labels_tpose.json")
    
    with open(labels_json, "r") as f:
        labels = json.loads(f.read())

    with open(labels_tpose_json, "r") as f:
        labels_tpose = json.loads(f.read())

    labels = labels + labels_tpose

    print("Dataset name:", dataset_name)
    print("Number of samples to process", len(labels))
    input("Continue?")

    paths_to_process = []
    for label in labels:
        paths_to_process.append(
            os.path.join(ROOT, label['dataset'], label['identity_name'], label['animation_name'], label['sample_id'])
        )

    try:
        p = Pool(n_jobs)
        p.map(render_views, paths_to_process)
    finally:
        p.close()
        p.join()