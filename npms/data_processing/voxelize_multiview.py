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
import cv2
import math
import json
import trimesh
import argparse
import traceback
import pyrender
import glob
from PIL import Image

import multiprocessing as mp
from multiprocessing import Pool

import utils.render_utils as render_utils
import utils.utils as utils
from utils.image_proc import backproject_depth, compute_normals_via_pca    
import data_processing.implicit_waterproofing as iw
from utils.voxels import VoxelGrid
from utils.pcd_utils import normalize
import config as cfg


class SimpleRenderer:
    def __init__(self, mesh, w, h, fx, fy, cx, cy, znear=0.05, zfar=5.0):

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
        # light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=1.0)
        # frontal_light = np.matmul(render_utils.T_opengl_cv, np.eye(4))
        # nl = pyrender.Node(light=light, matrix=frontal_light)
        # self.scene.add_node(nl)

        # Create a simple light setup with 4 directional lights and add it to the scene.
        num_lights = 1000
        for light_id in range(num_lights):
            world_to_light = self.compute_world_to_light(light_id, num_lights, radius)

            light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=1.0)
            nl = pyrender.Node(light=light, matrix=np.linalg.inv(world_to_light))
            self.scene.add_node(nl)

        # Most of the times meshes are given in OpenGL coordinate system.
        # We convert to vision CS first.
        mesh = mesh.apply_transform(render_utils.T_opengl_cv)
        
        # Create pyrender mesh and add it to scene.
        self.mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
        self.scene.add(self.mesh, name="mesh") 

    def compute_world_to_light(self, i, num_lights, radius):
        angle = (1.0 * i / num_lights) * 2 * math.pi
        rotation = render_utils.convert_axis_angle_to_rotation_matrix(np.array([0.0, 1.0, 0.0]), angle)

        unit_vec = np.matmul(rotation, np.array([0, 0, 1]))
        unit_vec /= norm(unit_vec)

        radius_vec = radius * unit_vec
        eye = -radius_vec
        look_dir = radius_vec
        camera_up = np.array([0.0, -1.0, 0.0])

        world_to_light = render_utils.compute_view_matrix(eye, look_dir, camera_up)

        return world_to_light
        
    def compute_camera_to_world(self, i, also_return_w2c=False):
        # Compute the transformation that brings a point in camera space to world space
        # Basically, the camera pose wrt to the world coordinates

        angle = (1.0 * i / self.num_cameras) * 2 * math.pi
        rotation = render_utils.convert_axis_angle_to_rotation_matrix(np.array([0.0, -1.0, 0.0]), angle)

        unit_vec = np.matmul(rotation, np.array([0, 0, 1]))
        unit_vec /= norm(unit_vec)

        radius_vec = radius * unit_vec
        eye = -radius_vec # Our mesh is centered around [0, 0, 0]
        look_dir = radius_vec
        camera_up = np.array([0.0, -1.0, 0.0])

        T_world_to_camera = render_utils.compute_view_matrix(eye, look_dir, camera_up)

        # The rotation around x-axis needs to be applied for OpenGL CS.
        T_world_to_camera = np.matmul(render_utils.T_opengl_cv, T_world_to_camera)
        
        T_camera_to_world = np.linalg.inv(T_world_to_camera)

        if also_return_w2c:
            return T_camera_to_world, T_world_to_camera
        else:
            return T_camera_to_world

    def render_views(self):

        views = {}

        for camera_id in range(self.num_cameras):

            # Compute current camera pose.
            T_camera_to_world, T_world_to_camera = self.compute_camera_to_world(camera_id, True)

            # Adapt the camera pose.
            camera_node = list(self.scene.get_nodes(name="camera"))[0]
            self.scene.set_pose(camera_node, pose=T_camera_to_world)

            ###################################################################################
            # Render color and depth.
            ###################################################################################
            renderer = pyrender.OffscreenRenderer(viewport_width=self.w, viewport_height=self.h)
            color, depth = renderer.render(self.scene)  
            renderer.delete()

            # Mask of valid pixel
            valid_depth_mask_image = depth > 0
            valid_depth_mask = np.reshape(valid_depth_mask_image, -1)

            # Backproject points
            point_image_cam = backproject_depth(
                depth, self.fx, self.fy, self.cx, self.cy, default="NaN", normalizer=1.0
            ) # (3, h, w)
            
            # View vectors (actually, from surface to camera center)
            view_vectors_cam = -point_image_cam

            ###################################################################################
            # Compute normals
            ###################################################################################
            normals_cam = compute_normals_via_pca(point_image_cam, default="NaN", kernel_size=5, max_distance=0.05)
            valid_normals_mask_image = np.all(np.isfinite(normals_cam), axis=-1).astype(np.bool)
            valid_normals_mask = np.reshape(valid_normals_mask_image, -1)

            # Compute final mask, as intersection of both depth and normals mask
            valid_points_mask = valid_depth_mask & valid_normals_mask

            # Reshape
            points_cam       = np.reshape(point_image_cam, (3, -1)) # [3, N]
            view_vectors_cam = np.reshape(view_vectors_cam, (3, -1)) # [3, N]
            normals_cam      = np.reshape(np.moveaxis(normals_cam, -1, 0), (3, -1)) # [3, N]
    
            # Get rid of invalid points
            points_cam       = points_cam[:, valid_points_mask]
            view_vectors_cam = view_vectors_cam[:, valid_points_mask]
            normals_cam      = normals_cam[:, valid_points_mask]

            # Transform to world space
            R_camera_to_world = T_camera_to_world[:3, :3]          # [3, 3]
            t_camera_to_world = T_camera_to_world[:3, -1][:, None] # [3, 1]

            # To vision
            points_world       = np.matmul(render_utils.T_opengl_cv_3x3, points_cam)
            view_vectors_world = np.matmul(render_utils.T_opengl_cv_3x3, view_vectors_cam)
            normals_world      = np.matmul(render_utils.T_opengl_cv_3x3, normals_cam)

            # To world space
            points_world       = np.matmul(R_camera_to_world, points_world) + t_camera_to_world
            view_vectors_world = np.matmul(R_camera_to_world, view_vectors_world) + t_camera_to_world
            normals_world      = np.matmul(R_camera_to_world, normals_world) + t_camera_to_world
            
            # To opengl
            points_world       = np.matmul(render_utils.T_opengl_cv_3x3, points_world)
            view_vectors_world = np.matmul(render_utils.T_opengl_cv_3x3, view_vectors_world)
            normals_world      = np.matmul(render_utils.T_opengl_cv_3x3, normals_world)

            # Swap axis
            points_world       = np.moveaxis(points_world, -1, 0)       # [N, 3]
            view_vectors_world = np.moveaxis(view_vectors_world, -1, 0) # [N, 3]
            normals_world      = np.moveaxis(normals_world, -1, 0)      # [N, 3]

            num_points = points_world.shape[0]
            assert points_world.shape == (num_points, 3) 
            assert view_vectors_world.shape == (num_points, 3) 
            assert normals_world.shape == (num_points, 3) 

            ###################################################################################
            # DEBUG
            ###################################################################################
            # pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_world))
            # pcd.paint_uniform_color([0, 1, 0]) # blue are source

            # along = points_world + 1 * view_vectors_world

            # along_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(along))
            # along_pcd.paint_uniform_color([0, 0, 1]) # blue are source

            # correspondences = [(i, i) for i in range(0, points_world.shape[0])]
            # lines = o3d.geometry.LineSet.create_from_point_cloud_correspondences(
            #     pcd, along_pcd, correspondences
            # )
            # lines.paint_uniform_color([0.5, 0.8, 0])

            # o3d.visualization.draw_geometries([pcd, along_pcd, lines])
            ###################################################################################

            # We'll also save T_world_to_camera
            R_world_to_camera = T_world_to_camera[:3, :3]          # [3, 3]
            t_world_to_camera = T_world_to_camera[:3, -1][:, None] # [3, 1]

            # Store
            views[camera_id] = {
                'points_world': points_world,
                'view_vectors_world': view_vectors_world,
                'normals_world': normals_world,
                'color': color,
                'point_image': point_image_cam,
                'R_world_to_camera': R_world_to_camera,
                't_world_to_camera': t_world_to_camera,
            }

        return views


def render_views(in_path):
    try:
        
        if "SPLITS" in in_path or in_path.endswith("json") or in_path.endswith("txt")or in_path.endswith("npz"):
            return

        # Dir where we'll be storing the outpout
        partial_views_dir = os.path.join(in_path, "partial_views")

        if not os.path.isdir(partial_views_dir):
            print(f"Creating dir for {in_path}")
            os.makedirs(partial_views_dir)

        # Path to input mesh
        mesh_path = os.path.join(in_path, MESH_FILENAME)
        
        if not os.path.isfile(mesh_path):
            print("--------------------------- Skipping", mesh_path)
            return

        print("Processing", mesh_path)
        
        # Read mesh
        mesh = trimesh.load(
            mesh_path, 
            process=False
        )

        # Prepare scene with mesh
        scene = SimpleRenderer(mesh, w=w, h=h, fx=fx, fy=fy, cx=cx, cy=cy, znear=znear, zfar=zfar) 
        
        # Render 4 views
        views = scene.render_views()

        for view_id, geom_dict in views.items():

            points_world       = geom_dict['points_world']
            view_vectors_world = geom_dict['view_vectors_world']
            normals_world      = geom_dict['normals_world']
            color              = geom_dict['color']
            point_image        = geom_dict['point_image']
            R_world_to_camera  = geom_dict['R_world_to_camera']
            t_world_to_camera  = geom_dict['t_world_to_camera']

            out_file = os.path.join(
                in_path, 
                partial_views_dir, 
                f"voxelized_view{view_id}_{args.res}res.npz"
            )

            if not OVERWRITE and os.path.exists(out_file):
                print(f'File {out_file} exists. Done.')
                return

            #######################
            # Save color image
            #######################
            if save_color_image:    
                out_color_image_path = os.path.join(in_path, partial_views_dir, f"color_view{view_id}.png")
                if OVERWRITE and os.path.isfile(out_color_image_path):
                    os.remove(out_color_image_path)
                img = Image.fromarray(color, 'RGB')
                img.save(out_color_image_path)

            #######################
            # Save point image
            #######################
            if save_point_image:
                out_point_image_path = os.path.join(in_path, partial_views_dir, f"point_image_view{view_id}.npz")
                np.savez(
                    out_point_image_path, 
                    point_image=point_image, 
                )

            # ############################ DEGUB ############################
            # from PIL import Image
            # Image.fromarray(mask_image).save(f"mask{view_id}.png")
            # trimesh.Trimesh(vertices=points_world, faces=[]).export(f"campoints{view_id}.ply")
            # ############################ DEGUB ############################

            #######################
            # Save occupancies
            #######################
            occupancies = np.zeros(len(grid_points), dtype=np.int8)

            _, idx = kdtree.query(points_world)
            occupancies[idx] = 1

            compressed_occupancies = np.packbits(occupancies)

            np.savez(
                out_file, 
                point_cloud=points_world, 
                # view_vectors=view_vectors_world,
                # normals=normals_world,
                compressed_occupancies=compressed_occupancies,
                bb_min=bb_min, bb_max=bb_max, 
                res=args.res
            )

            #######################
            # Transformation from world to camera
            #######################
            if save_world_to_camera:
                out_w2c_file = os.path.join(
                    in_path, 
                    partial_views_dir, 
                    f"world_to_camera_view{view_id}.npz"
                )
                np.savez(
                    out_w2c_file,
                    R_world_to_camera=R_world_to_camera,
                    t_world_to_camera=t_world_to_camera,
                )

            # ############################ DEGUB ############################
            # voxels = np.reshape(occupancies, (args.res,) * 3)
            # loc = ( (bb_min + bb_max) / 2, ) * 3
            # scale = bb_max - bb_min
            # VoxelGrid(voxels, loc, scale).to_mesh().export(f"VOXEL{view_id}.ply")
            # ############################ DEGUB ############################

        print('Finished {}'.format(in_path))

    except Exception as err:
        print('Error with {}: {}'.format(in_path, traceback.format_exc()))


if __name__ == "__main__":

    OVERWRITE = True

    parser = argparse.ArgumentParser(
        description='Run point cloud sampling'
    )

    parser.add_argument('-res', type=int, required=True)
    parser.add_argument('-t', '-max_threads', dest='max_threads', type=int, default=-1)

    args = parser.parse_args()

    num_cameras = 1
    radius = 1.4

    bb_min = -0.5
    bb_max = 0.5

    # VoxelGrid on which to sample our voxels from the depth maps
    grid_points = iw.create_grid_points_from_bounds(bb_min, bb_max, args.res)
    kdtree = KDTree(grid_points)

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

    # General params for the camera
    fx = 573.353
    fy = 576.057
    cx = 319.85
    cy = 240.632
    w = 640
    h = 480
    znear=0.05
    zfar=5.0

    ########################################################################

    ROOT = f'/cluster/lothlann/ppalafox/datasets'

    # -----------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------

    dataset_name = "CAPE-POSE-TRAIN-35id-subsampled-10119ts"

    # Mesh type used to generate the data
    MESH_FILENAME = 'mesh_normalized.ply' # Select between 'mesh_normalized.ply' and 'mesh_real_scan.ply' (for CAPE)
    
    assert MESH_FILENAME == 'mesh_normalized.ply' or MESH_FILENAME == 'mesh_real_scan.ply'
    
    print()
    print("----------------------------")
    print(f"Using {MESH_FILENAME}!")
    print("----------------------------")
    print()

    if "cape" in dataset_name.lower() and "TEST" in dataset_name and MESH_FILENAME != 'mesh_real_scan.ply':
        print("Use real scans for CAPE!")
        exit()

    # Whether to save some additional data for the baselines
    save_color_image = False
    save_point_image = False
    save_world_to_camera = False

    if "TEST" in dataset_name:
        save_color_image = True
        save_point_image = True
        save_world_to_camera = True

    print()
    print("save_color_image     ", save_color_image)
    print("save_point_image     ", save_color_image)
    print("save_world_to_camera ", save_color_image)
    print()

    # -----------------------------------------------------------------------------------------------
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
    
    # -----------------------------------------------------------------------------------------------
    
    path_to_samples = []
    for label in labels:
        path_to_samples.append(
            os.path.join(ROOT, label['dataset'], label['identity_name'], label['animation_name'], label['sample_id'])
        )

    print("Number of samples to process", len(labels))
    input("Continue?")

    ########################################################################

    try:
        p = Pool(n_jobs)
        p.map(render_views, path_to_samples)
    finally: # To make sure processes are closed in the end, even if errors happen
        p.close()
        p.join()