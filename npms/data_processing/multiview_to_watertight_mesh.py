import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import open3d as o3d

os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
# os.environ['PYOPENGL_PLATFORM'] = 'egl'
# os.environ['EGL_DEVICE_ID'] = os.environ['SLURM_STEP_GPUS']

import numpy as np
import time
from tqdm import tqdm
import math
import json
import trimesh
import argparse
import traceback
import pyrender
import glob
import subprocess as sp
import shutil
from scipy.linalg import expm, norm

import multiprocessing as mp
from multiprocessing import Pool

import utils.utils as utils
from NPMs._C import compute_mesh_from_depth as compute_mesh_from_depth_c
from utils.image_proc import backproject_depth    
from data_scripts import config_data as cfg

import utils.render_utils as render_utils

from utils.pcd_utils import (BBox,
                            transform_pointcloud_to_opengl_coords,
                            rotate_around_axis,
                            origin, normalize_transformation)


T_opengl_cv = np.array(
    [[1.0,  0.0,  0.0, 0.0],
    [0.0, -1.0,  0.0, 0.0],
    [0.0,  0.0, -1.0, 0.0],
    [0.0,  0.0,  0.0, 1.0]]
)

T_opengl_cv_3x3 = np.array(
    [[1.0,  0.0,  0.0],
    [0.0, -1.0,  0.0 ],
    [0.0,  0.0, -1.0 ]]
)


def merge_meshes(vertices_array, faces_array, total_num_vertices, total_num_faces):
    merged_vertices = np.zeros((total_num_vertices, 3), dtype=np.float32)
    merged_faces = np.zeros((total_num_faces, 3), dtype=np.int32)

    offset_vertices = 0
    offset_faces = 0

    for i in range(len(vertices_array)):
        vertices = vertices_array[i]
        faces = faces_array[i]

        merged_vertices[offset_vertices:offset_vertices+vertices.shape[0]] = vertices
        merged_faces[offset_faces:offset_faces+faces.shape[0]] = faces + offset_vertices

        offset_vertices += vertices.shape[0]
        offset_faces += faces.shape[0]
    
    return merged_vertices, merged_faces


def transform_mesh(vertices, R, t):
    num_vertices = vertices.shape[0]
    vertices = np.matmul(
        np.repeat(R.reshape(1, 3, 3), num_vertices, axis=0),
        vertices.reshape(-1, 3, 1)
    ) + np.repeat(t.reshape(1, 3, 1), num_vertices, axis=0)

    vertices = vertices.reshape(-1, 3)
    return vertices


class SimpleRenderer:
    def __init__(self, mesh, w, h, fx, fy, cx, cy, znear=0.05, zfar=5.0):

        self.w = w
        self.h = h
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

        self.scene = pyrender.Scene(ambient_light=np.array([1., 1., 1., 1.]))

        # Initialize camera setup.
        self.camera_setup = render_utils.CameraSetup(camera_setup, None, radius, num_subdivisions_ico)

        self.camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, znear=znear, zfar=zfar) 
        self.scene.add(self.camera, pose=np.eye(4), name="camera")

        # Most of the times meshes are given in OpenGL coordinate system.
        # We convert to vision CS first.
        mesh = mesh.apply_transform(T_opengl_cv)
        
        # Create pyrender mesh and add it to scene.
        self.mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
        self.scene.add(self.mesh, name="mesh") 

        # Compute initial camera pose.
        mesh_min = mesh.bounds[0]
        mesh_max = mesh.bounds[1]
        self.mesh_center = (mesh_min + mesh_max) / 2.0
        self.mesh_extent = mesh_max - mesh_min

        # Create a simple light setup with 4 directional lights and add it to the scene.
        num_lights = 4
        for light_id in range(num_lights):
            world_to_light = self.compute_world_to_light(light_id, num_lights, radius)

            light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=0.7)
            nl = pyrender.Node(light=light, matrix=np.linalg.inv(world_to_light))
            self.scene.add_node(nl)

    def compute_world_to_camera(self, i):
        radius_vec = self.camera_setup.get_camera_unit_vector(i)

        eye = self.mesh_center - radius_vec
        look_dir = self.mesh_center - eye
        camera_up = self.camera_setup.get_camera_up(i)

        T_world_to_camera = render_utils.compute_view_matrix(eye, look_dir, camera_up)

        # The rotation around x-axis needs to be applied for OpenGL CS.
        T_world_to_camera = np.matmul(T_opengl_cv, T_world_to_camera)

        return T_world_to_camera

    def compute_world_to_light(self, i, num_lights, radius):
        angle = (1.0 * i / num_lights) * 2 * math.pi
        rotation = render_utils.convert_axis_angle_to_rotation_matrix(np.array([0.0, 1.0, 0.0]), angle)

        unit_vec = np.matmul(rotation, np.array([0, 0, 1]))
        unit_vec /= norm(unit_vec)

        radius_vec = radius * unit_vec
        eye = self.mesh_center - radius_vec
        look_dir = self.mesh_center - eye
        camera_up = np.array([0.0, -1.0, 0.0])

        world_to_light = render_utils.compute_view_matrix(eye, look_dir, camera_up)

        return world_to_light

    def compute_merged_mesh(self):

        vertices_array = []
        faces_array = []

        total_num_vertices = 0
        total_num_faces = 0

        for camera_id in range(self.camera_setup.get_num_cameras()):

            # print(f"Rendering camera {camera_id} / {self.camera_setup.get_num_cameras()}")

            # Compute current camera pose.
            T_world_to_camera = self.compute_world_to_camera(camera_id)
            # T_camera_to_world = self.compute_camera_to_world(camera_id)

            T_camera_to_world = np.linalg.inv(T_world_to_camera)
            R_camera_to_world = T_camera_to_world[:3, :3]
            t_camera_to_world = T_camera_to_world[:3, 3]

            # Adapt the camera pose.
            camera_node = list(self.scene.get_nodes(name="camera"))[0]
            self.scene.set_pose(camera_node, pose=T_camera_to_world)

            ###################################################################################
            # Render color and depth.
            ###################################################################################
            renderer = pyrender.OffscreenRenderer(viewport_width=self.w, viewport_height=self.h)
            _, depth = renderer.render(self.scene)  
            renderer.delete()

            # Backproject points
            point_image_cam = backproject_depth(
                depth, self.fx, self.fy, self.cx, self.cy, default="NaN", normalizer=1.0
            ) # (3, h, w)

            # Compute mesh from point image.
            vertices = np.zeros((1), dtype=np.float32)
            faces = np.zeros((1), dtype=np.int32)

            compute_mesh_from_depth_c(point_image_cam, max_triangle_dist, vertices, faces)

            # Apply extrinsics to the mesh.
            vertices = transform_mesh(vertices, T_opengl_cv_3x3, np.zeros_like(t_camera_to_world))
            vertices = transform_mesh(vertices, R_camera_to_world, t_camera_to_world)

            # Store vertices and faces.
            vertices_array.append(vertices)
            faces_array.append(faces)

            total_num_vertices += vertices.shape[0]
            total_num_faces += faces.shape[0]

        # Merge meshes into a single mesh.
        vertices, faces = merge_meshes(vertices_array, faces_array, total_num_vertices, total_num_faces)

        # Visualize mesh.
        mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(faces))
        mesh.compute_vertex_normals()

        # Transform to OpenGL coordinates
        mesh = rotate_around_axis(mesh, axis_name="x", angle=-np.pi) 

        ###################################################################################
        # DEBUG
        ###################################################################################
        if False:
            o3d.visualization.draw_geometries([mesh])
        ###################################################################################

        return mesh


def render_views(in_path):
    try:
        
        if "SPLITS" in in_path or in_path.endswith("json") or in_path.endswith("txt")or in_path.endswith("npz"):
            return

        if "a_t_pose" not in in_path:
            return

        print()
        print()
        print()
        print("Processing", in_path)
        print()

        out_mesh_path = os.path.join(in_path, "mesh_watertight_poisson.ply")

        if not OVERWRITE and os.path.exists(out_mesh_path):
            print('Skipping {}'.format(out_mesh_path))
            return

        # Read mesh
        mesh = trimesh.load(
            os.path.join(in_path, 'mesh_normalized.off'), 
            process=False
        )

        # Prepare scene with mesh
        scene = SimpleRenderer(mesh, w=w, h=h, fx=fx, fy=fy, cx=cx, cy=cy, znear=znear, zfar=zfar) 
        
        # Render views and compute merged mesh
        merged_mesh = scene.compute_merged_mesh()

        # Store merged mesh
        merged_mesh_path = os.path.join(in_path, "merged_mesh.ply")
        o3d.io.write_triangle_mesh(merged_mesh_path, merged_mesh)
        
        # Run Screened Poisson
        sp.check_output(
            f'meshlabserver -i {merged_mesh_path} -o {out_mesh_path} -s {poisson_exec}',
            shell=True
        )

        os.remove(merged_mesh_path)

        print('Finished {}'.format(in_path))

    except Exception as err:
        print('Error with {}: {}'.format(in_path, traceback.format_exc()))


if __name__ == "__main__":

    OVERWRITE = False

    parser = argparse.ArgumentParser(
        description='Run point cloud sampling'
    )

    parser.add_argument('-t', '-max_threads', dest='max_threads', type=int, default=-1)

    args = parser.parse_args()

    workspace_dir = "/home/pablo/workspace/neshpod/"
    poisson_exec = os.path.join(workspace_dir, "data_processing/screened_poisson.mlx")

    ######################################################################################################
    # Options
    camera_setup = "icosahedron" # "icosahedron" or "octo" 
    num_subdivisions_ico = 1 # 0: 12 cameras | 1: 62 cameras | 2: 242 cameras
    radius = 0.8

    max_triangle_dist = 0.005

    bb_min = -0.5
    bb_max = 0.5

    # General params for the camera
    fx = 573.353
    fy = 576.057
    cx = 319.85
    cy = 240.632
    w = 640
    h = 480
    znear=0.05
    zfar=5.0
    ######################################################################################################

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

    # character_names = [
    #     'ninja',
    #     # 'olivia',
    #     # 'racer',
    #     # 'crypto',
    #     # 'alien'
    # ]
    character_names = cfg.identities + cfg.identities_augmented

    dataset_type = "datasets_multi"

    for character_name in sorted(character_names):

        ROOT = f'/cluster/lothlann/ppalafox/{dataset_type}/mixamo/{character_name}'
        render_views(os.path.join(ROOT, "a_t_pose", "000000"))