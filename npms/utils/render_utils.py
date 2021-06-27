import sys
import os
os.environ['PYOPENGL_PLATFORM'] = 'osmesa' # 'osmesa', 'egl'

import numpy as np
import trimesh
import math
import matplotlib.pyplot as plt
from skimage import io
import copy
from scipy.linalg import expm, norm
from utils import mesh_proc
import cv2

import pyrender


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


def convert_axis_angle_to_rotation_matrix(axis, theta):
    return expm(np.cross(np.eye(3), axis/norm(axis)*theta))


def convert_euler_angles_to_rotation_matrix(yaw, pitch, roll):
    # Conversion from vision to robotics CS.
    angle_z, angle_y, angle_x = pitch, -yaw, roll

    return np.array([
        [math.cos(angle_z) * math.cos(angle_y), 
        math.cos(angle_z) * math.sin(angle_y) * math.sin(angle_x) - math.sin(angle_z) * math.cos(angle_x),
        math.cos(angle_z) * math.sin(angle_y) * math.cos(angle_x) + math.sin(angle_z) * math.sin(angle_x)],
        [math.sin(angle_z) * math.cos(angle_y), 
        math.sin(angle_z) * math.sin(angle_y) * math.sin(angle_x) + math.cos(angle_z) * math.cos(angle_x),
        math.sin(angle_z) * math.sin(angle_y) * math.cos(angle_x) - math.cos(angle_z) * math.sin(angle_x)],
        [-math.sin(angle_y),
        math.cos(angle_y) * math.sin(angle_x),
        math.cos(angle_y) * math.cos(angle_x)]
    ])


def compute_view_matrix(eye, look_dir, camera_up):
    # Normalize viewing vectors.
    look_dir = look_dir / np.linalg.norm(look_dir)
    camera_up = camera_up / np.linalg.norm(camera_up)

    look_right = -np.cross(camera_up, look_dir) 
    look_right = look_right / np.linalg.norm(look_right)

    look_up = -np.cross(look_dir, look_right) 
    look_up = look_up / np.linalg.norm(look_up)

    # Compute camera pose.
    # The rows of the rotation matrix are the vectors that
    # we get by rotating unit vectors e0, e1, e2.
    cam_pose = np.eye(4)
    cam_pose[0, 0] = look_right[0]
    cam_pose[0, 1] = look_right[1]
    cam_pose[0, 2] = look_right[2]
    cam_pose[1, 0] = -look_up[0]
    cam_pose[1, 1] = -look_up[1]
    cam_pose[1, 2] = -look_up[2]
    cam_pose[2, 0] = look_dir[0]
    cam_pose[2, 1] = look_dir[1]
    cam_pose[2, 2] = look_dir[2]

    cam_pose[:3, 3] = -np.matmul(cam_pose[:3, :3], eye)

    return cam_pose


def visualize_textured_obj(mesh_path, w, h, fx, fy, cx, cy, znear=0.05, zfar=5.0, radius=1.0):
    # Load mesh.
    mesh = trimesh.load(mesh_path)

    # Most of the times meshes are given in OpenGL coordinate system.
    # We convert to vision CS first.
    T_opengl_cv = np.array([[1.0,  0.0,  0.0, 0.0],
                            [0.0, -1.0,  0.0, 0.0],
                            [0.0,  0.0, -1.0, 0.0],
                            [0.0,  0.0,  0.0, 1.0]])

    mesh.apply_transform(T_opengl_cv)

    # Create scene.
    scene = pyrender.Scene(ambient_light=np.array([1., 1., 1., 1.0]))
    
    # Create pyrender mesh and add it to scene.
    mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    scene.add(mesh, name="mesh")    

    # Prepare intrinsic camera and add it to scene.
    camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, znear=znear, zfar=zfar) 
    scene.add(camera, pose=np.eye(4))

    # Create a light and add it to the scene.
    light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=1.0)
    nl = pyrender.Node(light=light, matrix=np.eye(4))
    scene.add_node(nl)

    # Compute world-to-camera pose.
    mesh_min = mesh.bounds[0]
    mesh_max = mesh.bounds[1]
    mesh_center = (mesh_min + mesh_max) / 2.0
    mesh_extent = mesh_max - mesh_min  

    unit_vec = np.array([0, 0, 1])

    radius_vec = radius * unit_vec
    eye = mesh_center - radius_vec
    look_dir = mesh_center - eye
    camera_up = np.array([0.0, -1.0, 0.0])

    world_to_camera = compute_view_matrix(eye, look_dir, camera_up)

    # Apply motion to the mesh.
    # The rotation around x-axis needs to be apply for OpenGL CS.
    mesh_pose = np.matmul(T_opengl_cv, world_to_camera)

    mesh_node = list(scene.get_nodes(name="mesh"))[0]
    scene.set_pose(mesh_node, pose=mesh_pose)

    renderer = pyrender.OffscreenRenderer(viewport_width=w, viewport_height=h)
    color, depth = renderer.render(scene)  
    renderer.delete()
    
    # Visualize the rendered image.
    plt.imshow(color)
    plt.show(block=True)
    plt.pause(0.1)


def convert_to_mesh_array(scene_or_mesh):
    """
    Convert a possible scene to array of meshes.
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh_array = []  # empty scene
        else:
            # we lose texture information here
            mesh_array = [g for g in scene_or_mesh.geometry.values()] 
    else:
        assert(isinstance(scene_or_mesh, trimesh.Trimesh))
        mesh_array = [scene_or_mesh]
    return mesh_array


class CameraSetup:
    def __init__(self, setup_type="octo", vertical_disp=0.5, radius=0.3, num_subdivisions_ico=0):
        self.radius = radius
        self.non_planar_cameras = True
        self.num_subdivisions_ico = num_subdivisions_ico

        if setup_type == "octo":
            self.num_cameras = 8
            self.camera_deltas = np.zeros((8, 3), dtype=np.float32)

            for i in range(self.num_cameras):
                angle = (1.0 * i / self.num_cameras) * 2 * math.pi
                rotation = convert_axis_angle_to_rotation_matrix(np.array([0.0, 1.0, 0.0]), angle)

                unit_vec = np.array([0.0, 0.0, 1.0])
                if self.non_planar_cameras:
                    unit_vec[1] = vertical_disp if i % 2 == 0 else -vertical_disp

                unit_vec = np.matmul(rotation, unit_vec)
                unit_vec /= norm(unit_vec)

                self.camera_deltas[i] = unit_vec * self.radius

        elif setup_type == "icosahedron":
            # Generate icosahedron.
            vertices, triangles = mesh_proc.generate_icosahedron()

            # Subdivide mesh.
            for _ in range(self.num_subdivisions_ico):
                vertices, triangles = mesh_proc.subdivide_mesh(vertices, triangles)
            
            # Scale camera positions.
            self.num_cameras = vertices.shape[0]
            self.camera_deltas = self.radius * vertices

        else:
            print("Invalid camera configuration.")
            sys.exit()


    def get_num_cameras(self):
        return self.num_cameras
        
    def get_camera_unit_vector(self, i):
        assert i < self.num_cameras
        
        return self.camera_deltas[i]
    
    def get_camera_up(self, i):
        camera_delta = self.camera_deltas[i]
        if abs(camera_delta[1]) > self.radius * 0.9:
            return np.array([-1.0, 0.0, 0.0])
        else:
            return np.array([0.0, -1.0, 0.0])