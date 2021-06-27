import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import shutil
import open3d as o3d
import numpy as np
import json
import matplotlib.pyplot as plt

from utils.pcd_utils import (BBox,
                            transform_pointcloud_to_opengl_coords,
                            rotate_around_axis,
                            origin, normalize_transformation)

import config as cfg
from utils.gaps_utils import read_pts_file
from utils.utils import filter_identities
from utils.parsing_utils import get_dataset_type_from_dataset_name



class ViewerTPoses:
    def __init__(
        self, read_from_raw_obj=False, read_watertight=True, load_boundary_samples=True,
        out_root_dir=None,
        render_video_options="/rhome/ppalafox/workspace/render_video_options",

    ):
        self.time = 0

        self.obj = None
        self.gt_pcd_001_IN  = None
        self.gt_pcd_001_OUT = None
        
        self.show_obj = False
        self.show_gt_pcds    = False

        self.read_from_raw_obj = read_from_raw_obj
        self.read_watertight = read_watertight
        self.load_boundary_samples = load_boundary_samples

        # Recording options
        self.view = "frontal"
        self.render_json            = os.path.join(render_video_options, "render_options.json")
        self.viewpoint_json         = os.path.join(render_video_options, "viewpoint.json")
        self.viewpoint_lateral_json = os.path.join(render_video_options, "viewpoint_lateral.json")
        self.viewpoint_lateral_left_json = os.path.join(render_video_options, "viewpoint_lateral_left.json")
        os.makedirs(render_video_options, exist_ok=True)
        assert out_root_dir is not None, "set an output dir"
        self.out_dir = out_root_dir
        os.makedirs(self.out_dir, exist_ok=True)

        self.initialize()
    
    def initialize(self):

        self.obj_list  = []
        self.names_list = []

        self.gt_pcds_001_IN = []
        self.gt_pcds_001_OUT = []

        loaded_frames = 0
        global_id = -1

        for label in labels_tpose:

            global_id += 1

            if load_every and global_id % load_every != 0:
                continue

            if global_id < from_frame_id:
                continue

            character = label['identity_name']

            if self.read_from_raw_obj:
                obj_dir = f"{raw_data_dir}/data/{label['dataset']}/{character}/obj/a_t_pose"
                obj_filename = f"{obj_dir}/a_t_pose_000001.obj"
                color = [1, 0, 0]
            else:
                obj_dir = f"{data_dir}/{label['dataset']}/{character}/a_t_pose/000000"

                obj_filename = f"{obj_dir}/{MESH_FILENAME}"
                
                if self.read_watertight:
                    watertight_obj_filename = f"{obj_dir}/mesh_watertight_poisson.ply"
                    if os.path.isfile(watertight_obj_filename):
                        obj_filename = watertight_obj_filename
                    else:
                        watertight_obj_filename = f"{obj_dir}/mesh_watertight.ply"
                        if os.path.isfile(watertight_obj_filename):
                            obj_filename = watertight_obj_filename

                else:
                    if not os.path.isfile(obj_filename):
                        obj_filename = f"{obj_dir}/mesh_raw.ply"

                color = [0, 1, 0]

            if not os.path.isfile(obj_filename):
                print("!!! Could not find", obj_filename)
                continue

            print("Loaded", character)

            mesh = o3d.io.read_triangle_mesh(obj_filename)
            mesh.compute_vertex_normals()
            mesh.paint_uniform_color(color)

            self.obj_list.append(mesh)
            self.names_list.append(character)

            ##############################################################################################################
            # Boundary samples
            ##############################################################################################################
            if self.load_boundary_samples:
                # samples_surf_path = os.path.join(obj_dir, 'samples_surface.pts')
                samples_near_path = os.path.join(obj_dir, 'samples_near.sdf')
                samples_unif_path = os.path.join(obj_dir, 'samples_uniform.sdf')

                # surf_points = read_pts_file(samples_surf_path)[:, :3] # discard normals
                near_points = read_pts_file(samples_near_path)
                unif_points = read_pts_file(samples_unif_path)
                
                # Surface
                # surf_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(surf_points))

                # Near
                near_pts = near_points[:, :3]
                near_sdf = near_points[:, -1]
                near_pts_in  = near_pts[np.where(near_sdf < 0.0)]
                near_pts_out = near_pts[np.where(near_sdf > 0.0)]

                # Uniform
                unif_pts = unif_points[:, :3]
                unif_sdf = unif_points[:, -1]
                unif_pts_in  = unif_pts[np.where(unif_sdf < 0.0)]
                unif_pts_out = unif_pts[np.where(unif_sdf > 0.0)]

                if True:
                    all_pts_in  = np.concatenate([near_pts_in, unif_pts_in])
                    all_pts_out = np.concatenate([near_pts_out, unif_pts_out])
                else:
                    all_pts_in  = np.concatenate([near_pts_in])
                    all_pts_out = np.concatenate([near_pts_out])
            
                pts_001_IN_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(all_pts_in))
                pts_001_IN_pcd.paint_uniform_color([0, 0.4, 1])
                self.gt_pcds_001_IN.append(pts_001_IN_pcd)

                pts_001_OUT_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(all_pts_out))
                pts_001_OUT_pcd.paint_uniform_color([1, 0, 0])
                self.gt_pcds_001_OUT.append(pts_001_OUT_pcd)

            ##############################################################################################################
            # ALL
            ##############################################################################################################
            # sdf_samples_path = os.path.join(obj_dir, 'sdf_samples.npz')
            # sdf_samples_npz = np.load(sdf_samples_path)
            # points_sdf = sdf_samples_npz['points_sdf']

            # points, sdf = points_sdf[:, :3], points_sdf[:, 3]

            # pts_001_IN_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[sdf < 0]))
            # pts_001_IN_pcd.paint_uniform_color([0, 0.4, 1])
            # self.gt_pcds_001_IN.append(pts_001_IN_pcd)

            # pts_001_OUT_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[sdf > 0]))
            # pts_001_OUT_pcd.paint_uniform_color([1, 0, 0])
            # self.gt_pcds_001_OUT.append(pts_001_OUT_pcd)

            # break

            # Increase counter of evaluated frames
            loaded_frames += 1

            print(f'Loaded {loaded_frames} frames')

            if loaded_frames == num_samples:
                print()
                print(f"Stopping early. Already loaded {loaded_frames}")
                print()
                break

        print(len(self.obj_list))

        # ###############################################################################################
        # # Generate additional meshes.
        # ###############################################################################################
        self.world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.2, origin=[0, 0, 0]
        )
        
        # # unit bbox
        p_min = -0.5
        p_max =  0.5
        self.unit_bbox = BBox.compute_bbox_from_min_point_and_max_point(
            np.array([p_min]*3), np.array([p_max]*3)
        )
        # self.unit_bbox = rotate_around_axis(self.unit_bbox, axis_name="x", angle=-np.pi) 

    def update_obj(self, vis):
        param = vis.get_view_control().convert_to_pinhole_camera_parameters()

        if self.obj is not None:
            vis.remove_geometry(self.obj)

        if self.show_obj:
            self.obj = self.obj_list[self.time]
            vis.add_geometry(self.obj)

            print(self.time, self.names_list[self.time])

        ctr = vis.get_view_control()
        ctr.convert_from_pinhole_camera_parameters(param)

    def update_gt_pcds_001(self, vis):
        param = vis.get_view_control().convert_to_pinhole_camera_parameters()

        if self.gt_pcd_001_IN is not None:
            vis.remove_geometry(self.gt_pcd_001_IN)
            self.gt_pcd_001_IN  = None

        if self.gt_pcd_001_OUT is not None:
            vis.remove_geometry(self.gt_pcd_001_OUT)
            self.gt_pcd_001_OUT = None

        # If requested, we show a (new) mesh.
        if self.show_gt_pcds and len(self.gt_pcds_001_IN) > 0 and len(self.gt_pcds_001_OUT) > 0:
            
            self.gt_pcd_001_IN = self.gt_pcds_001_IN[self.time]
            vis.add_geometry(self.gt_pcd_001_IN)

            self.gt_pcd_001_OUT = self.gt_pcds_001_OUT[self.time]
            vis.add_geometry(self.gt_pcd_001_OUT)

        ctr = vis.get_view_control()
        ctr.convert_from_pinhole_camera_parameters(param)

    def _load_render_and_viewpoint_option(self, vis, view):
        vis.get_render_option().load_from_json(self.render_json)
        
        # change viewpoint
        ctr = vis.get_view_control()
        if view == "frontal":
            param = o3d.io.read_pinhole_camera_parameters(self.viewpoint_json)
        elif view == "lateral":
            param = o3d.io.read_pinhole_camera_parameters(self.viewpoint_lateral_json)
        elif view == "lateral_left":
            param = o3d.io.read_pinhole_camera_parameters(self.viewpoint_lateral_left_json)
        else:
            exit()
        ctr.convert_from_pinhole_camera_parameters(param)

    def render_image(self, vis, out_filename):
        image_np = np.asarray(vis.capture_screen_float_buffer(False))
        h, w, _ = image_np.shape
        new_h, new_w = 1200, 1600
        image_np = image_np[(h-new_h)//2:(h+new_h)//2, (w-new_w)//2:(w+new_w)//2,:]
        plt.imsave(f"{self.out_dir}/{out_filename}.jpg", image_np)
        
    def run(self):
        # Define callbacks.
        def toggle_next(vis):
            self.time += 1
            if self.time >= len(self.obj_list):
                self.time = 0
            self.update_obj(vis)
            self.update_gt_pcds_001(vis)
            return False
        
        def toggle_previous(vis):
            self.time -= 1
            if self.time < 0:
                self.time = len(self.obj_list) - 1
            self.update_obj(vis)
            self.update_gt_pcds_001(vis)
            return False

        def toggle_mesh(vis):
            self.show_obj = not self.show_obj
            self.update_obj(vis)
            return False

        def toggle_gt_pcds(vis):
            self.show_gt_pcds = not self.show_gt_pcds
            self.update_gt_pcds_001(vis)
            return False

        def take_screenshot_of_current_scene(vis):
            print("::taking screenshot")
            # self._load_render_and_viewpoint_option(vis, self.view)

            self.render_image(vis, f"point_samples")

            return False

        key_to_callback = {}
        key_to_callback[ord("D")] = toggle_next
        key_to_callback[ord("A")] = toggle_previous
        key_to_callback[ord("M")] = toggle_mesh
        key_to_callback[ord("K")] = toggle_gt_pcds
        key_to_callback[ord(",")] = take_screenshot_of_current_scene

        # Add mesh at initial time step.
        assert self.time < len(self.obj_list)

        print("Showing time", self.time)

        self.obj = self.obj_list[self.time]
        self.show_obj = True

        o3d.visualization.draw_geometries_with_key_callbacks([self.unit_bbox, self.obj], key_to_callback)
        # o3d.visualization.draw_geometries_with_key_callbacks([self.world_frame, self.unit_bbox, self.obj], key_to_callback)


if __name__ == "__main__":

    raw_data_dir = "/cluster_HDD/lothlann/ppalafox"
    
    # ------------------------------------------------------- #
    cluster  = "cluster"
    
    # dataset_name = "SHAPE_amass-train-419ts-419seqs"
    # dataset_name = "SHAPE_CAPE-train-41ts-41seqs"
    # dataset_name = "CAPE-SHAPE-TRAIN-35id"
    dataset_name = "MIX-POSE__AMASS-50id-5000__MIXAMO-165id-20000__CAPE-35id-20533"

    # MESH_FILENAME = "mesh_raw.ply"
    MESH_FILENAME = "mesh_normalized.ply"
    # ------------------------------------------------------- #

    data_dir = f"/{cluster}/lothlann/ppalafox/datasets"
    
    splits_dir = f"{cfg.splits_dir}_{get_dataset_type_from_dataset_name(dataset_name)}"
    labels_json = os.path.join(data_dir, splits_dir, dataset_name, "labels_tpose.json")
    input(f"Loading from {labels_json}")

    with open(labels_json, 'r') as f:
        labels_tpose = json.loads(f.read())

    read_from_raw_obj     = False
    read_watertight       = False
    load_boundary_samples = True

    if not load_boundary_samples:
        input("Will not load samples")
    
    num_samples   = 5
    load_every    = 50
    from_frame_id = 0

    viewer = ViewerTPoses(
        read_from_raw_obj=read_from_raw_obj, 
        read_watertight=read_watertight, 
        load_boundary_samples=load_boundary_samples,
        out_root_dir="/cluster_HDD/lothlann/ppalafox/qualitative_results__SHAPE_SAMPLES",
    )
    viewer.run()
