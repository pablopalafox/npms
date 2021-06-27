import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import shutil
import open3d as o3d
import numpy as np
import trimesh

from utils.pcd_utils import (BBox,
                            transform_pointcloud_to_opengl_coords,
                            rotate_around_axis,
                            origin, normalize_transformation)

from data_scripts import config_data as cfg
from utils.gaps_utils import read_pts_file
from utils.utils import filter_identities


class ViewerIdentityAnimations:
    def __init__(
        self, identity_name, animation_name
    ):
        self.identity_name = identity_name
        self.animation_name = animation_name

        self.time = 0

        self.obj = None
        self.gt_pcd_001_IN  = None
        self.gt_pcd_001_OUT = None
        
        self.show_obj = False
        self.show_gt_pcds    = False

        self.initialize()
    
    def initialize(self):

        self.obj_list  = []
        self.frames_list = []

        self.gt_pcds_001_IN = []
        self.gt_pcds_001_OUT = []

        read_obj = False

        if read_obj:
            animation_path = os.path.join(
                cfg.root_in, "data", "mixamo", self.identity_name, "obj", self.animation_name
            )
        else:
            animation_path = os.path.join(
                cfg.root_out, dataset_type, "mixamo", self.identity_name, self.animation_name
            )

        for frame in sorted(os.listdir(animation_path)):

            print("Loading", self.identity_name, self.animation_name)

            if read_obj:
                obj_filename = os.path.join(animation_path, frame)
            else:
                obj_filename = os.path.join(animation_path, frame, "mesh_normalized.ply")

            if not os.path.isfile(obj_filename):
                print(obj_filename)
                continue

            mesh = o3d.io.read_triangle_mesh(obj_filename)
            mesh.compute_vertex_normals()
            mesh.paint_uniform_color([1, 0, 0])

            # trimesh
            # mesh_tmp = trimesh.load(obj_filename, process=False)
            # bbox_bounds = mesh_tmp.bounds
            # bbox_min = bbox_bounds[0]
            # print(bbox_min[1])

            self.obj_list.append(mesh)
            self.frames_list.append(frame)

            # ##############################################################################################################
            # # 0.01
            # ##############################################################################################################
            # sigma = 0.01
            # points_001_path = os.path.join(obj_dir, f'boundary_samples_{sigma}.npz')
            # points_001_path_dict = np.load(points_001_path)
            # pts_001 = points_001_path_dict['points']
            # grd_001 = points_001_path_dict['grid_coords']
            # occ_001 = points_001_path_dict['occupancies']
            # # IN
            # pts_001_IN = pts_001[occ_001 == True]
            # grd_001_IN = grd_001[occ_001 == True]
            # pts_001_IN_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_001_IN))
            # grd_001_IN_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(grd_001_IN))
            # pts_001_IN_pcd.paint_uniform_color([0, 0.4, 1])
            # grd_001_IN_pcd.paint_uniform_color([0, 0.4, 1])
            # self.gt_pcds_001_IN.append(pts_001_IN_pcd)
            # # self.gt_grds_001_IN.append(grd_001_IN_pcd)
            # # OUT
            # pts_001_OUT = pts_001[occ_001 == False]
            # points_001_pcd_OUT = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_001_OUT))
            # points_001_pcd_OUT.paint_uniform_color([1, 0, 0])
            # self.gt_pcds_001_OUT.append(points_001_pcd_OUT)

        print(len(self.obj_list))

        # ###############################################################################################
        # # Generate additional meshes.
        # ###############################################################################################
        self.world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.1, origin=[0, 0, 0]
        )
        
        # # unit bbox
        # p_min = -0.5
        # p_max =  0.5
        # self.unit_bbox = BBox.compute_bbox_from_min_point_and_max_point(
        #     np.array([p_min]*3), np.array([p_max]*3)
        # )
        # self.unit_bbox = rotate_around_axis(self.unit_bbox, axis_name="x", angle=-np.pi) 

    def update_obj(self, vis):
        param = vis.get_view_control().convert_to_pinhole_camera_parameters()

        if self.obj is not None:
            vis.remove_geometry(self.obj)

        if self.show_obj:
            self.obj = self.obj_list[self.time]
            vis.add_geometry(self.obj)

            print(self.time, self.frames_list[self.time])

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
        if self.show_gt_pcds:
            
            self.gt_pcd_001_IN = self.gt_pcds_001_IN[self.time]
            vis.add_geometry(self.gt_pcd_001_IN)

            self.gt_pcd_001_OUT = self.gt_pcds_001_OUT[self.time]
            vis.add_geometry(self.gt_pcd_001_OUT)

        ctr = vis.get_view_control()
        ctr.convert_from_pinhole_camera_parameters(param)

    def run(self):
        # Define callbacks.
        def toggle_next(vis):
            self.time += 1
            if self.time >= len(self.obj_list):
                self.time = 0
            self.update_obj(vis)
            # self.update_gt_pcds_001(vis)
            return False
        
        def toggle_previous(vis):
            self.time -= 1
            if self.time < 0:
                self.time = len(self.obj_list) - 1
            self.update_obj(vis)
            # self.update_gt_pcds_001(vis)
            return False

        def toggle_mesh(vis):
            self.show_obj = not self.show_obj
            self.update_obj(vis)
            return False

        # def toggle_gt_pcds(vis):
        #     self.show_gt_pcds = not self.show_gt_pcds
        #     self.update_gt_pcds_001(vis)
        #     return False

        key_to_callback = {}
        key_to_callback[ord("D")] = toggle_next
        key_to_callback[ord("A")] = toggle_previous
        key_to_callback[ord("M")] = toggle_mesh
        # key_to_callback[ord("P")] = toggle_gt_pcds

        # Add mesh at initial time step.
        assert self.time < len(self.obj_list)

        print("Showing time", self.time)

        self.obj = self.obj_list[self.time]
        self.show_obj = True

        o3d.visualization.draw_geometries_with_key_callbacks([self.world_frame, self.obj], key_to_callback)


if __name__ == "__main__":


    animation_name = "capoeira_idle"

    viewer = ViewerIdentityAnimations(
        "alex",
        animation_name
    )
    viewer.run()