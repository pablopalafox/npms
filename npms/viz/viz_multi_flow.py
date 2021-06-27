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


class ViewerMulti:
    def __init__(
        self, 
        identity_names=[], 
        target_animations=[],
        exclude_identity_names=[],
    ):
        self.identity_names = identity_names
        self.target_animations = target_animations

        self.exclude_identity_names = exclude_identity_names

        self.time = 0

        self.obj = None
        self.p_ref = None
        self.p_cur = None
        self.line = None
        self.gt_pcd_001_IN  = None
        self.gt_pcd_001_OUT = None
        
        self.show_obj = False
        self.show_flow = False
        self.show_gt_pcds = False

        self.initialize()
    
    def initialize(self):

        self.obj_list  = []
        
        self.p_refs_list = {}
        self.p_curs_list = []
        self.lines_list = []

        self.frames_list = []

        self.identity_name_by_frame_id = {}

        self.gt_pcds_001_IN = []
        self.gt_pcds_001_OUT = []

        loaded_frames = 0
        global_id = -1

        is_done = False

        # Prepare identity names
        if len(self.identity_names) == 0:
            self.identity_names = os.listdir(dataset_dir)

        for identity_name in self.identity_names:

            if identity_name in self.exclude_identity_names:
                continue

            print()
            print("Loading", identity_name)

            identity_dir = os.path.join(dataset_dir, identity_name)
            
            # GT mesh
            ref_mesh_path = os.path.join(identity_dir, "a_t_pose", "000000", f"{mesh_filename}.ply")
            ref_mesh = o3d.io.read_triangle_mesh(ref_mesh_path)
            ref_vertices = np.asarray(ref_mesh.vertices)
            p_refA = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(ref_vertices))
            p_refA.paint_uniform_color([0, 0, 1])
            
            # Flow REF
            sigma = 0.002
            ref_sample_path = os.path.join(identity_dir, "a_t_pose", "000000", f"flow_samples_{sigma}.npz")
            if os.path.isfile(ref_sample_path):
                ref_samples_npz = np.load(ref_sample_path)
                p_ref_np = ref_samples_npz['points'].astype(np.float32)
                p_ref = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p_ref_np))
                p_ref.paint_uniform_color([0, 1, 0])
                self.p_refs_list[identity_name] = p_ref
            
            for animation_name in sorted(os.listdir(identity_dir)):

                if len(self.target_animations) > 0 and animation_name not in self.target_animations:
                    continue

                print("\tLoading", animation_name)
                
                animation_dir = os.path.join(identity_dir, animation_name)

                if not os.path.isdir(animation_dir):
                    print("Skipping", animation_dir)
                    continue

                for frame_id in sorted(os.listdir(animation_dir)):

                    global_id += 1

                    if load_every and global_id % load_every != 0:
                        continue

                    if global_id < from_frame_id:
                        continue

                    print("Loading", identity_name, animation_name, frame_id)

                    #################################################################
                    # GT MESH
                    #################################################################
                    obj_filename = os.path.join(animation_dir, frame_id, f"{mesh_filename}.ply")

                    if not os.path.isfile(obj_filename):
                        print("Could not find", obj_filename)
                        continue

                    mesh = o3d.io.read_triangle_mesh(obj_filename)
                    mesh.compute_vertex_normals()
                    mesh.paint_uniform_color([1, 0, 0])

                    # trimesh
                    # mesh = trimesh.load(obj_filename, process=False)
                    # bbox_bounds = mesh.bounds
                    # bbox_min = bbox_bounds[0]
                    # print(bbox_min[1])

                    self.obj_list.append(mesh)
                    self.frames_list.append(obj_filename)

                    #################################################################
                    # FLOW
                    #################################################################
                    cur_sample_path = os.path.join(identity_dir, animation_name, frame_id, f"flow_samples_{sigma}.npz")
                    if os.path.isfile(cur_sample_path):
                        cur_samples_npz = np.load(cur_sample_path)
                        p_cur_np = cur_samples_npz['points'].astype(np.float32)
                        p_cur = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p_cur_np))
                        p_cur.paint_uniform_color([0, 0, 1])

                        # LINE
                        corresp = [(k, k) for k in range(0, p_cur_np.shape[0])]
                        lines = o3d.geometry.LineSet.create_from_point_cloud_correspondences(p_cur, p_ref, corresp)
                        lines.paint_uniform_color([0.2, 0.8, 0.8])

                        if show_vertices:
                            cur_vertices = np.asarray(mesh.vertices)
                            p_curA = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cur_vertices))
                            p_curA.paint_uniform_color([0, 0, 1])
                            # LINE
                            corresp = [(k, k) for k in range(0, ref_vertices.shape[0])]
                            lines = o3d.geometry.LineSet.create_from_point_cloud_correspondences(p_refA, p_curA, corresp)
                            lines.paint_uniform_color([0.2, 0.8, 0.8])

                        self.p_curs_list.append(p_cur)
                        self.lines_list.append(lines)
                    else:
                        raise Exception(cur_sample_path)

                    self.identity_name_by_frame_id[loaded_frames] = identity_name

                    # Increase counter of evaluated frames
                    loaded_frames += 1

                    print(f'LOADED {loaded_frames} frames')

                    if loaded_frames == num_samples:
                        print()
                        print(f"Stopping early. Already loaded {loaded_frames}")
                        print()
                        is_done = True
                        break

                if is_done:
                    break
            
            if is_done:
                break


        print()
        print("Total num frames loaded:", len(self.obj_list))

        # ###############################################################################################
        # # Generate additional meshes.
        # ###############################################################################################
        self.world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.1, origin=[0, 0, 0]
        )
        
        # unit bbox
        p_min = -0.5
        p_max =  0.5
        self.unit_bbox = BBox.compute_bbox_from_min_point_and_max_point(
            np.array([p_min]*3), np.array([p_max]*3)
        )
        self.unit_bbox = rotate_around_axis(self.unit_bbox, axis_name="x", angle=-np.pi) 

        # floor
        # width, height, depth = 10, 10, 10
        # self.mesh_box = o3d.geometry.TriangleMesh.create_box(width=width,
        #                                                      height=height,
        #                                                      depth=depth)
        # self.mesh_box.compute_vertex_normals()
        # self.mesh_box.paint_uniform_color([0.7, 0.7, 0.7])
        # self.mesh_box.translate((-width/2, -height/2, -depth/2))

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

    def update_flow(self, vis):
        param = vis.get_view_control().convert_to_pinhole_camera_parameters()

        if self.p_ref is not None:
            vis.remove_geometry(self.p_ref)
            vis.remove_geometry(self.p_cur)
            vis.remove_geometry(self.line)

        if self.show_flow:
            self.p_ref = self.p_refs_list[self.identity_name_by_frame_id[self.time]]
            vis.add_geometry(self.p_ref)

            self.p_cur = self.p_curs_list[self.time]
            vis.add_geometry(self.p_cur)

            self.line = self.lines_list[self.time]
            vis.add_geometry(self.line)

            # print(self.time, self.frames_list[self.time])

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
            self.update_flow(vis)
            # self.update_gt_pcds_001(vis)
            return False
        
        def toggle_previous(vis):
            self.time -= 1
            if self.time < 0:
                self.time = len(self.obj_list) - 1
            self.update_obj(vis)
            self.update_flow(vis)
            # self.update_gt_pcds_001(vis)
            return False

        def toggle_mesh(vis):
            self.show_obj = not self.show_obj
            self.update_obj(vis)
            return False

        def toggle_flow(vis):
            self.show_flow = not self.show_flow
            self.update_flow(vis)
            return False

        # def toggle_gt_pcds(vis):
        #     self.show_gt_pcds = not self.show_gt_pcds
        #     self.update_gt_pcds_001(vis)
        #     return False

        key_to_callback = {}
        key_to_callback[ord("D")] = toggle_next
        key_to_callback[ord("A")] = toggle_previous
        key_to_callback[ord("M")] = toggle_mesh
        key_to_callback[ord("F")] = toggle_flow

        # Add mesh at initial time step.
        assert self.time < len(self.obj_list), len(self.obj_list)

        print("Showing time", self.time)

        self.obj = self.obj_list[self.time]
        self.show_obj = True

        o3d.visualization.draw_geometries_with_key_callbacks([self.unit_bbox, self.world_frame, self.obj], key_to_callback)


if __name__ == "__main__":

    mesh_filename = "mesh_normalized"

    dataset = "cape"
    dataset_dir = f"/cluster/lothlann/ppalafox/datasets/{dataset}"

    num_samples = 100
    load_every = 200
    from_frame_id = 0
    
    show_vertices = False

    viewer = ViewerMulti(
        # identity_names=['Transitionsmocap_s033'],
        # target_animations=['motion010'],
        exclude_identity_names=cfg.test_identities_cape,
    )
    viewer.run()

    