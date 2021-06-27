import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import shutil
import open3d as o3d
import numpy as np
import trimesh
import json

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
        exclude_identities_with=None,
        only_identities_with=None,
    ):
        self.identity_names = identity_names
        self.target_animations = target_animations

        self.time = 0

        self.p_ref = None
        self.p_cur = None
        self.mesh_cur = None
        
        self.show_ref = False
        self.show_cur = False

        self.exclude_identities_with = exclude_identities_with
        self.only_identities_with = only_identities_with

        self.initialize()
    
    def initialize(self):

        self.p_refs_list = {}
        self.p_curs_list = []
        self.mesh_curs_list = []

        self.frames_list = []

        self.identity_name_by_frame_id = {}
        self.animation_frame_id_by_frame_id = {}

        loaded_frames = 0
        global_id = -1

        is_done = False

        # Prepare identity names
        if len(self.identity_names) == 0:
            self.identity_names = os.listdir(dataset_dir)

        for identity_name in self.identity_names:

            identity_dir = os.path.join(dataset_dir, identity_name)

            if not os.path.isdir(identity_dir):
                print(f"Skipping {identity_dir}, since it's not a directory")
                continue

            if self.exclude_identities_with is not None and self.exclude_identities_with in identity_name:
                continue

            if self.only_identities_with is not None and self.only_identities_with not in identity_name:
                continue
            
            # T-Pose
            ref_mesh_path = os.path.join(identity_dir, "a_t_pose", "000000", f"{mesh_filename}")
            if os.path.isfile(ref_mesh_path):
                ref_mesh = o3d.io.read_triangle_mesh(ref_mesh_path)
                ref_vertices = np.asarray(ref_mesh.vertices)
                p_ref = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(ref_vertices))
                p_ref.paint_uniform_color([0, 1, 0])
                self.p_refs_list[identity_name] = p_ref
            
            # Go over animations
            for animation_name in sorted(os.listdir(identity_dir)):

                if len(self.target_animations) > 0 and animation_name not in self.target_animations:
                    continue

                animation_dir = os.path.join(identity_dir, animation_name)

                if not os.path.isdir(animation_dir):
                    continue
                
                # Go over the frames in the animation
                for frame_id in sorted(os.listdir(animation_dir)):

                    global_id += 1

                    if load_every and global_id % load_every != 0:
                        continue

                    if global_id < from_frame_id:
                        continue

                    #################################################################
                    # Current
                    #################################################################
                    cur_mesh_path = os.path.join(animation_dir, frame_id, f"{mesh_filename}")

                    if not os.path.isfile(cur_mesh_path):
                        print(f"File {cur_mesh_path} is not a mesh")
                        continue

                    print("Loading", identity_name, animation_name, frame_id)

                    cur_mesh = o3d.io.read_triangle_mesh(cur_mesh_path)
                    cur_mesh.compute_vertex_normals()
                    cur_vertices = np.asarray(cur_mesh.vertices)
                    p_cur = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cur_vertices))
                    p_cur.paint_uniform_color([0, 0, 1])
                    self.p_curs_list.append(p_cur)
                    self.mesh_curs_list.append(cur_mesh)

                    self.identity_name_by_frame_id[loaded_frames] = identity_name
                    self.animation_frame_id_by_frame_id[loaded_frames] = f"{animation_name}_{frame_id}"

                    # Increase counter of evaluated frames
                    loaded_frames += 1

                    print(f'Loaded {loaded_frames} frames')

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
        print("Total num frames loaded:", len(self.p_curs_list))

        # ###############################################################################################
        # # Generate additional meshes.
        # ###############################################################################################
        self.world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.2, origin=[0, 0, 0]
        )
        
        # unit bbox
        p_min = -0.5
        p_max =  0.5
        self.unit_bbox = BBox.compute_bbox_from_min_point_and_max_point(
            np.array([p_min]*3), np.array([p_max]*3)
        )
        print((np.array([p_min]*3) + np.array([p_max]*3)) / 2.)

        # Load bbox used to normalize
        bbox_json_path = os.path.join(dataset_dir, "bbox.json")
        if os.path.isfile(bbox_json_path):
            with open(bbox_json_path, 'r') as f:
                bbox = json.loads(f.read())
                bbox = np.array(bbox)
            self.max_bbox = BBox.compute_bbox_from_min_point_and_max_point(
                np.array(bbox[0]), np.array(bbox[1])
            )
       
    def update_ref(self, vis):
        param = vis.get_view_control().convert_to_pinhole_camera_parameters()

        if self.p_ref is not None:
            vis.remove_geometry(self.p_ref)

        if self.show_ref:
            if self.identity_name_by_frame_id[self.time] in self.p_refs_list:
                self.p_ref = self.p_refs_list[self.identity_name_by_frame_id[self.time]]
                vis.add_geometry(self.p_ref)

        ctr = vis.get_view_control()
        ctr.convert_from_pinhole_camera_parameters(param)

    def update_cur(self, vis):
        param = vis.get_view_control().convert_to_pinhole_camera_parameters()

        if self.p_cur is not None:
            vis.remove_geometry(self.p_cur)
        if self.mesh_cur is not None:
            vis.remove_geometry(self.mesh_cur)

        if self.show_cur:
            self.p_cur = self.p_curs_list[self.time]
            vis.add_geometry(self.p_cur)

            self.mesh_cur = self.mesh_curs_list[self.time]
            vis.add_geometry(self.mesh_cur)

        ctr = vis.get_view_control()
        ctr.convert_from_pinhole_camera_parameters(param)

    def run(self):
        # Define callbacks.
        def toggle_next(vis):
            self.time += 1
            if self.time >= len(self.p_curs_list):
                self.time = 0
            print(f"Frame: {self.time} - Identity: {self.identity_name_by_frame_id[self.time]} - {self.animation_frame_id_by_frame_id[self.time]}")
            self.update_ref(vis)
            self.update_cur(vis)
            return False
        
        def toggle_previous(vis):
            self.time -= 1
            if self.time < 0:
                self.time = len(self.p_curs_list) - 1
            print(f"Frame: {self.time} - Identity: {self.identity_name_by_frame_id[self.time]} - {self.animation_frame_id_by_frame_id[self.time]}")
            self.update_ref(vis)
            self.update_cur(vis)
            return False

        def toggle_ref(vis):
            self.show_ref = not self.show_ref
            self.update_ref(vis)
            return False

        def toggle_cur(vis):
            self.show_cur = not self.show_cur
            self.update_cur(vis)
            return False

        key_to_callback = {}
        key_to_callback[ord("D")] = toggle_next
        key_to_callback[ord("A")] = toggle_previous
        key_to_callback[ord("R")] = toggle_ref
        key_to_callback[ord("C")] = toggle_cur

        # Add mesh at initial time step.
        assert self.time < len(self.p_curs_list)

        print("Showing time", self.time)

        self.p_cur = self.p_curs_list[self.time]
        self.mesh_cur = self.mesh_curs_list[self.time]
        self.show_cur = True

        o3d.visualization.draw_geometries_with_key_callbacks([self.unit_bbox, self.world_frame, self.p_cur, self.mesh_cur], key_to_callback)
        # o3d.visualization.draw_geometries_with_key_callbacks(
        #     [self.max_bbox, self.world_frame, self.unit_bbox, self.p_cur, self.mesh_cur], key_to_callback
        # )


if __name__ == "__main__":
    
    PROCESSED_MESH_FILENAME = "mesh_raw.ply"
    REAL_SCAN_MESH_FILENAME = "mesh_real_scan.ply"
    NORMALIZED_MESH_FILENAME = "mesh_normalized.ply"

    # mesh_filename = "mesh_raw"
    mesh_filename = NORMALIZED_MESH_FILENAME

    cluster_location = "cluster"
    dataset = "mixamo_trans_all"
    dataset_dir = f"/{cluster_location}/lothlann/ppalafox/datasets/{dataset}"

    num_samples = 2
    load_every = 0
    from_frame_id = 0
    viewer = ViewerMulti(
        # identity_names=['00096_jerseyshort'],
        # target_animations=['ballerina_spin'],
        # ["astra"],
        # [cfg.identities[29]],
        # cfg.identities,
        # target_animations
        # exclude_identities_with='test',
        # only_identities_with='test',
    )
    viewer.run()