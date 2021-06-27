from genericpath import exists
import os
import numpy as np
import trimesh
import open3d as o3d
from tqdm import tqdm
from tqdm.contrib import tzip
import matplotlib.pyplot as plt
import shutil
import json
import glob
from utils.utils import query_yes_no

from utils.voxels import VoxelGrid
from utils.evaluation import eval_mesh, compute_accuracy_per_predicted_vertex, compute_completeness_per_gt_vertex

from utils.pcd_utils import (BBox,
                            open3d_to_trimesh,
                            filter_mesh_o3d,
                            rotate_around_axis)

import config as cfg

import copy


SCALE_VERTEX_COLOR = 1.8


def compute_vertex_colors_from_distances(dist, vertices):
    weight_shift = 0.0
    weight_scale = 2
    weights = dist / np.max(dist)

    # print(np.mean(weights), np.median(weights))
    # print(np.min(weights), np.max(weights))

    weights = weight_shift + weights * weight_scale
    weights = np.clip(weights, 0.0, 1.0)

    # print(np.mean(weights), np.median(weights))
    # print(np.min(weights), np.max(weights))

    assert np.max(weights) <= 1.0, np.max(weights)
    assert np.min(weights) >= 0.0, np.min(weights)

    high_error_color, low_error_color = np.array([0.5, 0.0, 0]), np.array([0.0, 0.7, 0.0])

    vertex_colors = np.zeros_like(vertices)
    vertex_colors[:, 0] = weights * high_error_color[0] + (1. - weights) * low_error_color[0]
    vertex_colors[:, 1] = weights * high_error_color[1] + (1. - weights) * low_error_color[1]
    vertex_colors[:, 2] = weights * high_error_color[2] + (1. - weights) * low_error_color[2]

    return vertex_colors


class ViewerFinal:
    def __init__(
        self, 
        gt_path_list,
        pred_path_list,
        video_dir,
        viz_type,
        filter_mesh_flag,
        view,
        num_to_eval=-1,
        load_every=None,
        from_frame_id=0,
        # recording options:
        record_directly=False,
        frame_rate=30,
        render_video_options="/rhome/ppalafox/workspace/render_video_options",
        scale=None,
        translation=None,
    ):
        self.scale = scale
        self.translation = translation

        self.viz_type = viz_type
        self.filter_mesh_flag = filter_mesh_flag
        
        self.gt_path_list = gt_path_list
        self.pred_path_list = pred_path_list

        self.num_to_eval = num_to_eval
        self.load_every = load_every
        self.from_frame_id = from_frame_id

        self.time = 0

        self.gt   = None
        self.pred = None

        self.show_pred = False
        self.show_gt   = False

        # Recording options
        self.view = view
        self.record_directly = record_directly
        self.render_json            = os.path.join(render_video_options, "render_options.json")
        self.viewpoint_json         = os.path.join(render_video_options, "viewpoint.json")
        self.viewpoint_lateral_json = os.path.join(render_video_options, "viewpoint_lateral.json")
        self.viewpoint_back_json = os.path.join(render_video_options, "viewpoint_back.json")
        os.makedirs(render_video_options, exist_ok=True)
        self.frame_rate = frame_rate
        self.video_dir = video_dir
        os.makedirs(self.video_dir, exist_ok=True)

        self.animating = False
        self.stop_animation = False # Flag to cancel current animation
        self.num_circles = 1 # Hardcoded to do only one circle when camera_motion="rotation"

        self.initialize()
    
    def initialize(self):

        self.pred_list = []
        self.gt_list   = []
        self.scan_list = []
        self.vox_list  = []
        self.pcd_list  = []

        ########################################################################################
        ########################################################################################
        # Go over all frames in the sequence
        ########################################################################################
        ########################################################################################
        self.loaded_frames = 0

        # Vertex colors
        vertex_colors = None
        
        for gt_mesh_path, pred_mesh_path in tzip(self.gt_path_list, self.pred_path_list):
            
            print("++ Loading", gt_mesh_path)

            ############################################################
            # Load groundtruth mesh
            ############################################################
            gt_mesh_o3d = o3d.io.read_triangle_mesh(gt_mesh_path)
            gt_mesh_o3d.paint_uniform_color([0, 0.4, 0])
            gt_mesh_o3d.compute_vertex_normals()

            ############################################################
            # Load predicted mesh
            ############################################################
            pred_mesh_o3d = o3d.io.read_triangle_mesh(pred_mesh_path)
            pred_mesh_o3d.compute_vertex_normals()
            
            if self.filter_mesh_flag:
                pred_mesh_o3d = filter_mesh_o3d(pred_mesh_o3d, cylinder_radius=0.4)

            # Maybe scale
            if self.scale is not None:
                pred_mesh_o3d.scale(self.scale, center=(0,0,0))
            if self.translation is not None:
                pred_mesh_o3d = copy.deepcopy(pred_mesh_o3d).translate(self.translation)

            ############################################################
            # PAINT groundtruth mesh
            ############################################################
            if self.viz_type == "completeness":
                gt_mesh_trimesh   = open3d_to_trimesh(gt_mesh_o3d)
                pred_mesh_trimesh = open3d_to_trimesh(pred_mesh_o3d)

                completeness_per_vertex = compute_completeness_per_gt_vertex(pred_mesh_trimesh, gt_mesh_trimesh)
                vertex_colors = compute_vertex_colors_from_distances(completeness_per_vertex, gt_mesh_trimesh.vertices)
                gt_mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

            self.gt_list.append(gt_mesh_o3d)
            
            ############################################################
            # PAINT predicted mesh
            ############################################################
            if self.viz_type == "accuracy":
                gt_mesh_trimesh   = open3d_to_trimesh(gt_mesh_o3d)
                pred_mesh_trimesh = open3d_to_trimesh(pred_mesh_o3d)

                accuracy_per_vertex = compute_accuracy_per_predicted_vertex(pred_mesh_trimesh, gt_mesh_trimesh)
                vertex_colors = compute_vertex_colors_from_distances(accuracy_per_vertex, pred_mesh_trimesh.vertices)
                pred_mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
            elif self.viz_type is None:
                # Initialize vertex colors if necessary
                if vertex_colors is None:
                    bbox_min = -0.5
                    bbox_max = 0.5
                    half_box = (bbox_max - bbox_min) / 2.0 
                    scale = SCALE_VERTEX_COLOR
                    vertex_colors = (scale * np.array(pred_mesh_o3d.vertices) + half_box) / (2.0 * half_box)
                    vertex_colors = np.clip(vertex_colors, 0.0, 1.0)
                
                if not self.filter_mesh_flag:
                    pred_mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

            self.pred_list.append(pred_mesh_o3d)

            # Increase counter of evaluated frames
            self.loaded_frames += 1

            print(f'Loaded {self.loaded_frames} frames')

            if self.loaded_frames == self.num_to_eval:
                print()
                print(f"Stopping early. Already loaded {self.loaded_frames}")
                print()
                break

        ###############################################################################################
        # Generate additional meshes.
        ###############################################################################################
        # unit bbox
        p_min = -0.5
        p_max =  0.5
        self.unit_bbox = BBox.compute_bbox_from_min_point_and_max_point(
            np.array([p_min]*3), np.array([p_max]*3), color=[0.7, 0.7, 0.7]
        )

        # world frame
        self.world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.0001, origin=[0, 0, 0]
        )

    def update_pred(self, vis):
        param = vis.get_view_control().convert_to_pinhole_camera_parameters()

        if self.pred is not None:
            vis.remove_geometry(self.pred)
        
        if self.show_pred:
            self.pred = self.pred_list[self.time]
            vis.add_geometry(self.pred)

        ctr = vis.get_view_control()
        ctr.convert_from_pinhole_camera_parameters(param)

    def update_gt(self, vis):
        param = vis.get_view_control().convert_to_pinhole_camera_parameters()

        if self.gt is not None:
            vis.remove_geometry(self.gt)
        
        if self.show_gt:
            self.gt = self.gt_list[self.time]
            vis.add_geometry(self.gt)


        ctr = vis.get_view_control()
        ctr.convert_from_pinhole_camera_parameters(param)

    def _load_render_and_viewpoint_option(self, vis, view):
        if self.animating:
            self.stop_animation = True

        vis.get_render_option().load_from_json(self.render_json)
        
        # change viewpoint
        ctr = vis.get_view_control()
        if view == "frontal":
            param = o3d.io.read_pinhole_camera_parameters(self.viewpoint_json)
        elif view == "back":
            param = o3d.io.read_pinhole_camera_parameters(self.viewpoint_back_json)
        elif view == "lateral":
            param = o3d.io.read_pinhole_camera_parameters(self.viewpoint_lateral_json)
        else:
            exit()
        ctr.convert_from_pinhole_camera_parameters(param)

    def run(self):
        # Define callbacks.
        def toggle_next(vis):
            self.time += 1
            if self.time >= self.loaded_frames:
                self.time = 0
            
            print(f"time {self.time}")

            self.update_gt(vis)
            self.update_pred(vis)
            return False
        
        def toggle_previous(vis):
            self.time -= 1
            if self.time < 0:
                self.time = self.loaded_frames - 1
            
            print(f"time {self.time}")

            self.update_gt(vis)
            self.update_pred(vis)
            return False

        def toggle_gt(vis):
            self.show_gt = not self.show_gt
            self.update_gt(vis)
            return False

        def toggle_pred(vis):
            self.show_pred = not self.show_pred
            self.update_pred(vis)
            return False

        # def rotate_and_record(vis):
        #     rotate(vis, animate_character=True, record_video=True)

        # def rotate(vis, animate_character=True, record_video=False):
        #     print("::rotate")

        #     self._load_render_and_viewpoint_option(vis, "frontal")
            
        #     if self.viz_type == "completeness":
        #         self.update_gt(vis)
        #     else:
        #         self.update_pred(vis)

        #     vis.poll_events()
        #     vis.update_renderer()

        #     self.animating = True
        #     self.stop_animation = False

        #     if record_video:
        #         rotation_dir = os.path.join(self.video_dir, "rotation")
        #         os.makedirs(rotation_dir, exist_ok=True)

        #     total = 2094 * self.num_circles
        #     speed = 5.0
        #     n = int(total / speed)

        #     for i in range(n):
                
        #         if self.stop_animation:
        #             self.stop_animation = False
        #             break
                
        #         ctr = vis.get_view_control()

        #         if record_video:
        #             # Render image
        #             image_np = np.asarray(vis.capture_screen_float_buffer(False))
        #             # Crop
        #             h, w, _ = image_np.shape
        #             new_h, new_w = 1200, 1600
        #             image_np = image_np[(h-new_h)//2:(h+new_h)//2, (w-new_w)//2:(w+new_w)//2,:]
        #             # Save
        #             plt.imsave(f"{rotation_dir}/{i:05d}.jpg", image_np)

        #         if animate_character and i % 3 == 0:
        #             toggle_next(vis)

        #         ctr.rotate(speed, 0.0)
        #         vis.poll_events()
        #         vis.update_renderer()

        #     ctr.rotate(0.4, 0.0)
        #     vis.poll_events()
        #     vis.update_renderer()

        #     if record_video:
        #         os.system(f"ffmpeg -r {self.frame_rate} -i {rotation_dir}/%05d.jpg -c:v libx264 -vf fps=1 -pix_fmt yuv420p -y {rotation_dir}/movie.mp4")
        #         exit()
            
        #     return False

        def animate_and_record(vis):
            animate(vis, record_video=True)

        def animate(vis, record_video=False):
            print("::animate")

            self._load_render_and_viewpoint_option(vis, self.view)

            # Start at the first frame
            self.time = 0

            num_meshes = len(self.pred_list)

            if self.viz_type == "completeness":
                self.update_gt(vis)
            else:
                self.update_pred(vis)
            
            vis.poll_events()
            vis.update_renderer()

            self.animating = True
            self.stop_animation = False

            if record_video:
                if self.viz_type is not None:
                    no_motion_dir = os.path.join(self.video_dir, f"animate_{self.view}__{viz_type.upper()}")
                else:
                    no_motion_dir = os.path.join(self.video_dir, f"animate_{self.view}")
                images_dir = os.path.join(no_motion_dir, "images")
                os.makedirs(images_dir, exist_ok=True)

            for i in range(num_meshes):
                
                if self.stop_animation:
                    self.stop_animation = False
                    break
                
                if record_video:
                    image_np = np.asarray(vis.capture_screen_float_buffer(False))
                    h, w, _ = image_np.shape
                    new_h, new_w = 1200, 1600
                    image_np = image_np[(h-new_h)//2:(h+new_h)//2, (w-new_w)//2:(w+new_w)//2,:]
                    plt.imsave(f"{images_dir}/{i:05d}.jpg", image_np)

                toggle_next(vis)

                vis.poll_events()
                vis.update_renderer()

            if record_video:
                os.system(f"ffmpeg -r {self.frame_rate} -i {images_dir}/%05d.jpg -c:v libx264 -vf fps=30 -pix_fmt yuv420p -y {no_motion_dir}/video.mp4")
                exit()
            
            return False

        def stop_animation(vis):
            self.stop_animation = True

        def load_render_and_viewpoint_option(vis):
            print("::load_render_and_viewpoint_option")
            self._load_render_and_viewpoint_option(vis, "frontal")
            return False

        def load_render_and_viewpoint_lateral_option(vis):
            print("::load_render_and_viewpoint_lateral_option")
            self._load_render_and_viewpoint_option(vis, "lateral")
            return False
        
        def save_viewpoint(vis):
            print("::save_viewpoint")
            param = vis.get_view_control().convert_to_pinhole_camera_parameters()
            o3d.io.write_pinhole_camera_parameters(self.viewpoint_json, param)
            return False

        def save_viewpoint_lateral(vis):
            print("::save_viewpoint")
            param = vis.get_view_control().convert_to_pinhole_camera_parameters()
            o3d.io.write_pinhole_camera_parameters(self.viewpoint_lateral_json, param)
            return False

        key_to_callback = {}
        key_to_callback[ord("D")] = toggle_next
        key_to_callback[ord("A")] = toggle_previous
        key_to_callback[ord("C")] = toggle_gt
        key_to_callback[ord("W")] = toggle_pred

        key_to_callback[ord("N")] = animate
        key_to_callback[ord("M")] = animate_and_record
        # key_to_callback[ord(",")] = rotate
        # key_to_callback[ord(".")] = rotate_and_record
        key_to_callback[ord("/")] = stop_animation
        key_to_callback[ord("[")] = save_viewpoint
        key_to_callback[ord("]")] = load_render_and_viewpoint_option
        key_to_callback[ord("'")] = save_viewpoint_lateral
        key_to_callback[ord("\\")] = load_render_and_viewpoint_lateral_option

        # Add mesh at initial time step.
        assert self.time < self.loaded_frames, f"{self.time} is not < {self.loaded_frames}"

        print("Showing time", self.time)

        if self.viz_type == "completeness":
            self.gt = self.gt_list[self.time]
            self.show_gt = True 
            o3d.visualization.draw_geometries_with_key_callbacks([self.world_frame, self.gt], key_to_callback)
        else:
            self.pred = self.pred_list[self.time]
            self.show_pred = True 
            o3d.visualization.draw_geometries_with_key_callbacks([self.world_frame, self.pred], key_to_callback)



if __name__ == "__main__":

    # If None, we color the mesh based on the initial frame. 
    # Otherwise, we color meshes based on the accuracy or completeness error
    viz_type = None
    VIZ_TYPES = ["accuracy", "completeness", None]
    assert viz_type in VIZ_TYPES

    # List of available methods
    AVAILABLE_METHODS = [
        "NPM",
        "SMPL",
        "OCCFLOW",
        "IFNET",   
        "IPNET",   
    ]
    AVAILABLE_METHODS = [AVAILABLE_METHODS].extend([m.lower() for m in AVAILABLE_METHODS])

    # List of available data types
    AVAILABLE_DATA_CLASSES = [
        "HUMAN",
        "MANO",
    ]
    AVAILABLE_DATA_CLASSES = [AVAILABLE_DATA_CLASSES].extend([dc.lower() for dc in AVAILABLE_DATA_CLASSES])

    import argparse
    parser = argparse.ArgumentParser(description="Viz")
    parser.add_argument('-n', '--num_to_eval', default=-1, type=int)
    parser.add_argument('-v', '--view', default='frontal', choices=['frontal', 'lateral', 'back'])
    parser.add_argument('-m', '--method', required=True, choices=AVAILABLE_METHODS)
    parser.add_argument('-d', '--data_class', choices=AVAILABLE_DATA_CLASSES, default='HUMAN')
    args = parser.parse_args()

    # Make them uppercase
    view       = args.view
    method     = args.method.upper()
    data_class = args.data_class.upper()


    #######################################################################################
    
    # ------------------------------------------- #
    print()
    print("#"*30)
    print(f"VIEW:   {view}")
    print(f"METHOD: {method}")
    print(f"CLASS:  {data_class}")
    print("#"*30)
    print()
    print("Continue?")
    # ------------------------------------------- #

    if method == "NPM":
        if data_class == "HUMAN":
            from configs_viz.config_viz_OURS import prepare_paths
        elif data_class == "MANO":
            from configs_viz.config_viz_OURS_MANO import prepare_paths
    elif method == "SMPL":
        from configs_viz.config_viz_SMPL import prepare_paths
    elif method == "OCCFLOW":
        from configs_viz.config_viz_OCCFLOW import prepare_paths
    elif method == "IFNET":
        from configs_viz.config_viz_IFNET import prepare_paths
    elif method == "IPNET":
        from configs_viz.config_viz_IPNET import prepare_paths
    else:
        raise Exception("Invalid method!")

    # Frame rate
    frame_rate = 30

    # Get method-specific data
    gt_path_list, pred_path_list, video_dir, filter_mesh_flag, frame_rate_method, scale_method, translation_method = prepare_paths()

    if frame_rate is not None:
        frame_rate = frame_rate_method

    input(f"Continue? - frame rate {frame_rate}")

    #######################################################################################

    viewer = ViewerFinal(
        gt_path_list,
        pred_path_list,
        video_dir,
        viz_type,
        filter_mesh_flag,
        view=args.view,
        num_to_eval=args.num_to_eval,
        load_every=None,
        from_frame_id=0,
        frame_rate=frame_rate,
        scale=scale_method,
        translation=translation_method,
    )
    viewer.run()