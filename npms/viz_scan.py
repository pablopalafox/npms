from genericpath import exists
import os
import numpy as np
import trimesh
import open3d as o3d
from tqdm import tqdm
import matplotlib.pyplot as plt
import shutil
import json
import glob
from utils.utils import query_yes_no

from utils.voxels import VoxelGrid
from utils.evaluation import eval_mesh

from utils.pcd_utils import (BBox,
                            rotate_around_axis)

import config as cfg


class ViewerFinal:
    def __init__(
        self, 
        labels, data_base_dir,
        video_dir,
        view,
        num_to_eval=-1,
        load_every=None,
        from_frame_id=0,
        res=256,
        compute_metrics=False,
        # recording options:
        record_directly=False,
        frame_rate=30,
        render_video_options="/rhome/ppalafox/workspace/render_video_options",
    ):
        self.labels = labels
        self.data_base_dir = data_base_dir

        self.num_to_eval = num_to_eval
        self.load_every = load_every
        self.from_frame_id = from_frame_id

        self.time = 0
        self.scan   = None
        self.res = res

        # Recording options
        self.view = view
        self.record_directly = record_directly
        self.render_json            = os.path.join(render_video_options, "render_options.json")
        self.viewpoint_json         = os.path.join(render_video_options, "viewpoint.json")
        self.viewpoint_lateral_json = os.path.join(render_video_options, "viewpoint_lateral.json")
        os.makedirs(render_video_options, exist_ok=True)
        self.frame_rate = frame_rate
        self.video_dir = video_dir
        os.makedirs(self.video_dir, exist_ok=True)

        self.animating = False
        self.stop_animation = False # Flag to cancel current animation
        self.num_circles = 1 # Hardcoded to do only one circle when camera_motion="rotation"

        self.initialize()
    
    def initialize(self):

        self.scan_list   = []

        ########################################################################################
        ########################################################################################
        # Go over all frames in the sequence
        ########################################################################################
        ########################################################################################
        self.loaded_frames = 0

        for frame_t, label in enumerate(tqdm(labels)):
            
            label = labels[frame_t]

            scan_dir = os.path.join(self.data_base_dir, label['dataset'], label['identity_name'], label['animation_name'], label['sample_id'])
            
            if not os.path.isdir(scan_dir):
                print("-- Skipping", scan_dir)
                continue

            print("++ Loading", scan_dir)

            ############################################################
            # Load real scan (if it exists)
            ############################################################
            print("Loading scan")
            scan_mesh_path = os.path.join(scan_dir, "mesh_real_scan.ply")
            if os.path.isfile(scan_mesh_path):
                scan_mesh_o3d = o3d.io.read_triangle_mesh(scan_mesh_path)
                scan_mesh_o3d.paint_uniform_color(SCAN_COLOR)
                scan_mesh_o3d.compute_vertex_normals()
                self.scan_list.append(scan_mesh_o3d)
            else:
                print("no real scan")

            # Increase counter of evaluated frames
            self.loaded_frames += 1

            print(f'Loaded {self.loaded_frames} frames out of {self.num_to_eval}')

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
            size=0.000001, origin=[0, 0, 0]
        )

    def update_scan(self, vis):
        param = vis.get_view_control().convert_to_pinhole_camera_parameters()

        if self.scan is not None:
            vis.remove_geometry(self.scan)
        
        self.scan = self.scan_list[self.time]
        vis.add_geometry(self.scan)

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

            self.update_scan(vis)
            return False
        
        def toggle_previous(vis):
            self.time -= 1
            if self.time < 0:
                self.time = self.loaded_frames - 1
            
            print(f"time {self.time}")

            self.update_scan(vis)
            return False

        def animate_and_record(vis):
            animate(vis, record_video=True)

        def animate(vis, record_video=False):
            print("::animate")

            self._load_render_and_viewpoint_option(vis, self.view)

            # Start at the first frame
            self.time = 0

            num_meshes = len(self.scan_list)

            self.update_scan(vis)
            vis.poll_events()
            vis.update_renderer()

            self.animating = True
            self.stop_animation = False

            if record_video:
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

        key_to_callback[ord("N")] = animate
        key_to_callback[ord("M")] = animate_and_record
        key_to_callback[ord("/")] = stop_animation
        key_to_callback[ord("[")] = save_viewpoint
        key_to_callback[ord("]")] = load_render_and_viewpoint_option
        key_to_callback[ord("'")] = save_viewpoint_lateral
        key_to_callback[ord("\\")] = load_render_and_viewpoint_lateral_option

        # Add mesh at initial time step.
        assert self.time < self.loaded_frames

        print("Showing time", self.time)

        self.scan = self.scan_list[self.time]

        # o3d.visualization.draw_geometries_with_key_callbacks([self.unit_bbox, self.pred, self.scan], key_to_callback)
        o3d.visualization.draw_geometries_with_key_callbacks([self.world_frame, self.scan], key_to_callback)


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description="Viz")
    parser.add_argument('-n', '--num_to_eval', default=-1, type=int)
    parser.add_argument('-v', '--view', required=True)
    args = parser.parse_args()

    debug = False
    if debug: input("Really want to debug?")

    #############################
    
    # unit bbox
    p_min = -0.5
    p_max =  0.5
    unit_bbox = BBox.compute_bbox_from_min_point_and_max_point(
        np.array([p_min]*3), np.array([p_max]*3)
    )

    SCAN_COLOR = [0.6, 0.7, 0.6]
    
    #############################

    ####################################
    # Real CAPE scans 
    ####################################
    # dataset_name = "CAPE-POSE-TEST-00032_shortlong-hips-1id-293ts-1seqs"
    # dataset_name = "CAPE-POSE-TEST-00032_shortshort-shoulders_mill-1id-207ts-1seqs"
    # dataset_name = "CAPE-POSE-TEST-03223_shortlong-shoulders_mill-1id-378ts-1seqs"
    # dataset_name = "CAPE-POSE-TEST-03223_shortshort-tilt_twist_left-1id-349ts-1seqs"

    dataset_name = "CAPE-POSE-TEST-00032_shortlong-tilt_twist_left-1id-100ts-1seqs-50to149"
    # dataset_name = "CAPE-POSE-TEST-00032_shortlong-tilt_twist_left-1id-100ts-1seqs-170to269"
    # dataset_name = "CAPE-POSE-TEST-03223_shortlong-tilt_twist_left-1id-100ts-1seqs-100to259"
    # dataset_name = "CAPE-POSE-TEST-03223_shortshort-tilt_twist_left-1id-300ts-1seqs-50to349"

    #############################
    # Groundtruth data    
    #############################
    from utils.parsing_utils import get_dataset_type_from_dataset_name
    dataset_type = get_dataset_type_from_dataset_name(dataset_name)
    splits_dir = f"{cfg.splits_dir}_{dataset_type}"

    data_base_dir = "/cluster/lothlann/ppalafox/datasets"
    data_dir = f"{data_base_dir}/{splits_dir}/{dataset_name}"
    assert os.path.isdir(data_dir), data_dir

    with open(os.path.join(data_dir, "labels.json"), 'r') as f:
        labels = json.loads(f.read())

    # Video dir
    video_dir = f"/cluster_HDD/lothlann/ppalafox/videos/{dataset_name}/scan"

    num_to_eval = args.num_to_eval
    print("Evaluating", num_to_eval)

    assert args.view == "frontal" or args.view == "lateral", args.view 
    input(args.view)

    viewer = ViewerFinal(
        labels, data_base_dir,
        video_dir,
        view=args.view,
        num_to_eval=num_to_eval,
        load_every=None,
        from_frame_id=0,
    )
    viewer.run()