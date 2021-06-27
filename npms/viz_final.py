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
        exp_dir,
        video_dir,
        view,
        num_to_eval=-1,
        load_every=None,
        from_frame_id=0,
        load_scan=False,
        load_voxel_input=False,
        res=256,
        compute_metrics=False,
        # recording options:
        record_directly=False,
        frame_rate=30,
        render_video_options="/rhome/ppalafox/workspace/render_video_options",
    ):
        self.labels = labels
        self.data_base_dir = data_base_dir
        self.exp_dir = exp_dir

        self.num_to_eval = num_to_eval
        self.load_every = load_every
        self.from_frame_id = from_frame_id

        self.load_scan = load_scan
        self.load_voxel_input = load_voxel_input

        self.time = 0

        self.gt_tpose = None
        self.pred_tpose = None
        
        self.pred = None
        self.gt   = None
        self.scan = None
        self.vox  = None
        self.pcd  = None

        self.show_gt_tpose = False
        self.show_pred_tpose = False

        self.show_pred = False
        self.show_gt   = False
        self.show_scan = False
        self.show_vox  = False
        self.show_pcd  = False

        self.res = res

        self.compute_metrics = compute_metrics

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

        self.pred_list = []
        self.gt_list   = []
        self.scan_list = []
        self.vox_list  = []
        self.pcd_list  = []

        self.loaded_frames = 0

        num_gt_meshes = len(labels)

        iou = []
        chamfer = []
        normal_consis = []

        ########################################################################################
        ########################################################################################
        # Load T-Pose
        ########################################################################################
        ########################################################################################
        
        ############################################################
        # Load groundtruth tpose
        ############################################################
        label_tpose = labels[0]
        gt_dir = os.path.join(self.data_base_dir, label_tpose['dataset'], label_tpose['identity_name'], 'a_t_pose', '000000')
        assert os.path.isdir(gt_dir), gt_dir
        
        gt_tpose_mesh_path = os.path.join(gt_dir, "mesh_normalized.ply")
        gt_tpose_mesh_o3d = o3d.io.read_triangle_mesh(gt_tpose_mesh_path)
        gt_tpose_mesh_o3d.paint_uniform_color([0.3, 0.3, 0.3])
        gt_tpose_mesh_o3d.compute_vertex_normals()
        self.gt_tpose = gt_tpose_mesh_o3d

        ############################################################
        # Load reconstructed tpose
        ############################################################
        tpose_frame_dir = os.path.join(self.exp_dir, 'a_t_pose')
        if os.path.isdir(tpose_frame_dir):
            pred_tpose_mesh_path = os.path.join(tpose_frame_dir, "ref_reconstructed.ply")
            pred_tpose_mesh_o3d = o3d.io.read_triangle_mesh(pred_tpose_mesh_path)
            pred_tpose_mesh_o3d.paint_uniform_color([0.6, 0.6, 0.6])
            pred_tpose_mesh_o3d.compute_vertex_normals()
            self.pred_tpose = pred_tpose_mesh_o3d

        ########################################################################################
        ########################################################################################
        # Go over all frames in the sequence
        ########################################################################################
        ########################################################################################
        self.loaded_frames = 0

        # Vertex colors
        vertex_colors = None
        
        for frame_t, label in enumerate(tqdm(labels)):
            
            label = labels[frame_t]

            gt_dir = os.path.join(self.data_base_dir, label['dataset'], label['identity_name'], label['animation_name'], label['sample_id'])
            
            if not os.path.isdir(gt_dir):
                print("-- Skipping", gt_dir)
                continue

            print("++ Loading", gt_dir)

            ############################################################
            # Load groundtruth mesh
            ############################################################
            gt_mesh_path = os.path.join(gt_dir, "mesh_normalized.ply")
            if self.compute_metrics:
                gt_mesh_trimesh = trimesh.load(gt_mesh_path, process=False)
            gt_mesh_o3d = o3d.io.read_triangle_mesh(gt_mesh_path)
            gt_mesh_o3d.paint_uniform_color([0, 0.4, 0])
            gt_mesh_o3d.compute_vertex_normals()
            self.gt_list.append(gt_mesh_o3d)

            ############################################################
            # Load real scan (if it exists)
            ############################################################
            if self.load_scan:
                print("Loading scan")
                scan_mesh_path = os.path.join(gt_dir, "mesh_real_scan.ply")
                if os.path.isfile(scan_mesh_path):
                    scan_mesh_o3d = o3d.io.read_triangle_mesh(scan_mesh_path)
                    scan_mesh_o3d.paint_uniform_color([0, 0.5, 0.5])
                    scan_mesh_o3d.compute_vertex_normals()
                    self.scan_list.append(scan_mesh_o3d)
                else:
                    print("no real scan")

            ############################################################
            # Load input voxel grid
            ############################################################
            if self.load_voxel_input:
                inputs_path = os.path.join(gt_dir, f'partial_views/voxelized_view0_{self.res}res.npz')
                occupancies = np.unpackbits(np.load(inputs_path)['compressed_occupancies'])
                input_voxels_np = np.reshape(occupancies, (self.res,)*3).astype(np.float32)
                voxels_trimesh = VoxelGrid(input_voxels_np).to_mesh()
                voxels_mesh = o3d.geometry.TriangleMesh(
                    o3d.utility.Vector3dVector(voxels_trimesh.vertices),
                    o3d.utility.Vector3iVector(voxels_trimesh.faces)
                )
                voxels_mesh.compute_vertex_normals()
                voxels_mesh.paint_uniform_color([0.2, 1, 0.5])
                self.vox_list.append(voxels_mesh)

                try:
                    input_points_np = np.load(inputs_path)['point_cloud']
                    p_partial_cur_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(input_points_np))
                    p_partial_cur_pcd.paint_uniform_color([0.5, 1, 0.2]) # current is in green
                    p_partial_cur_pcd.estimate_normals()
                    self.pcd_list.append(p_partial_cur_pcd)
                except:
                    pass

            ############################################################
            # Load predicted mesh
            ############################################################
            frame_dir = os.path.join(self.exp_dir, label['sample_id'])
            assert os.path.isdir(frame_dir), frame_dir
            pred_mesh_path = os.path.join(frame_dir, "ref_warped.ply")
            if self.compute_metrics:
                pred_mesh_trimesh = trimesh.load(pred_mesh_path, process=False)
            pred_mesh_o3d = o3d.io.read_triangle_mesh(pred_mesh_path)
            pred_mesh_o3d.compute_vertex_normals()

            # Initialize vertex colors if necessary
            if vertex_colors is None:
                bbox_min = -0.5
                bbox_max = 0.5
                half_box = (bbox_max - bbox_min) / 2.0 
                scale = 1.0
                vertex_colors = (scale * np.array(pred_mesh_o3d.vertices) + half_box) / (2.0 * half_box)
                vertex_colors = np.clip(vertex_colors, 0.0, 1.0)
                
            pred_mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
            self.pred_list.append(pred_mesh_o3d)

            ############################################################
            # Compute metrics
            ############################################################
            if self.compute_metrics:
                eval_res = eval_mesh(gt_mesh_trimesh, mesh_trimesh, -0.5, 0.5)
                    
                iou.append(eval_res['iou'])
                chamfer.append(eval_res['chamfer_l2'])
                normal_consis.append(eval_res['normals'])

            # ------------------------------------------------------- #

            # Increase counter of evaluated frames
            self.loaded_frames += 1

            print(f'Loaded {self.loaded_frames} frames')

            if self.loaded_frames == self.num_to_eval:
                print()
                print(f"Stopping early. Already loaded {self.loaded_frames}")
                print()
                break

        if self.compute_metrics:
            # Compute average metrics.
            iou_avg           = np.mean(np.array(iou))
            chamfer_avg       = np.mean(np.array(chamfer))
            normal_consis_avg = np.mean(np.array(normal_consis))

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
            size=0.01, origin=[0, 0, 0]
        )

    def update_pred_tpose(self, vis):
        param = vis.get_view_control().convert_to_pinhole_camera_parameters()

        # We remove a mesh if it's currently stored.
        if self.pred_tpose is not None:
            vis.remove_geometry(self.pred_tpose)

        # If requested, we show a (new) mesh.
        if self.show_pred_tpose and self.pred_tpose is not None:
            vis.add_geometry(self.pred_tpose)

        ctr = vis.get_view_control()
        ctr.convert_from_pinhole_camera_parameters(param)

    def update_gt_tpose(self, vis):
        param = vis.get_view_control().convert_to_pinhole_camera_parameters()

        if self.gt_tpose is not None:
            vis.remove_geometry(self.gt_tpose)
        
        if self.show_gt_tpose and self.gt_tpose is not None:
            vis.add_geometry(self.gt_tpose)

        ctr = vis.get_view_control()
        ctr.convert_from_pinhole_camera_parameters(param)

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

    def update_scan(self, vis):
        param = vis.get_view_control().convert_to_pinhole_camera_parameters()

        if self.scan is not None:
            vis.remove_geometry(self.scan)
        
        if self.show_scan and len(self.scan_list) > 0:
            self.scan = self.scan_list[self.time]
            vis.add_geometry(self.scan)

        ctr = vis.get_view_control()
        ctr.convert_from_pinhole_camera_parameters(param)

    def update_vox(self, vis):
        param = vis.get_view_control().convert_to_pinhole_camera_parameters()

        if self.vox is not None:
            vis.remove_geometry(self.vox)
        
        if self.show_vox and len(self.vox_list) > 0:
            self.vox = self.vox_list[self.time]
            vis.add_geometry(self.vox)

        ctr = vis.get_view_control()
        ctr.convert_from_pinhole_camera_parameters(param)

    def update_pcd(self, vis):
        param = vis.get_view_control().convert_to_pinhole_camera_parameters()

        if self.pcd is not None:
            vis.remove_geometry(self.pcd)
        
        if self.show_pcd and len(self.pcd_list) > 0:
            self.pcd = self.pcd_list[self.time]
            vis.add_geometry(self.pcd)

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

            self.update_gt_tpose(vis)
            self.update_pred_tpose(vis)
            self.update_gt(vis)
            self.update_pred(vis)
            self.update_scan(vis)
            self.update_vox(vis)
            self.update_pcd(vis)
            return False
        
        def toggle_previous(vis):
            self.time -= 1
            if self.time < 0:
                self.time = self.loaded_frames - 1
            
            print(f"time {self.time}")

            self.update_gt_tpose(vis)
            self.update_pred_tpose(vis)
            self.update_gt(vis)
            self.update_pred(vis)
            self.update_scan(vis)
            self.update_vox(vis)
            self.update_pcd(vis)
            return False

        def toggle_gt_tpose(vis):
            self.show_gt_tpose = not self.show_gt_tpose
            self.update_gt_tpose(vis)
            return False

        def toggle_pred_tpose(vis):
            self.show_pred_tpose = not self.show_pred_tpose
            self.update_pred_tpose(vis)
            return False

        def toggle_gt(vis):
            self.show_gt = not self.show_gt
            self.update_gt(vis)
            return False

        def toggle_scan(vis):
            self.show_scan = not self.show_scan
            self.update_scan(vis)
            return False
        
        def toggle_pred(vis):
            self.show_pred = not self.show_pred
            self.update_pred(vis)
            return False

        def toggle_vox(vis):
            self.show_vox = not self.show_vox
            self.update_vox(vis)
            return False

        def toggle_pcd(vis):
            self.show_pcd = not self.show_pcd
            self.update_pcd(vis)
            return False

        def take_screenshot_of_current_scene(vis):
            print("::taking screenshot")
            self._load_render_and_viewpoint_option(vis, self.view)

            vis.poll_events()
            vis.update_renderer()

            no_motion_dir = os.path.join(self.video_dir, f"animate_{self.view}_screenshot_current")
            images_dir = os.path.join(no_motion_dir, "images")
            os.makedirs(images_dir, exist_ok=True)

            image_np = np.asarray(vis.capture_screen_float_buffer(False))
            h, w, _ = image_np.shape
            new_h, new_w = 1200, 1600
            image_np = image_np[(h-new_h)//2:(h+new_h)//2, (w-new_w)//2:(w+new_w)//2,:]
            plt.imsave(f"{images_dir}/screenshot_current_scene.jpg", image_np)

            return False

        def animate_and_record(vis):
            animate(vis, record_video=True)

        def animate(vis, record_video=False):
            print("::animate")

            self._load_render_and_viewpoint_option(vis, self.view)

            # Start at the first frame
            self.time = 0

            num_meshes = len(self.pred_list)

            self.update_pred(vis)
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
        key_to_callback[ord("T")] = toggle_gt_tpose
        key_to_callback[ord("R")] = toggle_pred_tpose
        key_to_callback[ord("C")] = toggle_gt
        key_to_callback[ord("E")] = toggle_scan
        key_to_callback[ord("W")] = toggle_pred
        key_to_callback[ord("P")] = toggle_pcd
        key_to_callback[ord("V")] = toggle_vox

        key_to_callback[ord("N")] = animate
        key_to_callback[ord("M")] = animate_and_record
        key_to_callback[ord(",")] = take_screenshot_of_current_scene
        key_to_callback[ord("/")] = stop_animation
        key_to_callback[ord("[")] = save_viewpoint
        key_to_callback[ord("]")] = load_render_and_viewpoint_option
        key_to_callback[ord("'")] = save_viewpoint_lateral
        key_to_callback[ord("\\")] = load_render_and_viewpoint_lateral_option

        # Add mesh at initial time step.
        assert self.time < self.loaded_frames

        print("Showing time", self.time)

        self.pred = self.pred_list[self.time]
        self.show_pred = True 

        # self.gt = self.gt_list[self.time]
        # self.show_gt = True

        # o3d.visualization.draw_geometries_with_key_callbacks([self.unit_bbox, self.pred, self.gt], key_to_callback)
        o3d.visualization.draw_geometries_with_key_callbacks([self.world_frame, self.pred], key_to_callback)


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description="Viz")
    parser.add_argument('-n', '--num_to_eval', type=int, default=-1)
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
    unit_bbox = rotate_around_axis(unit_bbox, axis_name="x", angle=-np.pi) 
    
    #############################

    load_scan = False
    load_voxel_input = False

    #############################
    # Experiment
    #############################

    # ---------------------------------------------
    # ---------------------------------------------
    
    exp_name = "2021-02-10__NESHPOD__nss0.7_uni0.3__bs4__lr-0.0005-0.0005-0.001-0.001__s128-256__p256-512__wSE3__wShapePosEnc__wPosePosEnc__ON__MIX-POSE__AMASS-50id-10349__MIXAMO-165id-40000__CAPE-35id-20533"

    ##################
    # CAPE
    ##################
    ### Test 1 - "CAPE-POSE-TEST-00032_shortlong-hips-1id-293ts-1seqs"
    # run_name = "2021-02-15__CAPE-POSE-TEST-00032_shortlong-hips-1id-293ts-1seqs__partial__icp0.0005-500__iters1000__sreg0.1_preg0.0001__slr0.0005_plr0.001__interval500_factor0.5__clamp0.1__sigma0.015__tmpreg100__cptFromTrain92"

    ### Test 2 - "CAPE-POSE-TEST-00032_shortshort-shoulders_mill-1id-207ts-1seqs"
    run_name = "2021-02-15__CAPE-POSE-TEST-00032_shortshort-shoulders_mill-1id-207ts-1seqs__partial__icp0.0005-500__iters1000__sreg0.1_preg0.0001__slr0.0005_plr0.001__interval500_factor0.5__clamp0.1__sigma0.015__tmpreg100__cptFromTrain92"
    
    ### Test 3 - "CAPE-POSE-TEST-03223_shortlong-shoulders_mill-1id-378ts-1seqs"
    # run_name = "2021-02-15__CAPE-POSE-TEST-03223_shortlong-shoulders_mill-1id-378ts-1seqs__partial__icp0.0005-500__iters1000__sreg0.1_preg0.0001__slr0.0005_plr0.001__interval500_factor0.5__clamp0.1__sigma0.015__tmpreg100__cptFromTrain92"

    ### Test 4 - "CAPE-POSE-TEST-03223_shortshort-tilt_twist_left-1id-349ts-1seqs"
    # run_name = "2021-02-15__CAPE-POSE-TEST-03223_shortshort-tilt_twist_left-1id-349ts-1seqs__partial__icp0.0005-500__iters1000__sreg0.1_preg0.0001__slr0.0005_plr0.001__interval500_factor0.5__clamp0.1__sigma0.015__tmpreg100__cptFromTrain92"

    ##################
    # MIXAMO
    ##################
    ### Test 1 - "MIXAMO_TRANS_ALL-POSE-TEST-olivia-female_samba_ijexa_break-1id-300ts-1seqs"
    # run_name = "2021-02-15__MIXAMO_TRANS_ALL-POSE-TEST-olivia-female_samba_ijexa_break-1id-300ts-1seqs__partial__icp0.0005-500__iters1000__sreg0.1_preg0.0001__slr0.0005_plr0.001__interval500_factor0.5__clamp0.1__sigma0.015__tmpreg100__cptFromTrain92"

    ### Test 2 - "MIXAMO_TRANS_ALL-POSE-TEST-alien-female_hip_hop_slide_step_dancing-1id-108ts-1seqs"
    # run_name = "2021-02-15__MIXAMO_TRANS_ALL-POSE-TEST-alien-female_hip_hop_slide_step_dancing-1id-108ts-1seqs__partial__icp0.0005-500__iters1000__sreg0.1_preg0.0001__slr0.0005_plr0.001__interval500_factor0.5__clamp0.1__sigma0.015__tmpreg100__cptFromTrain92"

    ### Test 3 - "MIXAMO_TRANS_ALL-POSE-TEST-joe-breakdance_brooklyn_style_uprocking-1id-100ts-1seqs"
    # run_name = "2021-02-15__MIXAMO_TRANS_ALL-POSE-TEST-joe-breakdance_brooklyn_style_uprocking-1id-100ts-1seqs__partial__icp0.0005-500__iters1000__sreg0.1_preg0.0001__slr0.0005_plr0.001__interval500_factor0.5__clamp0.1__sigma0.015__tmpreg100__cptFromTrain92"

    ### Test 4 - "MIXAMO-POSE-TEST-sophie-female_salsa_dancing-test-1id-90ts-1seqs"
    # run_name = "2021-02-15__MIXAMO-POSE-TEST-sophie-female_salsa_dancing-test-1id-90ts-1seqs__partial__icp0.0005-500__iters1000__sreg0.1_preg0.0001__slr0.0005_plr0.001__interval500_factor0.5__clamp0.1__sigma0.015__tmpreg100__cptFromTrain92"
    

    ##################
    # DFAUST
    ##################
    # run_name = "2021-02-18__DFAUST-POSE-TEST-50021-chicken_wings-1id-17ts-1seqs__icp0.0005-500__iters1000__sreg0.1_preg0.0001__slr0.0005_plr0.001__interval500_factor0.5__clamp0.1__sigma0.015__tmpreg100__cptFromTrain92__woALT__woClampRed__part"
    # run_name = "2021-02-18__DFAUST-POSE-TEST-50021-chicken_wings-1id-17ts-1seqs__icp0.0005-500__iters1000__sreg0.1_preg0.0001__slr0.0005_plr0.001__interval500_factor0.5__clamp0.1__sigma0.015__tmpreg100__cptFromTrain92__woALT__woClampRed__comp"

    # run_name = "2021-02-19__DFAUST-POSE-TEST-50021-knees-1id-17ts-1seqs__icp0.0005-500__iters1000__sreg0.1_preg0.0001__slr0.0005_plr0.001__interval500_factor0.5__clamp0.1__sigma0.015__tmpreg100__cptFromTrain92__woALT__woClampRed__part"

    # BAD
    # run_name = "2021-02-19__DFAUST-POSE-TEST-50021-one_leg_jump-1id-17ts-1seqs__icp0.0005-500__iters1000__sreg0.1_preg0.0001__slr0.0005_plr0.001__interval500_factor0.5__clamp0.1__sigma0.015__tmpreg100__cptFromTrain92__woALT__woClampRed__part"
    # run_name = "2021-02-19__DFAUST-POSE-TEST-50021-one_leg_jump-1id-17ts-1seqs__icp0.01-500__iters1000__sreg0.1_preg0.0001__slr0.0005_plr0.001__interval500_factor0.5__clamp0.1__sigma0.015__tmpreg100__cptFromTrain92__woALT__woClampRed__part"
    # run_name = "2021-02-19__DFAUST-POSE-TEST-50021-one_leg_jump-1id-17ts-1seqs__icp0.0005-500__iters1000__sreg0.1_preg0.0001__slr0.0005_plr0.001__interval500_factor0.5__clamp0.1__sigma0.015__tmpreg100__cptFromTrain92__woALT__woClampRed__comp"
    
    # run_name = "2021-02-19__DFAUST-POSE-TEST-50021-one_leg_loose-1id-17ts-1seqs__icp0.0001-500__iters1000__sreg0.1_preg0.0001__slr0.0005_plr0.001__interval500_factor0.5__clamp0.1__sigma0.015__tmpreg100__cptFromTrain92__woALT__woClampRed__part"

    # run_name = "2021-02-19__DFAUST-POSE-TEST-50021-running_on_spot-1id-17ts-1seqs__icp0.0001-500__iters1000__sreg0.1_preg0.0001__slr0.0005_plr0.001__interval500_factor0.5__clamp0.1__sigma0.015__tmpreg100__cptFromTrain92__woALT__woClampRed__part"

    # ---------------------------------------------
    # ---------------------------------------------

    exp_name = "2021-03-15__NESHPOD__bs4__lr-0.0005-0.0005-0.001-0.001_intvl30__s256-512-8l__p256-1024-8l__woSE3__wShapePosEnc__wPosePosEnc__woDroutS__woDroutP__wWNormS__wWNormP__ON__MIX-POSE__AMASS-50id-5000__MIXAMO-165id-20000__CAPE-35id-20533"

    # run_name = "2021-03-23__CAPE-POSE-TEST-00032_shortlong-tilt_twist_left-1id-100ts-1seqs-50to149__bs4__icp0.0005-500__itrs1000__sreg0.1_preg0.0001__slr0.0005_plr0.001__interv250_factr0.5__clamp0.1__sigma0.015__tmpreg100__cpt128__woALT__part"
    run_name = "2021-03-22__CAPE-POSE-TEST-00032_shortshort-shoulders_mill-1id-207ts-1seqs__bs4__icp0.0005-500__itrs1000__sreg0.1_preg0.0001__slr0.0005_plr0.001__interv250_factr0.5__clamp0.1__sigma0.015__tmpreg100__cpt128__woALT__part"
    # run_name = "2021-03-22__CAPE-POSE-TEST-03223_shortshort-tilt_twist_left-1id-300ts-1seqs-50to349__bs4__icp0.0005-500__itrs1000__sreg0.1_preg0.0001__slr0.0005_plr0.001__interv250_factr0.5__clamp0.1__sigma0.015__tmpreg100__cpt128__woALT__part"

    # ---------------------------------------------
    # ---------------------------------------------


    # Extract dataset name
    tmp = run_name.split('__')
    dataset_name = tmp[1]

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

    exp_version = "sdf"
    exps_dir  = os.path.join(cfg.exp_dir, exp_version)
    exp_dir = os.path.join(exps_dir, exp_name, "optimization", run_name)
    predicted_meshes_dir_list = os.path.join(exp_dir, f"predicted_meshes*")
    print("Which one do you want?")
    predicted_meshes_dir = None
    for tmp in sorted(glob.glob(predicted_meshes_dir_list)):
        answer = query_yes_no(tmp, default="no")
        if answer:
            predicted_meshes_dir = tmp
            break
    assert predicted_meshes_dir is not None, "Please select a folder to read from!"
    assert os.path.isdir(predicted_meshes_dir), predicted_meshes_dir

    assert dataset_name in run_name, "Make sure the dataset_name matches that on which we optimized over!"

    # Video dir
    video_dir = f"/cluster_HDD/lothlann/ppalafox/videos/{dataset_name}/ours/{exp_name}/{run_name}"

    num_to_eval = args.num_to_eval

    assert args.view == "frontal" or args.view == "lateral", args.view 
    input(args.view)

    viewer = ViewerFinal(
        labels, data_base_dir,
        predicted_meshes_dir,
        video_dir,
        view=args.view,
        num_to_eval=num_to_eval,
        load_every=None,
        from_frame_id=0,
        load_scan=load_scan,
        load_voxel_input=load_voxel_input,
    )
    viewer.run()

    # print()
    # print("#"*60)
    # print("iou_avg:           {}".format(iou_avg))
    # print("chamfer_avg:       {}".format(chamfer_avg))
    # print("normal_consis_avg: {}".format(normal_consis_avg))
    # print("epe3d_avg:         {}".format(epe3d_avg))
    # print("#"*60)