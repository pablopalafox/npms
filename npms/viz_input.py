from genericpath import exists
import os
import numpy as np
import trimesh
import open3d as o3d
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
from utils.pcd_utils import BBox

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
        self.input   = None
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

        self.input_list   = []

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

            input_dir = os.path.join(self.data_base_dir, label['dataset'], label['identity_name'], label['animation_name'], label['sample_id'])
            
            if not os.path.isdir(input_dir):
                print("-- Skipping", input_dir)
                continue

            print("++ Loading", input_dir)

            ############################################################
            # Load input voxel grid
            ############################################################
            inputs_path = os.path.join(input_dir, f'partial_views/voxelized_view0_{self.res}res.npz')
            inputs_npz = np.load(inputs_path)

            # occupancies = np.unpackbits(np.load(inputs_path)['compressed_occupancies'])
            # input_voxels_np = np.reshape(occupancies, (self.res,)*3).astype(np.float32)
            # voxels_trimesh = VoxelGrid(input_voxels_np).to_mesh()
            # voxels_mesh = o3d.geometry.TriangleMesh(
            #     o3d.utility.Vector3dVector(voxels_trimesh.vertices),
            #     o3d.utility.Vector3iVector(voxels_trimesh.faces)
            # )
            # voxels_mesh.compute_vertex_normals()
            # voxels_mesh.paint_uniform_color([0.2, 1, 0.5])
            # self.input_list.append(voxels_mesh)

            input_points_np = inputs_npz['point_cloud']
            p_partial_cur_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(input_points_np))
            p_partial_cur_pcd.paint_uniform_color([0.3, 0.3, 0.3]) # current is in green
            p_partial_cur_pcd.estimate_normals()
            self.input_list.append(p_partial_cur_pcd)

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
            size=0.00001, origin=[0, 0, 0]
        )

    def update_input(self, vis):
        param = vis.get_view_control().convert_to_pinhole_camera_parameters()

        if self.input is not None:
            vis.remove_geometry(self.input)
        
        self.input = self.input_list[self.time]
        vis.add_geometry(self.input)

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

            self.update_input(vis)
            return False
        
        def toggle_previous(vis):
            self.time -= 1
            if self.time < 0:
                self.time = self.loaded_frames - 1
            
            print(f"time {self.time}")

            self.update_input(vis)
            return False

        def animate_and_record(vis):
            animate(vis, record_video=True)

        def animate(vis, record_video=False):
            print("::animate")

            self._load_render_and_viewpoint_option(vis, self.view)

            # Start at the first frame
            self.time = 0

            num_meshes = len(self.input_list)

            self.update_input(vis)
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

                # Also store the pointclouds
                pcl_dir = os.path.join(no_motion_dir, "pcl")
                os.makedirs(pcl_dir, exist_ok=True)
                o3d.io.write_point_cloud(f"{pcl_dir}/{i:05d}.ply", self.input)

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

        self.input = self.input_list[self.time]

        # o3d.visualization.draw_geometries_with_key_callbacks([self.unit_bbox, self.pred, self.input], key_to_callback)
        o3d.visualization.draw_geometries_with_key_callbacks([self.world_frame, self.input], key_to_callback)


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
    
    #############################

    ####################################
    # Real CAPE scans 
    ####################################
    # dataset_name = "CAPE-POSE-TEST-00032_shortlong-hips-1id-293ts-1seqs"
    # dataset_name = "CAPE-POSE-TEST-00032_shortshort-shoulders_mill-1id-207ts-1seqs"
    # dataset_name = "CAPE-POSE-TEST-03223_shortlong-shoulders_mill-1id-378ts-1seqs"
    # dataset_name = "CAPE-POSE-TEST-03223_shortshort-tilt_twist_left-1id-349ts-1seqs"

    # dataset_name = "CAPE-POSE-TEST-00032_shortlong-tilt_twist_left-1id-100ts-1seqs-50to149"
    # dataset_name = "CAPE-POSE-TEST-00032_shortlong-tilt_twist_left-1id-100ts-1seqs-170to269"
    # dataset_name = "CAPE-POSE-TEST-03223_shortlong-tilt_twist_left-1id-100ts-1seqs-100to259"
    # dataset_name = "CAPE-POSE-TEST-03223_shortshort-tilt_twist_left-1id-300ts-1seqs-50to349"

    ####################################
    # MIXAMO
    ####################################
    # dataset_name = "MIXAMO_TRANS_ALL-POSE-TEST-alien-female_hip_hop_slide_step_dancing-1id-108ts-1seqs"
    # dataset_name = "MIXAMO_TRANS_ALL-POSE-TEST-joe-breakdance_brooklyn_style_uprocking-1id-100ts-1seqs"
    # dataset_name = "MIXAMO_TRANS_ALL-POSE-TEST-olivia-female_samba_ijexa_break-1id-300ts-1seqs"
    # dataset_name = "MIXAMO-POSE-TEST-sophie-female_salsa_dancing-test-1id-90ts-1seqs"

    ############### slowdown
    # dataset_name = "MIXAMO_TRANS_ALL-POSE-TEST-olivia-female_samba_ijexa_break_sx2-1id-260ts-1seqs"
    # dataset_name = "MIXAMO_TRANS_ALL-POSE-TEST-alien-female_hip_hop_slide_step_dancing_sx8-1id-250ts-1seqs"
    # dataset_name = "MIXAMO_TRANS_ALL-POSE-TEST-joe-breakdance_brooklyn_style_uprocking_sx8-1id-240ts-1seqs"
    # dataset_name = "MIXAMO-POSE-TEST-sophie-female_salsa_dancing_sx3-1id-268ts-1seqs"

    ####################################
    # MANO
    ####################################
    # dataset_name = "MANO-POSE-TEST-test_id_39_26l_mirrored-1id-100ts-1seqs"
    # dataset_name = "MANO-POSE-TEST-test_id_10_45l_mirrored-1id-100ts-1seqs"
    # dataset_name = "MANO-POSE-TEST-test_id_24_38l_mirrored-1id-100ts-1seqs"
    # dataset_name = "MANO-POSE-TEST-test_id_30_42l_mirrored-1id-100ts-1seqs"
    # dataset_name = "MANO-POSE-TEST-test_id_50_43r-1id-100ts-1seqs"

    ####################################
    # DFAUST
    ####################################
    # dataset_name = "DFAUST-POSE-TEST-50021-chicken_wings-1id-17ts-1seqs"

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
    video_dir = f"/cluster_HDD/lothlann/ppalafox/videos/{dataset_name}/input"

    num_to_eval = args.num_to_eval

    assert args.view == "frontal" or args.view == "lateral", args.view 

    # Frame rate
    frame_rate = 30
    # if "MIXAMO" in video_dir and 'olivia' not in video_dir:
    #     frame_rate = 15
    if "DFAUST" in video_dir:
        frame_rate = 5

    input(f"Continue? - frame rate {frame_rate}")

    viewer = ViewerFinal(
        labels, data_base_dir,
        video_dir,
        view=args.view,
        num_to_eval=num_to_eval,
        load_every=None,
        from_frame_id=0,
        frame_rate=frame_rate
    )
    viewer.run()