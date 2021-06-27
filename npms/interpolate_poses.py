import sys,os

from numpy.core.fromnumeric import shape
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import os
import trimesh
import numpy as np
import json
import open3d as o3d
import torch
from tqdm import tqdm
from timeit import default_timer as timer
import utils.deepsdf_utils as deepsdf_utils 
import matplotlib.pyplot as plt

from models.pose_decoder import PoseDecoder, PoseDecoderSE3
from models.shape_decoder import ShapeDecoder

from utils.pcd_utils import (BBox,
                            trimesh_to_open3d,
                            transform_pointcloud_to_opengl_coords,
                            rotate_around_axis,
                            origin, normalize_transformation)

import utils.nnutils as nnutils
import config as cfg
import data_scripts.config_data as cfg_data


class ViewerInterpolatePose:
    def __init__(
        self, 
        labels,
        labels_tpose,
        shape_codes,
        pose_codes,
        out_root_dir,
        num_to_eval=-1,
        view="lateral_left",
        dataset_class=None,
        identity_name=None,
        animation_name=None,
        sample_id_start=None,
        sample_id_end=None,
        num_iterpolations=10,
        use_gt_tpose_mesh=False,
        render_video_options="/rhome/ppalafox/workspace/render_video_options",
        frame_rate=30,
        res_mult=1
    ):
        self.frame_rate = frame_rate

        mult = res_mult
        self.reconstruction_res = 256 * mult
        self.max_batch = (mult * 32)**3

        self.labels = labels
        self.labels_tpose = labels_tpose
        self.num_to_eval = num_to_eval
        
        self.identity_name = identity_name
        self.animation_name = animation_name
        self.sample_id_start = sample_id_start
        self.sample_id_end = sample_id_end

        # Find the identity's id
        for identity_id, label_tpose in enumerate(self.labels_tpose):
            if label_tpose['identity_name'] == self.identity_name:
                self.identity_id = identity_id

        # Find the start and end samples' indices, to then index pose_codes
        self.start_idx, self.end_idx = None, None
        for pose_idx, label in enumerate(self.labels):
            if label['identity_name' ] == self.identity_name and label['animation_name'] == self.animation_name:
                if label['sample_id'] == self.sample_id_start:
                    self.start_idx = pose_idx
                elif label['sample_id'] == self.sample_id_end:
                    self.end_idx = pose_idx

        assert self.start_idx is not None and self.end_idx is not None

        self.num_iterpolations = num_iterpolations

        self.shape_codes = shape_codes
        self.pose_codes = pose_codes

        self.dataset_class = dataset_class

        self.cur = None

        self.show_cur = False

        self.START_COLOR = np.array([0.4, 0.6, 0.4])
        self.END_COLOR   = np.array([0.6, 0.4, 0.4])

        self.time = 0

        self.use_gt_tpose_mesh = use_gt_tpose_mesh

        # Recording options
        self.view = view
        self.render_json            = os.path.join(render_video_options, "render_options.json")
        self.viewpoint_json         = os.path.join(render_video_options, "viewpoint.json")
        self.viewpoint_lateral_json = os.path.join(render_video_options, "viewpoint_lateral.json")
        self.viewpoint_lateral_left_json = os.path.join(render_video_options, "viewpoint_lateral_left.json")
        os.makedirs(render_video_options, exist_ok=True)
        self.out_dir = os.path.join(out_root_dir, f"{self.identity_name}__{self.animation_name}__start_{sample_id_start}__end_{sample_id_end}__@{self.reconstruction_res}__fps{self.frame_rate}")
        os.makedirs(self.out_dir, exist_ok=True)

        self.initialize()


    def initialize(self):

        self.cur_list = []

        self.loaded_frames = 0

        ref_path = os.path.join(data_dir, self.dataset_class, self.identity_name, "a_t_pose", "000000")

        ################################################################################################################
        # Load the T-pose
        ################################################################################################################
        print()
        print("Computing T-Pose...")
        if self.use_gt_tpose_mesh:
            ref_sample_path = os.path.join(ref_path, 'mesh_normalized.ply')
            assert os.path.isfile(ref_sample_path), ref_sample_path
            ref_mesh = trimesh.load(ref_sample_path)
        else:
            ref_mesh = deepsdf_utils.create_mesh(
                shape_decoder, self.shape_codes, identity_ids=[self.identity_id], shape_codes_dim=shape_codes_dim,
                N=self.reconstruction_res, max_batch=self.max_batch
            )
        p_ref = ref_mesh.vertices.astype(np.float32)
        p_ref_cuda = torch.from_numpy(p_ref)[None, :].cuda()
        p_ref_cuda_flat = p_ref_cuda.reshape(-1, 3) # [100000, 3]

        ### src ref (mesh)
        # ref_mesh_o3d = trimesh_to_open3d(ref_mesh, self.CUR_COLOR)
        # o3d.visualization.draw_geometries([ref_mesh_o3d])

        ### Prepare shape codes ###
        shape_codes_batch = self.shape_codes[[self.identity_id], :] # [bs, 1, C]
        assert shape_codes_batch.shape == (1, 1, shape_codes_dim), f"{shape_codes_batch} vs {(1, 1, shape_codes_dim)}"
        # Extent latent code to all sampled points
        shape_codes_repeat = shape_codes_batch.expand(-1, p_ref_cuda_flat.shape[0], -1) # [bs, N, C]
        shape_codes_inputs = shape_codes_repeat.reshape(-1, shape_codes_dim) # [bs*N, C]

        print("T-pose successfully reconstructed!")

        ################################################################################################################
        # Prepare the start and end pose codes
        ################################################################################################################
        pose_code_start = self.pose_codes[[self.start_idx], ...] # [bs, 1, C]
        pose_code_end   = self.pose_codes[[self.end_idx], ...] # [bs, 1, C]

        ################################################################################################################
        ################################################################################################################
        # Go over the different poses
        ################################################################################################################
        ################################################################################################################
        
        for i in range(self.num_iterpolations + 1):

            alpha = i / self.num_iterpolations

            # Prepare interpolated pose code
            pose_code_interp = (1 - alpha) * pose_code_start + alpha * pose_code_end

            with torch.no_grad():
                
                ##########################################################################################
                ### Prepare pose codes
                pose_codes_batch = pose_code_interp
                assert pose_codes_batch.shape == (1, 1, pose_codes_dim), f"{pose_codes_batch.shape} vs {(1, 1, pose_codes_dim)}"

                # Extent latent code to all sampled points
                pose_codes_repeat = pose_codes_batch.expand(-1, p_ref_cuda_flat.shape[0], -1) # [bs, N, C]
                pose_codes_inputs = pose_codes_repeat.reshape(-1, pose_codes_dim) # [bs*N, C]
                ##########################################################################################

                # Concatenate pose and shape codes
                shape_pose_codes_inputs = torch.cat([shape_codes_inputs, pose_codes_inputs], 1)
                
                # Concatenate (for each sample point), the corresponding code and the p_cur coords
                pose_inputs = torch.cat([shape_pose_codes_inputs, p_ref_cuda_flat], 1)

                # Predict delta flow
                p_ref_warped, _ = pose_decoder(pose_inputs) # [bs*N, 3]

            # REFW
            p_ref_warped = p_ref_warped.detach().cpu().numpy()
            ref_warped_mesh_o3d = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(p_ref_warped),
                o3d.utility.Vector3iVector(ref_mesh.faces),
            )
            ref_warped_mesh_o3d.compute_vertex_normals()

            COLOR_INTERP = (1 - alpha) * self.START_COLOR + alpha * self.END_COLOR
            ref_warped_mesh_o3d.paint_uniform_color(COLOR_INTERP)

            if False:
                o3d.visualization.draw_geometries([ref_warped_mesh_o3d])
            
            self.cur_list.append(ref_warped_mesh_o3d)

            # Increase counter of evaluated frames
            self.loaded_frames += 1

            print(f'Loaded {self.loaded_frames} frames')

            if self.loaded_frames == self.num_to_eval:
                print()
                print(f"Stopping early. Already loaded {self.loaded_frames}")
                print()
                break

            # break

        # ###############################################################################################
        # # Generate additional meshes.
        # ###############################################################################################
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


    def update_cur(self, vis):
        param = vis.get_view_control().convert_to_pinhole_camera_parameters()

        if self.cur is not None:
            vis.remove_geometry(self.cur)
        
        if self.show_cur:
            self.cur = self.cur_list[self.time]
            vis.add_geometry(self.cur)

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

        def update_all(vis):
            self.update_cur(vis)
            return False

        # Define callbacks.
        def toggle_next(vis):
            self.time += 1
            if self.time >= self.loaded_frames:
                self.time = 0
            update_all(vis)
            return False
        
        def toggle_previous(vis):
            self.time -= 1
            if self.time < 0:
                self.time = self.loaded_frames - 1
            update_all(vis)
            return False

        def toggle_cur(vis):
            self.show_cur = not self.show_cur
            update_all(vis)
            return False

        def render(vis):
            print("::render")

            self._load_render_and_viewpoint_option(vis, self.view)

            ##################################################
            # Render the interpolated poses
            ##################################################
            self.time = 0
            self.show_cur = True 
            self.cur = self.cur_list[self.time]
            update_all(vis)
            vis.poll_events()
            vis.update_renderer()
            
            for i in range(len(self.cur_list)):
                # Render
                self.render_image(vis, f"interp_{str(i).zfill(2)}")
                toggle_next(vis)
                vis.poll_events()
                vis.update_renderer()
            self.show_src_refw = False

            os.system(f"ffmpeg -r {self.frame_rate} -i {self.out_dir}/interp_%02d.jpg -c:v libx264 -vf fps=30 -pix_fmt yuv420p -y {self.out_dir}/{self.identity_name}__{self.animation_name}__start_{sample_id_start}__end_{sample_id_end}.mp4")
            exit() 

            return False

        def save_viewpoint_lateral_left(vis):
            print("::save_viewpoint")
            param = vis.get_view_control().convert_to_pinhole_camera_parameters()
            o3d.io.write_pinhole_camera_parameters(self.viewpoint_lateral_left_json, param)
            return False

        key_to_callback = {}
        key_to_callback[ord("D")] = toggle_next
        key_to_callback[ord("A")] = toggle_previous
        key_to_callback[ord("T")] = toggle_cur
        key_to_callback[ord("R")] = render
        key_to_callback[ord("'")] = save_viewpoint_lateral_left

        # Add mesh at initial time step.
        assert self.time < self.loaded_frames

        print("Showing time", self.time)

        # Start showing the tgt tpose
        self.cur = self.cur_list[self.time]
        self.show_cur = True

        o3d.visualization.draw_geometries_with_key_callbacks([self.world_frame, self.cur], key_to_callback)


################################################################################################################################
################################################################################################################################


if __name__ == "__main__":

    torch.backends.cudnn.benchmark = True

    out_root_dir = "/cluster_HDD/lothlann/ppalafox/qualitative_results__interpolation_POSE__AMASS"
    
    viz = False

    # ------------------------------------------- #
    DATASET_TYPE = "HUMAN"
    print("#"*30)
    print(f"DATASET_TYPE: {DATASET_TYPE}")
    print("#"*30)
    # input("Continue?")
    # ------------------------------------------- #

    if DATASET_TYPE == "HUMAN":
        from configs_eval.config_eval_HUMAN import *
    elif DATASET_TYPE == "MANO":
        from configs_eval.config_eval_MANO import *

    ########################################################################################################################
    ########################################################################################################################

    data_dir = f"{ROOT}/datasets_mix"

    # Extract dataset name
    tmp = exp_name.split('__ON__')
    dataset_name = tmp[-1]

    from utils.parsing_utils import get_dataset_type_from_dataset_name
    dataset_type = get_dataset_type_from_dataset_name(dataset_name)
    splits_dir = f"{cfg.splits_dir}_{dataset_type}"

    labels_json       = os.path.join(data_dir, splits_dir, dataset_name, "labels.json")
    labels_tpose_json = os.path.join(data_dir, splits_dir, dataset_name, "labels_tpose.json")

    print("Reading from:")
    print(labels_json)
    print("Dataset name:")
    print(dataset_name)
    print()
    
    train_to_augmented_json = os.path.join(data_dir, splits_dir, dataset_name, "train_to_augmented.json")

    #######################################################################################################
    # Data
    #######################################################################################################
    with open(labels_json, "r") as f:
        labels = json.loads(f.read())

    with open(labels_tpose_json, "r") as f:
        labels_tpose = json.loads(f.read())

    train_to_augmented = None
    if os.path.isfile(train_to_augmented_json):
        with open(train_to_augmented_json, "r") as f:
            train_to_augmented = json.loads(f.read())

    batch_size = min(batch_size, len(labels))
    print("batch_size", batch_size)

    print("clamping distance", clamping_distance)

    num_identities = len(labels_tpose)
    num_frames = len(labels)

    print()
    print("#"*60)
    print("Num identities", num_identities)
    print("Num frames    ", num_frames)
    print()
    print('Interval', interval)
    print('Factor', factor)
    print("#"*60)
    print()

    ########################################################################################################################
    ########################################################################################################################

    # Pose MLP
    exp_dir = os.path.join(exps_dir, exp_name)
    checkpoint = nnutils.load_checkpoint(exp_dir, checkpoint_epoch)    

    ########################
    # Shape decoder
    ########################
    shape_decoder = ShapeDecoder(shape_codes_dim, **shape_network_specs).cuda()
    shape_decoder.load_state_dict(checkpoint['model_state_dict_shape_decoder'])
    for param in shape_decoder.parameters():
        param.requires_grad = False
    shape_decoder.eval()
    nnutils.print_num_parameters(shape_decoder)

    ########################
    # Pose decoder
    ########################
    if use_se3:
        print()
        print("Using SE(3) formulation for the PoseDecoder")
        pose_decoder = PoseDecoderSE3(
            pose_codes_dim + shape_codes_dim, **pose_network_specs
        ).cuda()
    else:
        print()
        print("Using normal (translation) formulation for the PoseDecoder")
        pose_decoder = PoseDecoder(
            pose_codes_dim + shape_codes_dim, **pose_network_specs
        ).cuda()
    pose_decoder.load_state_dict(checkpoint['model_state_dict_pose_decoder'])
    for param in pose_decoder.parameters():
        param.requires_grad = False
    pose_decoder.eval()
    nnutils.print_num_parameters(pose_decoder)
    
    ########################
    # SHAPE Codes
    ########################
    shape_codes = torch.ones(num_identities, 1, shape_codes_dim).normal_(0, 0.1).cuda()
    
    pretrained_shape_codes = checkpoint['shape_codes'].cuda().detach().clone()

    if shape_codes.shape[0] != pretrained_shape_codes.shape[0] and train_to_augmented is not None:
        print("Loading shape codes - Since shape_codes and pretrained_shapes_codes dont match in shape, have to compute mapping...")
        shape_codes = pretrained_shape_codes[list(train_to_augmented.values())].detach().clone()
        if len(shape_codes.shape) == 2:
            shape_codes = shape_codes.unsqueeze(0)
    else:
        print("Loading shape codes - Perfect: shapes match between pretrained and current shape_codes")
        shape_codes = checkpoint['shape_codes'].cuda().detach().clone()
    
    ##################################################################
    # Use codes from training
    ##################################################################

    print()
    print("Using pretrained pose codes")
    print()

    pretrained_pose_codes = checkpoint['pose_codes'].cuda().detach().clone()

    if pretrained_pose_codes.shape[0] != len(labels):
        raise Exception("Number of pose codes != lenght of dataset")

    pose_codes = pretrained_pose_codes
    pose_codes.requires_grad = False

    assert pose_codes.shape[1] == 1 and pose_codes.shape[2] == pose_codes_dim

    ##################################################################################################################
    ##################################################################################################################
    print()
    print()
    print("#######################################################################")
    print("Final visualization")
    print("#######################################################################")

    # -----------------------------------------------------------------

    """ Used for paper """
    """
    identity_name   = "kate"
    animation_name  = "hip_hop_just_listening_dancing_variation"
    sample_id_start = str(90).zfill(6)
    sample_id_end   = str(371).zfill(6)

    # identity_name   = "regina"
    # animation_name  = "male_salsa_variation_eight"
    # sample_id_start = str(31).zfill(6)
    # sample_id_end   = str(195).zfill(6)

    # identity_name   = "lewis"
    # animation_name  = "female_samba_pagode_variation_five_loop"
    # sample_id_start = str(303).zfill(6)
    # sample_id_end   = str(419).zfill(6)

    # identity_name   = "adam"
    # animation_name  = "male_salsa_variation_eight"
    # sample_id_start = str(173).zfill(6)
    # sample_id_end   = str(346).zfill(6)
    """

    # identity_name   = "adam"
    # animation_name  = "male_salsa_variation_eight"
    # sample_id_start = str(151).zfill(6)
    # sample_id_end   = str(308).zfill(6)

    # identity_name   = "KIT_s384"
    # animation_name  = "motion013"
    # sample_id_start = str(274).zfill(6)
    # sample_id_end   = str(474).zfill(6)

    identity_name   = "BMLrub_s136"
    animation_name  = "motion001"
    sample_id_start = str(109).zfill(6)
    sample_id_end   = str(909).zfill(6)

    # -----------------------------------------------------------------

    view = "frontal"

    # -----------------------------------------------------------------

    viewer = ViewerInterpolatePose(
        labels,
        labels_tpose,
        shape_codes,
        pose_codes,
        out_root_dir,
        num_to_eval=-1,
        view=view,
        dataset_class="mixamo_trans_all",
        identity_name=identity_name,
        animation_name=animation_name,
        sample_id_start=sample_id_start,
        sample_id_end=sample_id_end,
        num_iterpolations=30,
        frame_rate=30,
        res_mult=1,
    )
    viewer.run()