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


class Viewer:
    def __init__(
        self, 
        labels,
        labels_tpose,
        shape_codes,
        pose_codes,
        out_root_dir,
        num_to_eval=-1,
        dataset_class=None,
        identity=None,
        render_video_options="/rhome/ppalafox/workspace/render_video_options",
    ):
        mult = 1
        self.reconstruction_res = 256 * mult
        self.max_batch = (mult * 32)**3

        self.labels = labels
        self.labels_tpose = labels_tpose
        self.num_to_eval = num_to_eval
        self.identity = identity

        self.frame_id_list = self.compute_animations()

        self.shape_codes = shape_codes
        self.pose_codes = pose_codes

        self.dataset_class = dataset_class

        self.ref  = None
        self.refw = None

        self.show_ref  = False
        self.show_refw = False

        self.COLOR = [0.6, 0.6, 0.6]

        self.time = 0

        # Recording options
        self.view = "frontal"
        self.render_json            = os.path.join(render_video_options, "render_options.json")
        self.viewpoint_json         = os.path.join(render_video_options, "viewpoint.json")
        self.viewpoint_lateral_json = os.path.join(render_video_options, "viewpoint_lateral.json")
        os.makedirs(render_video_options, exist_ok=True)
        self.out_dir = os.path.join(out_root_dir, identity)
        os.makedirs(self.out_dir, exist_ok=True)

        self.initialize()

    def compute_animations(self):
        # Find the identities' ids
        for identity_id, label_tpose in enumerate(self.labels_tpose):
            if label_tpose['identity_name'] == self.identity:
                self.identity_id = identity_id

        # Find the animations available for the given identity
        frame_id_list = []
        for frame_i, label in enumerate(self.labels):
            if label['identity_name'] == self.identity:
                frame_id_list.append(frame_i)

        return frame_id_list

    def initialize(self):

        self.refw_list = []

        self.loaded_frames = 0

        ################################################################################################################
        # src ref
        ################################################################################################################
        ref_mesh = deepsdf_utils.create_mesh(
            shape_decoder, self.shape_codes, identity_ids=[self.identity_id], shape_codes_dim=shape_codes_dim,
            N=self.reconstruction_res, max_batch=self.max_batch
        )
        p_ref = ref_mesh.vertices.astype(np.float32)
        p_ref_cuda = torch.from_numpy(p_ref)[None, :].cuda()

        # src ref (mesh)
        self.ref = trimesh_to_open3d(ref_mesh, self.COLOR)

        ################################################################################################################
        ################################################################################################################
        # Go over the different poses
        ################################################################################################################
        ################################################################################################################
        for frame_i in tqdm(self.frame_id_list):

            ##########################################################################################
            ##########################################################################################
            ##########################################################################################
            # SRC
            ##########################################################################################
            ##########################################################################################
            ##########################################################################################
            
            points = p_ref_cuda    # [1, 100000, 3]
            points_flat = points.reshape(-1, 3) # [100000, 3]

            with torch.no_grad():
                
                ##########################################################################################
                ### Prepare shape codes
                shape_codes_batch = self.shape_codes[[self.identity_id], :] # [bs, 1, C]
                assert shape_codes_batch.shape == (1, 1, shape_codes_dim), f"{shape_codes_batch} vs {(1, 1, shape_codes_dim)}"

                # Extent latent code to all sampled points
                shape_codes_repeat = shape_codes_batch.expand(-1, points_flat.shape[0], -1) # [bs, N, C]
                shape_codes_inputs = shape_codes_repeat.reshape(-1, shape_codes_dim) # [bs*N, C]

                ### Prepare pose codes
                pose_codes_batch = self.pose_codes[[frame_i], ...] # [bs, 1, C]
                assert pose_codes_batch.shape == (1, 1, pose_codes_dim), f"{pose_codes_batch.shape} vs {(1, 1, pose_codes_dim)}"

                # Extent latent code to all sampled points
                pose_codes_repeat = pose_codes_batch.expand(-1, points_flat.shape[0], -1) # [bs, N, C]
                pose_codes_inputs = pose_codes_repeat.reshape(-1, pose_codes_dim) # [bs*N, C]
                ##########################################################################################

                # Concatenate pose and shape codes
                shape_pose_codes_inputs = torch.cat([shape_codes_inputs, pose_codes_inputs], 1)
                
                # Concatenate (for each sample point), the corresponding code and the p_cur coords
                pose_inputs = torch.cat([shape_pose_codes_inputs, points_flat], 1)

                # Predict delta flow
                p_ref_warped, _ = pose_decoder(pose_inputs) # [bs*N, 3]

            # REFW
            p_ref_warped = p_ref_warped.detach().cpu().numpy()
            ref_warped_mesh_o3d = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(p_ref_warped),
                o3d.utility.Vector3iVector(ref_mesh.faces),
            )
            ref_warped_mesh_o3d.compute_vertex_normals()
            ref_warped_mesh_o3d.paint_uniform_color(self.COLOR)
            self.refw_list.append(ref_warped_mesh_o3d)

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

    def update_refw(self, vis):
        param = vis.get_view_control().convert_to_pinhole_camera_parameters()

        if self.refw is not None:
            vis.remove_geometry(self.refw)
        
        if self.show_refw and len(self.refw_list) > 0:
            self.refw = self.refw_list[self.time]
            vis.add_geometry(self.refw)

        ctr = vis.get_view_control()
        ctr.convert_from_pinhole_camera_parameters(param)

    def update_ref(self, vis):
        param = vis.get_view_control().convert_to_pinhole_camera_parameters()

        if self.ref is not None:
            vis.remove_geometry(self.ref)
        
        if self.show_ref:
            vis.add_geometry(self.ref)

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
        else:
            exit()
        ctr.convert_from_pinhole_camera_parameters(param)

    def render_image(self, vis, out_filename):
        image_np = np.asarray(vis.capture_screen_float_buffer(False))
        h, w, _ = image_np.shape
        new_h, new_w = h, h
        image_np = image_np[(h-new_h)//2:(h+new_h)//2, (w-new_w)//2:(w+new_w)//2,:]
        plt.imsave(f"{self.out_dir}/{out_filename}.jpg", image_np)
        
    def run(self):

        def update_all(vis):
            self.update_refw(vis)
            self.update_ref(vis)
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

        def toggle_ref(vis):
            self.show_ref = not self.show_ref
            self.update_ref(vis)
            return False

        def toggle_refw(vis):
            self.show_refw = not self.show_refw
            self.update_refw(vis)
            return False

        def render(vis):
            print("::render")

            self._load_render_and_viewpoint_option(vis, self.view)

            ##################################################
            # Render the tpose of the tgt identity
            ##################################################
            self.show_ref = True
            update_all(vis)
            vis.poll_events()
            vis.update_renderer()
            self.render_image(vis, "Tpose")
            self.show_ref = False

            ##################################################
            # Render the poses
            ##################################################
            self.time = 0
            self.show_refw = True 
            self.refw = self.refw_list[self.time]
            update_all(vis)
            vis.poll_events()
            vis.update_renderer()
            for i in range(len(self.refw_list)):
                # Render
                self.render_image(vis, f"posed_{str(i).zfill(2)}")
                toggle_next(vis)
                vis.poll_events()
                vis.update_renderer()
            self.show_refw = False 

            return False

        key_to_callback = {}
        key_to_callback[ord("D")] = toggle_next
        key_to_callback[ord("A")] = toggle_previous
        key_to_callback[ord("Z")] = toggle_ref
        key_to_callback[ord("X")] = toggle_refw
        key_to_callback[ord("R")] = render

        # Add mesh at initial time step.
        assert self.time < self.loaded_frames

        print("Showing time", self.time)

        # Start showing the tgt tpose
        self.show_ref = True

        o3d.visualization.draw_geometries_with_key_callbacks([self.world_frame, self.ref], key_to_callback)


################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################


if __name__ == "__main__":

    torch.backends.cudnn.benchmark = True

    ########################################################################################################################
    # Options
    ########################################################################################################################
    # import argparse
    # parser = argparse.ArgumentParser(
    #     description='Run Model'
    # )
    # parser.add_argument('-o', '--optimize_codes', action='store_true')
    # parser.add_argument('-e', '--extra_name', default="")
    # parser.add_argument('-v', '--viz', action='store_true')
    # parser.add_argument('-n', '--optim_name', default=None)

    # try:
    #     args = parser.parse_args()
    # except:
    #     args = parser.parse_known_args()[0]

    out_root_dir = "/cluster_HDD/lothlann/ppalafox/qualitative_results__FOR_ARCHITECTURE_FIGURE"
    
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

    dataset_name = "MIX-POSE__AMASS-50id-10349__MIXAMO-165id-40000__CAPE-35id-20533"

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
    
    #######################################################################################################
    # Data
    #######################################################################################################
    with open(labels_json, "r") as f:
        labels = json.loads(f.read())

    with open(labels_tpose_json, "r") as f:
        labels_tpose = json.loads(f.read())

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
    if checkpoint_tpose is None:
        checkpoint_tpose = checkpoint
    
    shape_decoder = ShapeDecoder(shape_codes_dim, **shape_network_specs).cuda()
    shape_decoder.load_state_dict(checkpoint_tpose['model_state_dict_shape_decoder'])
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

    print("Loading shape codes")
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

    viewer = Viewer(
        labels,
        labels_tpose,
        shape_codes,
        pose_codes,
        out_root_dir,
        num_to_eval=-1,
        dataset_class="mixamo_trans_all",
        identity="adam",
    )
    viewer.run()