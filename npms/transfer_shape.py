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


class ViewerShapeTransfer:
    def __init__(
        self, 
        labels,
        labels_tpose,
        shape_codes,
        pose_codes,
        out_root_dir,
        dataset_class=None,
        source_identity=None, # we'll use the shape code from this identity
        target_identity=None, # we'll use the pose code from this identity 
        target_identity_animation=None,
        target_identity_sample=None,
        use_gt_tpose_mesh=False,
        render_video_options="/rhome/ppalafox/workspace/render_video_options",
    ):
        mult = 2
        self.reconstruction_res = 256 * mult
        self.max_batch = (mult * 32)**3

        self.labels = labels
        self.labels_tpose = labels_tpose
        self.num_to_eval = num_to_eval
        self.src_identity = source_identity
        self.tgt_identity = target_identity
        self.target_identity_animation = target_identity_animation
        self.target_identity_sample = target_identity_sample

        self.shape_codes = shape_codes
        self.pose_codes = pose_codes

        self.dataset_class = dataset_class

        self.src_ref = None
        self.tgt_ref = None
        self.src_refw = None
        self.tgt_refw = None

        self.show_src_ref = False
        self.show_tgt_ref = False
        self.show_src_refw = False
        self.show_tgt_refw = False

        # Find the source and target identity's id
        self.src_identity_id, self.tgt_identity_id = None, None
        for identity_id, label_tpose in enumerate(self.labels_tpose):
            if label_tpose['identity_name'] == self.src_identity:
                self.src_identity_id = identity_id
            if label_tpose['identity_name'] == self.tgt_identity:
                self.tgt_identity_id = identity_id
        assert self.src_identity_id is not None and self.tgt_identity_id is not None

        # Find the index of the tgt sample_id
        self.frame_idx = None
        for pose_idx, label in enumerate(self.labels):
            if label['identity_name' ] == self.tgt_identity \
            and label['animation_name'] == self.target_identity_animation \
            and label['sample_id'] == self.target_identity_sample:
                self.frame_idx = pose_idx
        assert self.frame_idx is not None

        # print(self.src_identity_id, self.tgt_identity_id, self.frame_idx)

        self.SRC_REFW_COLOR = SRC_REFW_COLOR
        self.TGT_REFW_COLOR = TGT_REFW_COLOR

        self.use_gt_tpose_mesh = use_gt_tpose_mesh

        # Recording options
        self.view = "frontal"
        self.render_json            = os.path.join(render_video_options, "render_options.json")
        self.viewpoint_json         = os.path.join(render_video_options, "viewpoint.json")
        self.viewpoint_lateral_json = os.path.join(render_video_options, "viewpoint_lateral.json")
        os.makedirs(render_video_options, exist_ok=True)
        self.out_dir = os.path.join(out_root_dir, f"{target_identity}__ON__{target_identity_animation}__{target_identity_sample}__TO__{source_identity}")
        os.makedirs(self.out_dir, exist_ok=True)

        self.initialize()

    def initialize(self):

        self.loaded_frames = 0

        src_ref_path = os.path.join(data_dir, self.dataset_class, self.src_identity, "a_t_pose", "000000")
        tgt_ref_path = os.path.join(data_dir, self.dataset_class, self.tgt_identity, "a_t_pose", "000000")

        ################################################################################################################
        # src ref
        ################################################################################################################
        if self.use_gt_tpose_mesh:
            src_ref_sample_path = os.path.join(src_ref_path, 'mesh_normalized.ply')
            assert os.path.isfile(src_ref_sample_path), src_ref_sample_path
            src_ref_mesh = trimesh.load(src_ref_sample_path)
        else:
            print("Generating src tpose mesh from code")
            src_ref_mesh = deepsdf_utils.create_mesh(
                shape_decoder, self.shape_codes, identity_ids=[self.src_identity_id], shape_codes_dim=shape_codes_dim,
                N=self.reconstruction_res, max_batch=self.max_batch
            )
            print("Generated")
        p_src_ref = src_ref_mesh.vertices.astype(np.float32)
        p_src_ref_cuda = torch.from_numpy(p_src_ref)[None, :].cuda()

        # src ref (mesh)
        self.src_ref = trimesh_to_open3d(src_ref_mesh, self.SRC_REFW_COLOR)

        ################################################################################################################
        # tgt ref
        ################################################################################################################
        if self.use_gt_tpose_mesh:
            tgt_ref_sample_path = os.path.join(tgt_ref_path, 'mesh_normalized.ply')
            assert os.path.isfile(tgt_ref_sample_path), tgt_ref_sample_path
            tgt_ref_mesh = trimesh.load(tgt_ref_sample_path)
        else:
            print("Generating tgt tpose mesh from code")
            tgt_ref_mesh = deepsdf_utils.create_mesh(
                shape_decoder, self.shape_codes, identity_ids=[self.tgt_identity_id], shape_codes_dim=shape_codes_dim,
                N=self.reconstruction_res, max_batch=self.max_batch
            )
            print("Generated")
        p_tgt_ref = tgt_ref_mesh.vertices.astype(np.float32)
        p_tgt_ref_cuda = torch.from_numpy(p_tgt_ref)[None, :].cuda()

        # tgt ref (mesh)
        self.tgt_ref = trimesh_to_open3d(tgt_ref_mesh, self.TGT_REFW_COLOR)
        
        # o3d.visualization.draw_geometries([self.src_ref])
        # o3d.visualization.draw_geometries([self.tgt_ref])
        # exit()

        ################################################################################################################
        ################################################################################################################
        # Pose both src and tgt identities with the pose code of tgt
        ################################################################################################################
        ################################################################################################################

        ##########################################################################################
        # SRC
        ##########################################################################################
        points = p_src_ref_cuda    # [1, 100000, 3]
        points_flat = points.reshape(-1, 3) # [100000, 3]

        with torch.no_grad():
            ##########################################################################################
            ### Prepare shape codes
            shape_codes_batch = self.shape_codes[[self.src_identity_id], :] # [bs, 1, C]
            assert shape_codes_batch.shape == (1, 1, shape_codes_dim), f"{shape_codes_batch} vs {(1, 1, shape_codes_dim)}"

            # Extent latent code to all sampled points
            shape_codes_repeat = shape_codes_batch.expand(-1, points_flat.shape[0], -1) # [bs, N, C]
            shape_codes_inputs = shape_codes_repeat.reshape(-1, shape_codes_dim) # [bs*N, C]

            ### Prepare pose codes
            pose_codes_batch = self.pose_codes[[self.frame_idx], ...] # [bs, 1, C]
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
            p_src_ref_warped, _ = pose_decoder(pose_inputs) # [bs*N, 3]

            # REFW
            p_src_ref_warped = p_src_ref_warped.detach().cpu().numpy()
            src_ref_warped_mesh_o3d = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(p_src_ref_warped),
                o3d.utility.Vector3iVector(src_ref_mesh.faces),
            )
            src_ref_warped_mesh_o3d.compute_vertex_normals()
            src_ref_warped_mesh_o3d.paint_uniform_color(self.SRC_REFW_COLOR)
            self.src_refw = src_ref_warped_mesh_o3d

        ##########################################################################################
        # TGT
        ##########################################################################################
        points = p_tgt_ref_cuda    # [1, 100000, 3]
        points_flat = points.reshape(-1, 3) # [100000, 3]

        with torch.no_grad():
            ##########################################################################################
            ### Prepare shape codes
            shape_codes_batch = self.shape_codes[[self.tgt_identity_id], :] # [bs, 1, C]
            assert shape_codes_batch.shape == (1, 1, shape_codes_dim), f"{shape_codes_batch} vs {(1, 1, shape_codes_dim)}"

            # Extent latent code to all sampled points
            shape_codes_repeat = shape_codes_batch.expand(-1, points_flat.shape[0], -1) # [bs, N, C]
            shape_codes_inputs = shape_codes_repeat.reshape(-1, shape_codes_dim) # [bs*N, C]

            ### Prepare pose codes
            pose_codes_batch = self.pose_codes[[self.frame_idx], ...] # [bs, 1, C]
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
            p_tgt_ref_warped, _ = pose_decoder(pose_inputs) # [bs*N, 3]

            # REFW
            # REFW
            p_tgt_ref_warped = p_tgt_ref_warped.detach().cpu().numpy()
            tgt_ref_warped_mesh_o3d = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(p_tgt_ref_warped),
                o3d.utility.Vector3iVector(tgt_ref_mesh.faces),
            )
            tgt_ref_warped_mesh_o3d.compute_vertex_normals()
            tgt_ref_warped_mesh_o3d.paint_uniform_color(self.TGT_REFW_COLOR)
            self.tgt_refw = tgt_ref_warped_mesh_o3d

            ##########################################################################################
            ##########################################################################################

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

    def update_src_refw(self, vis):
        param = vis.get_view_control().convert_to_pinhole_camera_parameters()

        if self.src_refw is not None:
            vis.remove_geometry(self.src_refw)
        
        if self.show_src_refw:
            vis.add_geometry(self.src_refw)

        ctr = vis.get_view_control()
        ctr.convert_from_pinhole_camera_parameters(param)

    def update_tgt_refw(self, vis):
        param = vis.get_view_control().convert_to_pinhole_camera_parameters()

        if self.tgt_refw is not None:
            vis.remove_geometry(self.tgt_refw)
        
        if self.show_tgt_refw:
            vis.add_geometry(self.tgt_refw)

        ctr = vis.get_view_control()
        ctr.convert_from_pinhole_camera_parameters(param)

    def update_src_ref(self, vis):
        param = vis.get_view_control().convert_to_pinhole_camera_parameters()

        if self.src_ref is not None:
            vis.remove_geometry(self.src_ref)
        
        if self.show_src_ref:
            vis.add_geometry(self.src_ref)

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
            self.update_src_refw(vis)
            self.update_tgt_refw(vis)
            self.update_src_ref(vis)
            return False

        def toggle_src_refw(vis):
            self.show_src_refw = not self.show_src_refw
            self.update_src_refw(vis)
            return False

        def toggle_tgt_refw(vis):
            self.show_tgt_refw = not self.show_tgt_refw
            self.update_tgt_refw(vis)
            return False

        def toggle_src_ref(vis):
            self.show_src_ref = not self.show_src_ref
            self.update_src_ref(vis)
            return False
        
        def render(vis):
            print("::render")

            self._load_render_and_viewpoint_option(vis, self.view)

            self.show_src_ref = False
            self.show_tgt_ref = False
            self.show_src_refw = False
            self.show_tgt_refw = False

            ##################################################
            # Render the posed target
            ##################################################
            self.show_tgt_refw = True
            update_all(vis)
            vis.poll_events()
            vis.update_renderer()
            self.render_image(vis, "tgt_posed")
            self.show_tgt_refw = False

            ##################################################
            # Render the tpose of the src identity
            ##################################################
            self.show_src_ref = True
            update_all(vis)
            vis.poll_events()
            vis.update_renderer()
            self.render_image(vis, "src_Tpose")
            self.show_src_ref = False

            ##################################################
            # Render the posed src
            ##################################################
            self.show_src_refw = True
            update_all(vis)
            vis.poll_events()
            vis.update_renderer()
            self.render_image(vis, "src_posed")
            self.show_src_refw = False
            
            return False

        key_to_callback = {}
        key_to_callback[ord("T")] = toggle_src_ref
        key_to_callback[ord("W")] = toggle_src_refw
        key_to_callback[ord("V")] = toggle_tgt_refw
        key_to_callback[ord("R")] = render

        # Start showing the tgt tpose
        self.show_src_refw = True

        # o3d.visualization.draw_geometries_with_key_callbacks([self.unit_bbox, self.src_refw, self.src_cur], key_to_callback)
        o3d.visualization.draw_geometries_with_key_callbacks([self.world_frame, self.src_refw], key_to_callback)


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

    demo_name = "qualitative_results__transfer_shape"

    out_root_dir = f"/cluster_HDD/lothlann/ppalafox/{demo_name}"
    
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

    if shape_codes.shape[0] != pretrained_shape_codes.shape[0] and train_to_augmented is not None:
        print("Loading shape codes - A")
        shape_codes = pretrained_shape_codes[list(train_to_augmented.values())].detach().clone()
        if len(shape_codes.shape) == 2:
            shape_codes = shape_codes.unsqueeze(0)
    else:
        print("Loading shape codes - B")
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
    
    TGT_REFW_COLOR = [0.6, 0.4, 0.4]

    print()
    print()
    print("#######################################################################")
    print("Final visualization")
    print("#######################################################################")

    # source_identity = "adam"
    # target_identity = "kate"
    # target_identity_animation = "breakdance_uprock"
    # target_identity_sample = 13
    # SRC_REFW_COLOR = [0.4, 0.6, 0.4]

    source_identity = "adam"
    target_identity = "megan"
    target_identity_animation = "afoxe_samba_reggae_dance"
    target_identity_sample = 2
    SRC_REFW_COLOR = [0.4, 0.6, 0.6]

    # source_identity = "racer"
    # target_identity = "megan"
    # target_identity_animation = "afoxe_samba_reggae_dance"
    # target_identity_sample = 2
    # SRC_REFW_COLOR = [0.4, 0.6, 0.4]
    
    # source_identity = "douglas"
    # target_identity = "megan"
    # target_identity_animation = "afoxe_samba_reggae_dance"
    # target_identity_sample = 2
    # SRC_REFW_COLOR = [0.6, 0.6, 0.4]
    

    viewer = ViewerShapeTransfer(
        labels,
        labels_tpose,
        shape_codes,
        pose_codes,
        out_root_dir,
        dataset_class="mixamo_trans_all",
        source_identity=source_identity, # we'll use the shape code from this identity
        target_identity=target_identity, # we'll use the pose code from this identity 
        target_identity_animation=target_identity_animation,
        target_identity_sample= str(target_identity_sample).zfill(6),
    )
    viewer.run()