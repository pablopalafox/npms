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
import argparse

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


class ViewerRepose:
    def __init__(
        self, 
        labels,
        labels_tpose,
        shape_codes_pretrained,
        shape_codes_sequence,
        pose_codes_sequence,
        out_root_dir,
        num_to_eval=-1,
        dataset_class=None,
        target_identity=None,
        test_dataset_name=None,
        render_video_options="/rhome/ppalafox/workspace/render_video_options",
        frame_rate=30,
        res_mult=1,
    ):
        self.frame_rate = frame_rate

        self.num_to_eval = num_to_eval
        print("num to eval", self.num_to_eval)

        self.test_dataset_name = test_dataset_name

        mult = res_mult
        self.reconstruction_res = 256 * mult
        self.max_batch = (mult * 32)**3

        self.labels = labels
        self.labels_tpose = labels_tpose
        self.tgt_identity = target_identity

        # Find the identities' ids
        for identity_id, label_tpose in enumerate(self.labels_tpose):
            if label_tpose['identity_name'] == self.tgt_identity:
                self.tgt_identity_id = identity_id

        self.shape_codes_pretrained = shape_codes_pretrained
        self.shape_codes_sequence = shape_codes_sequence
        self.pose_codes_sequence = pose_codes_sequence

        self.dataset_class = dataset_class

        self.src_refw = None
        self.tgt_ref  = None
        self.tgt_refw = None

        self.show_src_refw = False
        self.show_tgt_ref = False
        self.show_tgt_refw = False

        self.SRC_REFW_COLOR = [0.4, 0.6, 0.4]
        self.TGT_REFW_COLOR = [0.6, 0.4, 0.4]

        self.time = 0

        # Recording options
        self.view = "frontal"
        self.render_json            = os.path.join(render_video_options, "render_options.json")
        self.viewpoint_json         = os.path.join(render_video_options, "viewpoint.json")
        self.viewpoint_lateral_json = os.path.join(render_video_options, "viewpoint_lateral.json")
        os.makedirs(render_video_options, exist_ok=True)
        self.out_dir = os.path.join(out_root_dir, f"{test_dataset_name}__to__{target_identity}__@{self.reconstruction_res}")
        os.makedirs(self.out_dir, exist_ok=True)

        self.initialize()

    def initialize(self):

        self.src_refw_list = []
        self.tgt_refw_list = []

        self.loaded_frames = 0

        ################################################################################################################
        # src ref
        ################################################################################################################
        # Extract
        src_ref_mesh = deepsdf_utils.create_mesh(
            shape_decoder, self.shape_codes_sequence, identity_ids=[0], shape_codes_dim=shape_codes_dim,
            N=self.reconstruction_res, max_batch=self.max_batch
        )
        p_src_ref = src_ref_mesh.vertices.astype(np.float32)
        p_src_ref_cuda = torch.from_numpy(p_src_ref)[None, :].cuda()
        p_src_ref_flat_cuda = p_src_ref_cuda.reshape(-1, 3) # [100000, 3]

        # src ref (mesh)
        src_ref_mesh_o3d = trimesh_to_open3d(src_ref_mesh, self.SRC_REFW_COLOR)
        self.src_ref = src_ref_mesh_o3d
        print("Extracted src shape")

        ### Prepare shape codes
        src_shape_codes_batch = self.shape_codes_sequence[[0], :] # [bs, 1, C]
        assert src_shape_codes_batch.shape == (1, 1, shape_codes_dim), f"{src_shape_codes_batch} vs {(1, 1, shape_codes_dim)}"

        # Extent latent code to all sampled points
        src_shape_codes_repeat = src_shape_codes_batch.expand(-1, p_src_ref_flat_cuda.shape[0], -1) # [bs, N, C]
        src_shape_codes_inputs = src_shape_codes_repeat.reshape(-1, shape_codes_dim) # [bs*N, C]

        ################################################################################################################
        # tgt ref
        ################################################################################################################
        tgt_ref_mesh = deepsdf_utils.create_mesh(
            shape_decoder, self.shape_codes_pretrained, identity_ids=[self.tgt_identity_id], shape_codes_dim=shape_codes_dim,
            N=self.reconstruction_res, max_batch=self.max_batch
        )
        p_tgt_ref = tgt_ref_mesh.vertices.astype(np.float32)
        p_tgt_ref_cuda = torch.from_numpy(p_tgt_ref)[None, :].cuda()
        p_tgt_ref_flat_cuda = p_tgt_ref_cuda.reshape(-1, 3) # [100000, 3]

        # tgt ref (mesh)
        tgt_ref_mesh_o3d = trimesh_to_open3d(tgt_ref_mesh, self.TGT_REFW_COLOR)
        self.tgt_ref = tgt_ref_mesh_o3d
        print("Extracted tgt shape")

        ### Prepare shape codes
        tgt_shape_codes_batch = self.shape_codes_pretrained[[self.tgt_identity_id], :] # [bs, 1, C]
        assert tgt_shape_codes_batch.shape == (1, 1, shape_codes_dim), f"{tgt_shape_codes_batch} vs {(1, 1, shape_codes_dim)}"

        # Extent latent code to all sampled points
        tgt_shape_codes_repeat = tgt_shape_codes_batch.expand(-1, p_tgt_ref_flat_cuda.shape[0], -1) # [bs, N, C]
        tgt_shape_codes_inputs = tgt_shape_codes_repeat.reshape(-1, shape_codes_dim) # [bs*N, C]
        
        ################################################################################################################
        # o3d.visualization.draw_geometries([src_ref_mesh_o3d])
        # o3d.visualization.draw_geometries([tgt_ref_mesh_o3d])
        ################################################################################################################

        ################################################################################################################
        ################################################################################################################
        # Go over the different poses
        ################################################################################################################
        ################################################################################################################
        for frame_i in tqdm(range(len(self.pose_codes_sequence))):

            ### Prepare pose codes
            pose_codes_batch = self.pose_codes_sequence[[frame_i], ...] # [bs, 1, C]
            assert pose_codes_batch.shape == (1, 1, pose_codes_dim), f"{pose_codes_batch.shape} vs {(1, 1, pose_codes_dim)}"

            ##########################################################################################
            ##########################################################################################
            # SRC
            ##########################################################################################
            ##########################################################################################
            with torch.no_grad():
                ##########################################################################################
                # Extent latent code to all sampled points
                pose_codes_repeat = pose_codes_batch.expand(-1, p_src_ref_flat_cuda.shape[0], -1) # [bs, N, C]
                pose_codes_inputs = pose_codes_repeat.reshape(-1, pose_codes_dim) # [bs*N, C]
                ##########################################################################################

                # Concatenate pose and shape codes
                shape_pose_codes_inputs = torch.cat([src_shape_codes_inputs, pose_codes_inputs], 1)
                
                # Concatenate (for each sample point), the corresponding code and the p_cur coords
                pose_inputs = torch.cat([shape_pose_codes_inputs, p_src_ref_flat_cuda], 1)

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
            self.src_refw_list.append(src_ref_warped_mesh_o3d)

            ##########################################################################################
            ##########################################################################################
            # TGT
            ##########################################################################################
            ##########################################################################################
            with torch.no_grad():
                
                ##########################################################################################
                # Extent latent code to all sampled points
                pose_codes_repeat = pose_codes_batch.expand(-1, p_tgt_ref_flat_cuda.shape[0], -1) # [bs, N, C]
                pose_codes_inputs = pose_codes_repeat.reshape(-1, pose_codes_dim) # [bs*N, C]
                ##########################################################################################

                # Concatenate pose and shape codes
                shape_pose_codes_inputs = torch.cat([tgt_shape_codes_inputs, pose_codes_inputs], 1)
                
                # Concatenate (for each sample point), the corresponding code and the p_cur coords
                pose_inputs = torch.cat([shape_pose_codes_inputs, p_tgt_ref_flat_cuda], 1)

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
            self.tgt_refw_list.append(tgt_ref_warped_mesh_o3d)

            ##########################################################################################
            # o3d.visualization.draw_geometries([src_ref_warped_mesh_o3d])
            # o3d.visualization.draw_geometries([tgt_ref_warped_mesh_o3d])
            ##########################################################################################

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

    def update_src_refw(self, vis):
        param = vis.get_view_control().convert_to_pinhole_camera_parameters()

        if self.src_refw is not None:
            vis.remove_geometry(self.src_refw)
        
        if self.show_src_refw:
            self.src_refw = self.src_refw_list[self.time]
            vis.add_geometry(self.src_refw)

        ctr = vis.get_view_control()
        ctr.convert_from_pinhole_camera_parameters(param)

    def update_tgt_refw(self, vis):
        param = vis.get_view_control().convert_to_pinhole_camera_parameters()

        if self.tgt_refw is not None:
            vis.remove_geometry(self.tgt_refw)
        
        if self.show_tgt_refw:
            self.tgt_refw = self.tgt_refw_list[self.time]
            vis.add_geometry(self.tgt_refw)

        ctr = vis.get_view_control()
        ctr.convert_from_pinhole_camera_parameters(param)

    def update_tgt_ref(self, vis):
        param = vis.get_view_control().convert_to_pinhole_camera_parameters()

        if self.tgt_ref is not None:
            vis.remove_geometry(self.tgt_ref)
        
        if self.show_tgt_ref:
            vis.add_geometry(self.tgt_ref)

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
        new_h, new_w = 1200, 1600
        image_np = image_np[(h-new_h)//2:(h+new_h)//2, (w-new_w)//2:(w+new_w)//2,:]
        plt.imsave(f"{self.out_dir}/{out_filename}.jpg", image_np)
        
    def run(self):

        def update_all(vis):
            self.update_src_refw(vis)
            self.update_tgt_refw(vis)
            self.update_tgt_ref(vis)
            return False

        # Define callbacks.
        def toggle_next(vis):
            self.time += 1
            if self.time >= self.loaded_frames:
                self.time = 0
            self.update_src_refw(vis)
            self.update_tgt_refw(vis)
            self.update_tgt_ref(vis)
            return False
        
        def toggle_previous(vis):
            self.time -= 1
            if self.time < 0:
                self.time = self.loaded_frames - 1
            self.update_src_refw(vis)
            self.update_tgt_refw(vis)
            self.update_tgt_ref(vis)
            return False

        def toggle_src_refw(vis):
            self.show_src_refw = not self.show_src_refw
            self.update_src_refw(vis)
            return False

        def toggle_tgt_refw(vis):
            self.show_tgt_refw = not self.show_tgt_refw
            self.update_tgt_refw(vis)
            return False

        def toggle_tgt_ref(vis):
            self.show_tgt_ref = not self.show_tgt_ref
            self.update_tgt_ref(vis)
            return False
        
        def render(vis):
            print("::render")

            self.show_tgt_ref = False
            self.show_tgt_ref = False
            self.show_tgt_ref = False
            vis.poll_events()
            vis.update_renderer()

            self._load_render_and_viewpoint_option(vis, self.view)

            ##################################################
            # Render the tpose of the tgt identity
            ##################################################
            self.show_tgt_ref = True
            update_all(vis)
            vis.poll_events()
            vis.update_renderer()
            self.render_image(vis, "tgt_Tpose")
            self.show_tgt_ref = False

            ##################################################
            # Render the src poses
            ##################################################
            self.time = 0
            self.show_src_refw = True 
            self.src_refw = self.src_refw_list[self.time]
            update_all(vis)
            vis.poll_events()
            vis.update_renderer()
            for i in range(len(self.src_refw_list)):
                # Render
                self.render_image(vis, f"src_posed_{str(i).zfill(2)}")
                toggle_next(vis)
                vis.poll_events()
                vis.update_renderer()
            self.show_src_refw = False 

            os.system(f"ffmpeg -r {self.frame_rate} -i {self.out_dir}/src_posed_%02d.jpg -c:v libx264 -vf fps=30 -pix_fmt yuv420p -y {self.out_dir}/{self.test_dataset_name}__to__{self.tgt_identity}__@{self.reconstruction_res}__SRC.mp4")

            ##################################################
            # Render the tgt poses
            ##################################################
            self.time = 0
            self.show_tgt_refw = True 
            self.tgt_refw = self.tgt_refw_list[self.time]
            update_all(vis)
            vis.poll_events()
            vis.update_renderer()
            for i in range(len(self.tgt_refw_list)):
                # Render
                self.render_image(vis, f"tgt_posed_{str(i).zfill(2)}")
                toggle_next(vis)
                vis.poll_events()
                vis.update_renderer()

            os.system(f"ffmpeg -r {self.frame_rate} -i {self.out_dir}/tgt_posed_%02d.jpg -c:v libx264 -vf fps=30 -pix_fmt yuv420p -y {self.out_dir}/{self.test_dataset_name}__to__{self.tgt_identity}__@{self.reconstruction_res}_TGT.mp4")
            exit() 

            return False

        key_to_callback = {}
        key_to_callback[ord("D")] = toggle_next
        key_to_callback[ord("A")] = toggle_previous
        key_to_callback[ord("X")] = toggle_src_refw
        key_to_callback[ord("V")] = toggle_tgt_refw
        key_to_callback[ord("T")] = toggle_tgt_ref
        key_to_callback[ord("R")] = render

        # Add mesh at initial time step.
        assert self.time < self.loaded_frames

        print("Showing time", self.time)

        # Start showing the tgt tpose
        self.show_tgt_ref = True

        # o3d.visualization.draw_geometries_with_key_callbacks([self.unit_bbox, self.src_refw, self.src_cur], key_to_callback)
        o3d.visualization.draw_geometries_with_key_callbacks([self.world_frame, self.tgt_ref], key_to_callback)


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

    ########################################################################################################################
    # optim_name = "2021-03-21__CAPE-POSE-TEST-03223_shortshort-tilt_twist_left-1id-300ts-1seqs-50to349__bs4__icp0.0005-500__itrs1000__sreg0.1_preg0.0001__slr0.0005_plr0.001__interv250_factr0.5__clamp0.1__sigma0.015__tmpreg100__cpt365__woALT__part"
    # optim_name = "2021-03-20__MIXAMO-POSE-TEST-sophie-female_salsa_dancing_sx3-1id-268ts-1seqs__bs4__icp0.0005-500__itrs1000__sreg0.1_preg0.0001__slr0.0005_plr0.001__interv250_factr0.5__clamp0.1__sigma0.015__tmpreg100__cpt365__woALT__part"
    # optim_name = "2021-03-22__MIXAMO_TRANS_ALL-POSE-TEST-joe-breakdance_brooklyn_style_uprocking_sx8-1id-240ts-1seqs__bs4__icp0.0005-500__itrs1000__sreg0.1_preg0.0001__slr0.0005_plr0.001__interv250_factr0.5__clamp0.1__sigma0.015__tmpreg100__cpt171__woALT__part"

    target_identity = "regina"
    optim_name = "2021-03-23__MIXAMO_TRANS_ALL-POSE-TEST-alien-female_hip_hop_slide_step_dancing_sx8-1id-250ts-1seqs__bs4__icp0.0005-500__itrs1000__sreg0.1_preg0.0001__slr0.0005_plr0.001__interv250_factr0.5__clamp0.1__sigma0.015__tmpreg100__cpt128__woALT__part"
    
    # target_identity = "adam"
    # optim_name = "2021-03-23__MIXAMO-POSE-TEST-sophie-female_salsa_dancing_sx3-1id-268ts-1seqs__bs4__icp0.0005-500__itrs1000__sreg0.1_preg0.0001__slr0.0005_plr0.001__interv250_factr0.5__clamp0.1__sigma0.015__tmpreg100__cpt128__woALT__part"
    ########################################################################################################################

    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser(
        description='Run Model'
    )
    parser.add_argument('-n', '--optim_name', default=optim_name)
    parser.add_argument('-d', '--dataset_type', choices=['HUMAN', 'MANO'], required=True)

    try:
        args = parser.parse_args()
    except:
        args = parser.parse_known_args()[0]

    ########################################################################################################################
    ########################################################################################################################
    out_root_dir = "/cluster_HDD/lothlann/ppalafox/qualitative_results__transfer_pose_sequence"
    ########################################################################################################################
    ########################################################################################################################
    
    # ------------------------------------------- #
    dataset_type = args.dataset_type
    print("#"*30)
    print(f"DATASET_TYPE: {dataset_type}")
    print("#"*30)
    # input("Continue?")
    # ------------------------------------------- #

    if dataset_type == "HUMAN":
        from configs_eval.config_eval_HUMAN import *
    elif dataset_type == "MANO":
        from configs_eval.config_eval_MANO import *

    ########################################################################################################################
    ########################################################################################################################

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
    
    #######################################################################################################
    # Data
    #######################################################################################################
    with open(labels_json, "r") as f:
        labels = json.loads(f.read())

    with open(labels_tpose_json, "r") as f:
        labels_tpose = json.loads(f.read())

    num_identities = len(labels_tpose)
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
    shape_codes_pretrained = checkpoint['shape_codes'].cuda().detach().clone()
    print("train shape codes:", shape_codes_pretrained.shape)

    ##################################################################
    # Load optimized codes from test sequence 
    ##################################################################

    print()
    print("Using pretrained pose codes")
    print()

    optim_name = args.optim_name

    print("###############################################")
    print("Reading from:")
    print("exp_dir", exp_dir)
    print("run_dir", args.optim_name)
    print("###############################################")

    optimization_dir = os.path.join(exp_dir, "optimization", optim_name)
    optimized_codes_path = os.path.join(optimization_dir, "codes.npz")
    
    assert os.path.isfile(optimized_codes_path), f"File {optimized_codes_path} does not exist!"

    # Maybe visualize results
    codes_npz = np.load(optimized_codes_path)
    shape_codes_sequence = torch.from_numpy(codes_npz['shape']).cuda()[:, [-1], :]
    pose_codes_sequence  = torch.from_numpy(codes_npz['pose']).cuda()[:, [-1], :]

    shape_codes_sequence.requires_grad = False
    pose_codes_sequence.requires_grad = False

    print("test shape codes", shape_codes_sequence.shape)
    print("test pose  codes", pose_codes_sequence.shape)

    assert pose_codes_sequence.shape[1] == 1 and pose_codes_sequence.shape[2] == pose_codes_dim

    ##################################################################################################################
    ##################################################################################################################
    
    print()
    print()
    print("#######################################################################")
    print("Final visualization")
    print("#######################################################################")

    ########################################################################################################################
    num_to_eval = -1

    res_mult = 2
    ########################################################################################################################

    viewer = ViewerRepose(
        labels,
        labels_tpose,
        shape_codes_pretrained,
        shape_codes_sequence,
        pose_codes_sequence,
        out_root_dir,
        num_to_eval=num_to_eval,
        dataset_class="mixamo_trans_all",
        target_identity=target_identity,
        test_dataset_name=dataset_name,
        res_mult=res_mult
    )
    viewer.run()