import os
import numpy as np
import trimesh
import torch
import open3d as o3d
from tqdm import tqdm

from utils.voxels import VoxelGrid
import utils.deepsdf_utils as deepsdf_utils 
from utils.evaluation import eval_mesh

from utils.pcd_utils import (BBox,
                            transform_pointcloud_to_opengl_coords,
                            rotate_around_axis,
                            origin, normalize_transformation)


class ViewerOptim:
    def __init__(
        self, 
        labels, data_dir,
        shape_decoder, pose_decoder,
        shape_codes, pose_codes,
        num_to_eval=-1,
        load_every=None,
        from_frame_id=0,
        reconstruction_res=256,
        input_res=256,
        max_batch=32**3,
        warp_reconstructed_mesh=False,
        exp_dir=None,
        use_pred_vertices=True,
        viz_mesh=True, # instead of pointcloud
        cache_for_viz=False
    ):
        self.labels = labels
        self.data_dir = data_dir

        self.shape_decoder = shape_decoder
        self.pose_decoder = pose_decoder

        self.shape_codes = shape_codes
        self.pose_codes = pose_codes

        self.shape_codes_dim = self.shape_codes.shape[-1]
        self.pose_codes_dim = self.pose_codes.shape[-1]

        self.num_to_eval = num_to_eval
        self.load_every = load_every
        self.from_frame_id = from_frame_id

        self.time = 0
        self.num_codes_optim_video = pose_codes.shape[1]
        self.iter = self.num_codes_optim_video - 1 # start at the final (optimized) pose code 
        
        self.ref_gt = None
        self.ref  = None
        self.cur  = None
        self.refw  = None
        self.vox = None
        self.pcd = None

        self.show_ref_gt = False
        self.show_ref  = False
        self.show_cur  = False
        self.show_refw = False
        self.show_vox = False
        self.show_pcd = False

        self.reconstruction_res = reconstruction_res
        self.input_res = input_res
        self.max_batch = max_batch

        self.warp_reconstructed_mesh = warp_reconstructed_mesh

        self.use_pred_vertices = use_pred_vertices
        self.viz_mesh = viz_mesh

        self.cache_for_viz = cache_for_viz

        # we store generated meshes here
        self.exp_dir = exp_dir
        if self.exp_dir is not None:
            self.exp_dir = os.path.join(self.exp_dir, f"predicted_meshes_{reconstruction_res}res")
            if not os.path.isdir(self.exp_dir):
                os.makedirs(self.exp_dir)

        self.initialize()
    
    def initialize(self):

        self.ref_gt_dict = {}
        self.ref_dict  = {}
        self.cur_list  = []
        self.refw_list = []
        self.vox_list = []
        self.pcd_list = []

        self.identity_name_by_loaded_frame = {}

        self.loaded_frames = 0

        # Go over sequence and compute losses 
        for frame_i in tqdm(range(len(self.labels))):

            if self.load_every and frame_i % self.load_every != 0:
                continue

            if frame_i < self.from_frame_id:
                continue
            
            label = self.labels[frame_i]

            dataset, identity_id, identity_name, animation_name, sample_id = \
                label['dataset'], label['identity_id'], label['identity_name'], label['animation_name'], label['sample_id']

            ref_path = os.path.join(self.data_dir, dataset, identity_name, "a_t_pose", "000000")
            cur_path = os.path.join(self.data_dir, dataset, identity_name, animation_name, sample_id)

            print(f"Loading ref: {ref_path}")
            print(f"Loading cur: {cur_path}")

            sample_num = 100000

            # Cur
            cur_mesh_path = os.path.join(cur_path, 'mesh_normalized.ply')
            cur_mesh = trimesh.load(cur_mesh_path)
            # p_cur_np, _ = trimesh.sample.sample_surface_even(cur_mesh, sample_num)
            p_cur_np = cur_mesh.sample(sample_num)
            p_cur_np = p_cur_np.astype(np.float32)
            p_cur = torch.from_numpy(p_cur_np)[None, :].cuda()
            p_cur_flat = p_cur.reshape(-1, 3) # [100000, 3]

            # Ref
            ref_mesh_path = os.path.join(ref_path, 'mesh_normalized.ply')
            p_ref_gt = None
            if os.path.isfile(ref_mesh_path):
                ref_mesh = trimesh.load(ref_mesh_path)
                # p_ref_np, _ = trimesh.sample.sample_surface_even(ref_mesh, sample_num)
                p_ref_np = ref_mesh.sample(sample_num)
                p_ref_np = p_ref_np.astype(np.float32)
                p_ref_gt = p_ref_np.copy()
                p_ref_gt = np.reshape(p_ref_gt, (-1, 3))
            
                if identity_name not in self.ref_dict and not self.warp_reconstructed_mesh:
                    p_ref = torch.from_numpy(p_ref_np)[None, :].cuda()
                    p_ref_flat = p_ref.reshape(-1, 3) # [100000, 3]

            ########################################################################
            # Inference for points REF 2 CUR
            ########################################################################

            # REF WARPED
            ref_optim  = []
            refw_optim = []

            for it in range(self.num_codes_optim_video):

                if not self.cache_for_viz:
                    # If we won't be visualizing the results, then we don't need
                    # to compute meshes for all the optimization steps -- we just need the final step
                    if it < (self.num_codes_optim_video - 1):
                        continue
                    
                print("\tit", it)
                with torch.no_grad():

                    # Warp the reconstructed t-pose
                    if identity_name not in self.ref_dict and self.warp_reconstructed_mesh:
                        mesh_pred_trimesh = deepsdf_utils.create_mesh(
                            self.shape_decoder, self.shape_codes[:, [it], :], identity_ids=[identity_id], shape_codes_dim=self.shape_codes_dim,
                            N=self.reconstruction_res, max_batch=self.max_batch
                        )
                        if self.use_pred_vertices or self.viz_mesh:
                            p_ref_np = mesh_pred_trimesh.vertices
                        else:
                            p_ref_np, _ = trimesh.sample.sample_surface_even(mesh_pred_trimesh, p_cur_np.shape[0])
                        p_ref_np = p_ref_np.astype(np.float32)
                        p_ref = torch.from_numpy(p_ref_np)[None, :].cuda()
                        p_ref_flat = p_ref.reshape(-1, 3) # [100000, 3]

                    ##########################################################################################
                    # Prepare shape codes
                    shape_codes_batch = self.shape_codes[identity_id, it, :] # [bs, 1, C]
                    shape_codes_batch = shape_codes_batch[None, None, :]

                    assert shape_codes_batch.shape == (1, 1, self.shape_codes_dim), f"{shape_codes_batch.shape[0]} vs {(1, 1, self.shape_codes_dim)}"

                    # Extent latent code to all sampled points
                    shape_codes_repeat = shape_codes_batch.expand(-1, p_ref_flat.shape[0], -1) # [bs, N, C]
                    shape_codes_inputs = shape_codes_repeat.reshape(-1, self.shape_codes_dim) # [bs*N, C]

                    ##########################################################################################
                    
                    ##########################################################################################
                    # Prepare pose codes
                    pose_codes_batch = self.pose_codes[frame_i, it, :] # [bs, 1, C]
                    pose_codes_batch = pose_codes_batch[None, None, :]

                    assert pose_codes_batch.shape == (1, 1, self.pose_codes_dim), f"{pose_codes_batch.shape[0]} vs {(1, 1, self.pose_codes_dim)}"

                    # Extent latent code to all sampled points
                    pose_codes_repeat = pose_codes_batch.expand(-1, p_ref_flat.shape[0], -1) # [bs, N, C]
                    pose_codes_inputs = pose_codes_repeat.reshape(-1, self.pose_codes_dim) # [bs*N, C]
                    ##########################################################################################

                    # Concatenate pose and shape codes
                    shape_pose_codes_inputs = torch.cat([shape_codes_inputs, pose_codes_inputs], 1)

                    # Concatenate (for each sample point), the corresponding code and the p_cur coords
                    pose_inputs = torch.cat([shape_pose_codes_inputs, p_ref_flat], 1)

                    # Predict delta flow
                    p_ref_warped_to_ref, _ = self.pose_decoder(pose_inputs) # [bs*N, 3]

                # ref
                if identity_name not in self.ref_dict and self.warp_reconstructed_mesh:
                    p_ref_flat_np = p_ref_flat.detach().cpu().numpy()
                    p_ref_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p_ref_flat_np))
                    p_ref_pcd.paint_uniform_color([1, 0, 0]) # ref is in red
                    p_ref_pcd.estimate_normals()
                    if self.cache_for_viz:
                        ref_optim.append(p_ref_pcd)
                
                # ref warped
                p_ref_warped_to_ref = p_ref_warped_to_ref.detach().cpu().numpy()
                if self.viz_mesh:
                    ref_warped_o3d = o3d.geometry.TriangleMesh(
                        o3d.utility.Vector3dVector(p_ref_warped_to_ref),
                        o3d.utility.Vector3iVector(mesh_pred_trimesh.faces),
                    )
                    ref_warped_o3d.compute_vertex_normals()
                    if self.cache_for_viz:
                        refw_optim.append(ref_warped_o3d)
                else:    
                    p_ref_warped_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p_ref_warped_to_ref))
                    p_ref_warped_pcd.paint_uniform_color([0, 0, 1]) # ref warped to current is in blue
                    p_ref_warped_pcd.estimate_normals()
                    if self.cache_for_viz:
                        refw_optim.append(p_ref_warped_pcd)

                # ------------------------------------------------------------------------------------------------
                # Store mesh from the final iteration
                # ------------------------------------------------------------------------------------------------
                if self.exp_dir is not None:
                    if it == self.num_codes_optim_video - 1:
                        
                        assert self.warp_reconstructed_mesh and self.use_pred_vertices
                        
                        ##########################################
                        # Save warped shape
                        ##########################################
                        # Form mesh using the deformed points and the faces from the reconstructed tpose
                        ref_warped_o3d = o3d.geometry.TriangleMesh(
                            o3d.utility.Vector3dVector(p_ref_warped_to_ref),
                            o3d.utility.Vector3iVector(mesh_pred_trimesh.faces),
                        )
                        ref_warped_o3d.compute_vertex_normals()
                        # o3d.visualization.draw_geometries([ref_warped_o3d])

                        # Save
                        sample_out_dir = os.path.join(self.exp_dir, label['sample_id'])
                        if not os.path.isdir(sample_out_dir): os.mkdir(sample_out_dir)
                        sample_out_path = os.path.join(sample_out_dir, "ref_warped.ply")
                        o3d.io.write_triangle_mesh(sample_out_path, ref_warped_o3d)

                        ##########################################
                        # Also save t-pose if not already saved
                        ##########################################
                        ref_o3d = o3d.geometry.TriangleMesh(
                            o3d.utility.Vector3dVector(mesh_pred_trimesh.vertices),
                            o3d.utility.Vector3iVector(mesh_pred_trimesh.faces),
                        )
                        ref_o3d.compute_vertex_normals()
                        tpose_sample_out_dir = os.path.join(self.exp_dir, 'a_t_pose')
                        if not os.path.isdir(tpose_sample_out_dir): os.mkdir(tpose_sample_out_dir)
                        tpose_sample_out_path = os.path.join(tpose_sample_out_dir, "ref_reconstructed.ply")
                        o3d.io.write_triangle_mesh(tpose_sample_out_path, ref_o3d)

                # ------------------------------------------------------------------------------------------------

                # End loop over optimization iterations

            if self.cache_for_viz:
                self.refw_list.append(refw_optim)

            # REF (PRED)
            if identity_name not in self.ref_dict:
                if self.warp_reconstructed_mesh:
                    self.ref_dict[identity_name] = ref_optim
                else:
                    p_ref_flat_np = p_ref_flat.detach().cpu().numpy()
                    p_ref_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p_ref_flat_np))
                    p_ref_pcd.paint_uniform_color([1, 0, 0]) # ref is in red
                    p_ref_pcd.estimate_normals()
                    if self.cache_for_viz:
                        ref_optim.append(p_ref_pcd)
                        self.ref_dict[identity_name] = ref_optim

            # REF GT
            if p_ref_gt is not None:
                p_ref_gt_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p_ref_gt))
                p_ref_gt_pcd.paint_uniform_color([0, 1, 0]) # current is in green
                p_ref_gt_pcd.estimate_normals()
                self.ref_gt_dict[identity_name] = p_ref_gt_pcd

            # CUR GT
            p_cur_flat = p_cur_flat.detach().cpu().numpy()
            p_cur_pcd  = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p_cur_flat))
            p_cur_pcd.paint_uniform_color([0, 1, 0]) # current is in green
            p_cur_pcd.estimate_normals()
            if self.cache_for_viz:
                self.cur_list.append(p_cur_pcd)

            # identity_name by frame_i
            self.identity_name_by_loaded_frame[self.loaded_frames] = identity_name 

            ######################################################
            # CUR VOXELS
            inputs_path = os.path.join(cur_path, f'partial_views/voxelized_view0_{self.input_res}res.npz')
            occupancies = np.unpackbits(np.load(inputs_path)['compressed_occupancies'])
            input_voxels_np = np.reshape(occupancies, (self.input_res,)*3).astype(np.float32)
            voxels_trimesh = VoxelGrid(input_voxels_np).to_mesh()
            voxels_mesh = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(voxels_trimesh.vertices),
                o3d.utility.Vector3iVector(voxels_trimesh.faces)
            )
            voxels_mesh.compute_vertex_normals()
            voxels_mesh.paint_uniform_color([0.2, 1, 0.5])
            if self.cache_for_viz:
                self.vox_list.append(voxels_mesh)

            try:
                input_points_np = np.load(inputs_path)['point_cloud']
                p_partial_cur_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(input_points_np))
                p_partial_cur_pcd.paint_uniform_color([0.5, 1, 0.2]) # current is in green
                p_partial_cur_pcd.estimate_normals()
                if self.cache_for_viz:
                    self.pcd_list.append(p_partial_cur_pcd)
            except:
                pass
                
            ######################################################

            # Increase counter of evaluated frames
            self.loaded_frames += 1

            print(f'Loaded {self.loaded_frames} frames')

            if self.loaded_frames == self.num_to_eval:
                print()
                print(f"Stopping early. Already loaded {self.loaded_frames}")
                print()
                break

        print()
        print("#"*60)
        print("Num cur", len(self.cur_list))
        print("Num refw", len(self.refw_list))
        print()
        print("Initialized VizOptim")
        print("#"*60)
        print()

        ###############################################################################################
        # Generate additional meshes.
        ###############################################################################################
        # unit bbox
        p_min = -0.5
        p_max =  0.5
        self.unit_bbox = BBox.compute_bbox_from_min_point_and_max_point(
            np.array([p_min]*3), np.array([p_max]*3), color=[0.7, 0.7, 0.7]
        )
        self.unit_bbox = rotate_around_axis(self.unit_bbox, axis_name="x", angle=-np.pi) 

    def update_ref(self, vis):
        param = vis.get_view_control().convert_to_pinhole_camera_parameters()

        # We remove a mesh if it's currently stored.
        if self.ref is not None:
            vis.remove_geometry(self.ref)

        # If requested, we show a (new) mesh.
        if self.show_ref:
            self.ref = self.ref_dict[self.identity_name_by_loaded_frame[self.time]][self.iter]
            vis.add_geometry(self.ref)

        ctr = vis.get_view_control()
        ctr.convert_from_pinhole_camera_parameters(param)

    def update_ref_gt(self, vis):
        param = vis.get_view_control().convert_to_pinhole_camera_parameters()

        if self.ref_gt is not None:
            vis.remove_geometry(self.ref_gt)
        
        if self.show_ref_gt and len(self.ref_gt_dict) > 0:
            self.ref_gt = self.ref_gt_dict[self.identity_name_by_loaded_frame[self.time]]
            vis.add_geometry(self.ref_gt)

        ctr = vis.get_view_control()
        ctr.convert_from_pinhole_camera_parameters(param)

    def update_cur(self, vis):
        param = vis.get_view_control().convert_to_pinhole_camera_parameters()

        if self.cur is not None:
            vis.remove_geometry(self.cur)
        
        if self.show_cur:
            self.cur = self.cur_list[self.time]
            vis.add_geometry(self.cur)

        ctr = vis.get_view_control()
        ctr.convert_from_pinhole_camera_parameters(param)

    def update_refw(self, vis):
        param = vis.get_view_control().convert_to_pinhole_camera_parameters()

        if self.refw is not None:
            vis.remove_geometry(self.refw)
        
        if self.show_refw:
            self.refw = self.refw_list[self.time][self.iter]
            vis.add_geometry(self.refw)


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

    def run(self):
        # Define callbacks.
        def toggle_next(vis):
            self.time += 1
            if self.time >= self.loaded_frames:
                self.time = 0
            self.iter = self.num_codes_optim_video - 1
            
            identity_name = self.identity_name_by_loaded_frame[self.time]
            print(f"time {self.time} - iter {self.iter} - {identity_name}")

            self.update_ref_gt(vis)
            self.update_ref(vis)
            self.update_cur(vis)
            self.update_refw(vis)
            self.update_vox(vis)
            self.update_pcd(vis)
            return False
        
        def toggle_previous(vis):
            self.time -= 1
            if self.time < 0:
                self.time = self.loaded_frames - 1
            self.iter = self.num_codes_optim_video - 1
            
            identity_name = self.identity_name_by_loaded_frame[self.time]
            print(f"time {self.time} - iter {self.iter} - {identity_name}")

            self.update_ref_gt(vis)
            self.update_ref(vis)
            self.update_cur(vis)
            self.update_refw(vis)
            self.update_vox(vis)
            self.update_pcd(vis)
            return False

        def toggle_ref_gt(vis):
            self.show_ref_gt = not self.show_ref_gt
            self.update_ref_gt(vis)
            return False

        def toggle_ref(vis):
            self.show_ref = not self.show_ref
            self.update_ref(vis)
            return False

        def toggle_cur(vis):
            self.show_cur = not self.show_cur
            self.update_cur(vis)
            return False
        
        def toggle_refw(vis):
            self.show_refw = not self.show_refw
            self.update_refw(vis)
            return False

        def toggle_vox(vis):
            self.show_vox = not self.show_vox
            self.update_vox(vis)
            return False

        def toggle_pcd(vis):
            self.show_pcd = not self.show_pcd
            self.update_pcd(vis)
            return False

        def toggle_next_iter(vis):
            self.iter += 1
            if self.iter >= self.num_codes_optim_video:
                self.iter = 0
            identity_name = self.identity_name_by_loaded_frame[self.time]
            print(f"time {self.time} - iter {self.iter} - {identity_name}")
            self.update_ref(vis)
            self.update_refw(vis)
            return False
        
        def toggle_previous_iter(vis):
            self.iter -= 1
            if self.iter < 0:
                self.iter = self.num_codes_optim_video - 1
            identity_name = self.identity_name_by_loaded_frame[self.time]
            print(f"time {self.time} - iter {self.iter} - {identity_name}")
            self.update_ref(vis)
            self.update_refw(vis)
            return False
     
        key_to_callback = {}
        key_to_callback[ord("D")] = toggle_next
        key_to_callback[ord("A")] = toggle_previous
        key_to_callback[ord("T")] = toggle_ref_gt
        key_to_callback[ord("R")] = toggle_ref
        key_to_callback[ord("C")] = toggle_cur
        key_to_callback[ord("W")] = toggle_refw
        key_to_callback[ord("L")] = toggle_next_iter
        key_to_callback[ord("J")] = toggle_previous_iter
        key_to_callback[ord("P")] = toggle_pcd
        key_to_callback[ord("V")] = toggle_vox

        # Add mesh at initial time step.
        assert self.time < self.loaded_frames

        print("Showing time", self.time)

        self.refw = self.refw_list[self.time][self.iter]
        self.show_refw = True 

        self.cur = self.cur_list[self.time]
        self.show_cur = True

        o3d.visualization.draw_geometries_with_key_callbacks([self.unit_bbox, self.refw, self.cur], key_to_callback)
