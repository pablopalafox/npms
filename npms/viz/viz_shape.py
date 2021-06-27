import os
import numpy as np
import trimesh
import open3d as o3d
from tqdm import tqdm

import utils.deepsdf_utils as deepsdf_utils 
from utils.evaluation import eval_mesh

from utils.pcd_utils import (BBox,
                            rotate_around_axis)


class ViewerShape:
    def __init__(
        self, 
        labels, data_dir,
        shape_decoder, shape_codes,
        num_to_eval=-1,
        load_every=None,
        from_frame_id=0,
        res=256,
        max_batch=32**3,
    ):
        self.res = res
        self.max_batch = max_batch

        self.labels = labels
        self.data_dir = data_dir

        self.shape_decoder = shape_decoder
        self.shape_codes = shape_codes
        self.shape_codes_dim = self.shape_codes.shape[-1]

        self.num_to_eval = num_to_eval
        self.load_every = load_every
        self.from_frame_id = from_frame_id

        self.time = 0
        
        self.gt_ref = None
        self.gt_cur = None
        self.pred = None
        self.pcd_IN  = None
        self.pcd_OUT = None

        self.show_gt_ref = False
        self.show_gt_cur = False
        self.show_pred = False
        self.show_pcds = False

        self.initialize()
    
    def initialize(self):

        print("Initializing viewer...")
        print()

        self.gt_ref_list = []
        self.gt_cur_list = []
        self.pred_list = []
        self.pcds_IN_list  = []
        self.pcds_OUT_list = []

        self.names_list = []

        self.loaded_frames = 0

        self.iou = []
        self.chamfer = []
        self.normal_consis = []

        # Go over sequence and compute losses 
        for frame_i in tqdm(range(len(self.labels))):

            if self.load_every and frame_i % self.load_every != 0:
                continue

            if frame_i < self.from_frame_id:
                continue
            
            label = self.labels[frame_i]

            dataset, identity_id, identity_name, animation_name, sample_id = \
                label['dataset'], label['identity_id'], label['identity_name'], label['animation_name'], label['sample_id']

            print(dataset, identity_id, identity_name, animation_name, sample_id)

            ref_path = os.path.join(self.data_dir, dataset, identity_name, "a_t_pose", "000000")
            cur_path = os.path.join(self.data_dir, dataset, identity_name, animation_name, sample_id)

            self.names_list.append(identity_name)

            ########################################################################
            # Pred
            ########################################################################
            # mesh_pred_trimesh = inference_utils.reconstruct_shape_sdf(
            #     reconstruct_resolution, batch_points,
            #     shape_decoder, shape_codes, identity_ids=[identity_id], shape_codes_dim=shape_codes_dim
            # )
            # mesh_pred_trimesh = deepsdf_utils.create_mesh_ours_supersampling(
            #     shape_decoder, shape_codes, identity_ids=[identity_id], shape_codes_dim=shape_codes_dim
            # )
            # mesh_pred_trimesh = deepsdf_utils.create_mesh_ours(
            #     shape_decoder, shape_codes, identity_ids=[identity_id], shape_codes_dim=shape_codes_dim
            # )

            mesh_pred_trimesh = deepsdf_utils.create_mesh(
                self.shape_decoder, self.shape_codes, 
                identity_ids=[identity_id], shape_codes_dim=self.shape_codes_dim,
                N=self.res, max_batch=self.max_batch
            )

            if mesh_pred_trimesh.vertices.shape[0] == 0:
                print("Failed to reconstruct")
                exit()

            # mesh_pred_trimesh = mesh_pred_trimesh.detach().cpu().squeeze(0)
            # print(mesh_pred_trimesh.shape)
            # print(mesh_pred_trimesh_B.shape)

            # print(mesh_pred_trimesh[-10:-1], mesh_pred_trimesh_B[-10:-1])
            # print(torch.eq(mesh_pred_trimesh[-2], mesh_pred_trimesh_B[-2]))

            # print(torch.all(torch.isclose(mesh_pred_trimesh.squeeze(0), mesh_pred_trimesh_B)))
            
            # exit()

            mesh_pred = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(mesh_pred_trimesh.vertices),
                o3d.utility.Vector3iVector(mesh_pred_trimesh.faces)
            )
            mesh_pred.compute_vertex_normals()
            mesh_pred.paint_uniform_color([0, 0, 1])
            
            self.pred_list.append(mesh_pred)

            ########################################################################
            # GT cur
            ########################################################################
            gt_ref_mesh_path = os.path.join(cur_path, "mesh_normalized.ply")
            mesh_gt_cur = o3d.io.read_triangle_mesh(gt_ref_mesh_path)
            mesh_gt_cur.compute_vertex_normals()
            mesh_gt_cur.paint_uniform_color([1, 0, 0])
            self.gt_cur_list.append(mesh_gt_cur)

            ########################################################################
            # GT ref
            ########################################################################
            gt_ref_mesh_path = os.path.join(ref_path, "mesh_normalized.ply")
            if os.path.isfile(gt_ref_mesh_path):
                mesh_gt_ref_trimesh = trimesh.load_mesh(gt_ref_mesh_path, process=False)
                mesh_gt_ref = o3d.io.read_triangle_mesh(gt_ref_mesh_path)
                mesh_gt_ref.compute_vertex_normals()
                mesh_gt_ref.paint_uniform_color([0, 1, 0])
                self.gt_ref_list.append(mesh_gt_ref)

                # Metrics
                print()
                print("NOTE!!")
                print("Make sure you use the watertight gt mesh for final metrics!")
                print("NOTE!")
                print()
                eval_res = eval_mesh(mesh_pred_trimesh, mesh_gt_ref_trimesh, -0.5, 0.5)
            
                self.iou.append(eval_res['iou'])
                self.chamfer.append(eval_res['chamfer_l2'])
                self.normal_consis.append(eval_res['normals'])

            ##############################################################################################################
            # 0.01
            ##############################################################################################################
            # sigma = 0.01
            # points_001_path = os.path.join(cur_path, f'boundary_samples_{sigma}.npz')
            # points_001_path_dict = np.load(points_001_path)
            # pts_001 = points_001_path_dict['points']
            # occ_001 = points_001_path_dict['occupancies']
            # # IN
            # pts_001_IN = pts_001[occ_001 == True]
            # pts_001_IN_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_001_IN))
            # pts_001_IN_pcd.paint_uniform_color([0, 0.4, 1])
            # self.pcds_IN_list.append(pts_001_IN_pcd)
            # # OUT
            # pts_001_OUT = pts_001[occ_001 == False]
            # points_001_pcd_OUT = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_001_OUT))
            # points_001_pcd_OUT.paint_uniform_color([1, 0, 0])
            # self.pcds_OUT_list.append(points_001_pcd_OUT)

            # Increase counter of evaluated frames
            self.loaded_frames += 1

            print(f'Loaded {self.loaded_frames} frames')

            if self.loaded_frames == self.num_to_eval:
                print()
                print(f"Stopping early. Already loaded {self.loaded_frames}")
                print()
                break

        if len(self.iou) > 0:
            print()
            print("Num identities", len(self.names_list))
            print("iou          ", sum(self.iou) / len(self.iou))
            print("chamfer      ", sum(self.chamfer) / len(self.chamfer))
            print("normal_consis", sum(self.normal_consis) / len(self.normal_consis))
            print()

        ###############################################################################################
        # Generate additional meshes.
        ###############################################################################################
        # unit bbox
        p_min = -0.5
        p_max =  0.5
        self.unit_bbox = BBox.compute_bbox_from_min_point_and_max_point(
            np.array([p_min]*3), np.array([p_max]*3)
        )
        self.unit_bbox = rotate_around_axis(self.unit_bbox, axis_name="x", angle=-np.pi) 

    def update_gt(self, vis):
        param = vis.get_view_control().convert_to_pinhole_camera_parameters()

        if self.gt_cur is not None:
            vis.remove_geometry(self.gt_cur)

        if self.show_gt_cur and len(self.gt_cur_list) > 0:
            self.gt_cur = self.gt_cur_list[self.time]
            vis.add_geometry(self.gt_cur)

        if self.gt_ref is not None:
            vis.remove_geometry(self.gt_ref)

        if self.show_gt_ref and len(self.gt_ref_list) > 0:
            self.gt_ref = self.gt_ref_list[self.time]
            vis.add_geometry(self.gt_ref)

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

    def update_pcd(self, vis):
        param = vis.get_view_control().convert_to_pinhole_camera_parameters()

        if self.pcd_IN is not None:
            vis.remove_geometry(self.pcd_IN)
            self.pcd_IN  = None

        if self.pcd_OUT is not None:
            vis.remove_geometry(self.pcd_OUT)
            self.pcd_OUT = None

        # If requested, we show a (new) mesh.
        if self.show_pcds and len(self.pcds_IN_list) > 0 and len(self.pcds_OUT_list) > 0:
            
            self.pcd_IN = self.pcds_IN_list[self.time]
            vis.add_geometry(self.pcd_IN)

            self.pcd_OUT = self.pcds_OUT_list[self.time]
            vis.add_geometry(self.pcd_OUT)

        ctr = vis.get_view_control()
        ctr.convert_from_pinhole_camera_parameters(param)

    def run(self):
        # Define callbacks.
        def toggle_next(vis):
            self.time += 1
            if self.time >= self.loaded_frames:
                self.time = 0
            print(self.time, self.names_list[self.time])
            self.update_gt(vis)
            self.update_pred(vis)
            self.update_pcd(vis)
            return False
        
        def toggle_previous(vis):
            self.time -= 1
            if self.time < 0:
                self.time = self.loaded_frames - 1
            print(self.time, self.names_list[self.time])
            self.update_gt(vis)
            self.update_pred(vis)
            self.update_pcd(vis)
            return False

        def toggle_gt_ref(vis):
            self.show_gt_ref = not self.show_gt_ref
            self.update_gt(vis)
            return False

        def toggle_gt_cur(vis):
            self.show_gt_cur = not self.show_gt_cur
            self.update_gt(vis)
            return False

        def toggle_pred(vis):
            self.show_pred = not self.show_pred
            self.update_pred(vis)
            return False
        
        def toggle_pcd(vis):
            self.show_pcds = not self.show_pcds
            self.update_pcd(vis)
            return False

        key_to_callback = {}
        key_to_callback[ord("D")] = toggle_next
        key_to_callback[ord("A")] = toggle_previous
        key_to_callback[ord("V")] = toggle_gt_ref
        key_to_callback[ord("Z")] = toggle_gt_cur
        key_to_callback[ord("X")] = toggle_pred
        key_to_callback[ord("C")] = toggle_pcd

        # Add mesh at initial time step.
        assert self.time < self.loaded_frames

        print("Showing time", self.time)

        self.pred = self.pred_list[self.time]
        self.show_pred= True 

        o3d.visualization.draw_geometries_with_key_callbacks([self.unit_bbox, self.pred], key_to_callback)