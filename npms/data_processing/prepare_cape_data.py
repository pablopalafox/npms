# Pablo Palafox 2021
# Loosely based off of cape_utils (https://github.com/qianlim/cape_utils/blob/master/dataset_utils.py) by qianlim (https://github.com/qianlim),
# which provides tools to interact with CAPE (https://cape.is.tue.mpg.de/dataset). Make sure to [register](https://cape.is.tue.mpg.de/en/sign_up)
# and that you agree to their terms before downloading their dataset. 

import os, sys

import numpy as np

import open3d as o3d
import trimesh
import tqdm
import pickle as pkl
from scipy.spatial import cKDTree as KDTree
from random import randrange

import torch # https://stackoverflow.com/questions/65710713/importerror-libc10-so-cannot-open-shared-object-file-no-such-file-or-director

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config as cfg
from body_model.smpl.smpl import Smpl


class Processor():

    def __init__(
        self, data_root_dir, processed_data_dir, specific_sample_paths
    ):
        # Raw data dirs
        self.data_root_dir = data_root_dir

        self.data_dir = os.path.join(self.data_root_dir, "sequences")
        self.scans_dir = os.path.join(self.data_root_dir, "raw_scans")
        self.minimal_body_shape_dir = os.path.join(self.data_root_dir, "minimal_body_shape")
        
        self.faces = np.load(os.path.join(self.data_root_dir, 'misc', 'smpl_tris.npy'))
        self.gender = pkl.load(open(os.path.join(self.data_root_dir, 'misc', 'subj_genders.pkl'), 'rb'))

        # For filtering out the real scans
        self.max_dist_scan_to_registr = 0.03

        # SMPL model
        self.body_model = Smpl(device='cuda:0')

        self.world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.1, origin=[0, 0, 0]
        )

        # Output dir
        self.processed_data_dir = processed_data_dir

        identity_dict = {}

        for identity_name in sorted(os.listdir(self.data_dir)):

            identity_dir = os.path.join(self.data_dir, identity_name)

            for cloth_animation_name in os.listdir(identity_dir):

                cloth_animation_name = cloth_animation_name.split('_')
                cloth_name, animation_name = cloth_animation_name[0], cloth_animation_name[1:]
                animation_name = '_'.join(animation_name)

                identity_cloth_name = f"{identity_name}_{cloth_name}"

                if identity_cloth_name not in identity_dict:
                    identity_dict[identity_cloth_name] = []
                identity_dict[identity_cloth_name].append(animation_name)

        self.identity_dict = identity_dict     

        # Labels (if they are None, we process everything)
        self.specific_sample_paths = specific_sample_paths
        self.only_specific_samples = True if len(specific_sample_paths) > 0 else False 
        
        print()
        print("Num identities:", len(self.identity_dict))
        # print(self.identity_dict)

        # Load the body meshes (in tpose) for each identity
        self.body_vertices_by_identity = {}
        for identity_id in os.listdir(self.minimal_body_shape_dir):
            minimal_body_shape_identity_dir = os.path.join(self.minimal_body_shape_dir, identity_id)
            minimal_body_shape_identity_path = os.path.join(minimal_body_shape_identity_dir, f"{identity_id}_minimal.ply")
            assert os.path.isfile(minimal_body_shape_identity_path)
            mesh = trimesh.load(minimal_body_shape_identity_path, process=False)
            self.body_vertices_by_identity[identity_id] = mesh.vertices

            # mesh_o3d = trimesh_to_open3d(mesh)
            # o3d.visualization.draw_geometries([mesh_o3d])
            
    def load_single_frame(self, npz_fn):
        '''
        given path to a single data frame, return the contents in the data
        '''
        try:
            data = np.load(npz_fn)
        except:
            return None

        try:
            return data['v_cano'], data['v_posed'], data['pose'], data['transl']
        except:
            return None

    def get_tpose_body_mesh(self, sample_path):
        result_load_single_frame = self.load_single_frame(sample_path)
        
        if result_load_single_frame is None:
            return None

        v_cano, _, _, _ = result_load_single_frame

        body_pose = torch.zeros((1, 69), dtype=torch.float32).cuda()
        body_pose[0, 2], body_pose[0, 5] = 0.5, -0.5 # open legs

        v_cano = torch.from_numpy(v_cano).cuda().type(torch.float32)

        body_vertices = self.body_model(
            body_pose=body_pose,
            t_pose=v_cano,
        ).cpu().numpy()

        tpose_mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(body_vertices), 
            o3d.utility.Vector3iVector(self.faces)
        )
        tpose_mesh.compute_vertex_normals()
        return tpose_mesh

    def get_posed_body_mesh(self, sample_path, identity_name):
        result_load_single_frame = self.load_single_frame(sample_path)

        if result_load_single_frame is None:
            return None

        v_cano, _, pose_params, trans = result_load_single_frame

        orient    = pose_params[:3]
        body_pose = pose_params[3:]

        orient    = torch.from_numpy(orient).cuda().type(torch.float32).unsqueeze(0)
        body_pose = torch.from_numpy(body_pose).cuda().type(torch.float32).unsqueeze(0)

        ##############################################################################
        # Clothed mesh
        ##############################################################################
        v_cano_clothed = torch.from_numpy(v_cano).cuda().type(torch.float32)
        clothed_body_vertices = self.body_model(
            orient=orient,
            body_pose=body_pose,
            t_pose=v_cano_clothed,
        ).cpu().numpy()
        clothed_posed_mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(clothed_body_vertices), 
            o3d.utility.Vector3iVector(self.faces)
        )
        clothed_posed_mesh.compute_vertex_normals()

        ##############################################################################
        # Body mesh
        ##############################################################################
        v_cano_body = self.body_vertices_by_identity[identity_name]
        v_cano_body = torch.from_numpy(v_cano_body).cuda().type(torch.float32)
        body_vertices = self.body_model(
            orient=orient,
            body_pose=body_pose,
            t_pose=v_cano_body,
        ).cpu().numpy()
        body_posed_mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(body_vertices), 
            o3d.utility.Vector3iVector(self.faces)
        )
        body_posed_mesh.compute_vertex_normals()

        if viz:
            o3d.visualization.draw_geometries([body_posed_mesh])
            o3d.visualization.draw_geometries([clothed_posed_mesh])

        return clothed_posed_mesh, body_posed_mesh, trans
    
    def filter_real_scan(self, real_scan_mesh, posed_mesh):
        real_scan_vertices  = np.array(real_scan_mesh.vertices)
        posed_mesh_vertices = np.array(posed_mesh.vertices)

        kdtree = KDTree(posed_mesh_vertices)
        dist, _ = kdtree.query(real_scan_vertices, k=1)
        
        invalid_vertices_mask = dist > self.max_dist_scan_to_registr
        real_scan_mesh.remove_vertices_by_mask(invalid_vertices_mask)
        real_scan_mesh.compute_vertex_normals()

        if False:
            posed_mesh.paint_uniform_color([1, 0, 0])
            o3d.visualization.draw_geometries([real_scan_mesh, posed_mesh])
            o3d.visualization.draw_geometries([real_scan_mesh])
        
        return real_scan_mesh
        
    def process(self, overwrite=False, compute_only_delta=False):

        print()
        print("Started processin...")
        print()

        real_scans_seq_computed = 0
        skipped_frames_due_to_loading_issue = 0

        ################################################################################################
        # Go over identities
        ################################################################################################
        for identity_cloth_name, animation_name_list in self.identity_dict.items():
            
            print("#"*60)
            print("Processing identity", identity_cloth_name)
            print("#"*60)

            identity_name, cloth_name = identity_cloth_name.split('_')

            # Create output dir for current identity_cloth
            output_identity_cloth_dir = os.path.join(self.processed_data_dir, identity_cloth_name)
            os.makedirs(output_identity_cloth_dir, exist_ok=True)

            requires_tpose = True

            # Go over the animations of a given identity
            for animation_name in sorted(animation_name_list):

                print("\tAnimation", animation_name)

                # Input animation dir
                animation_dir = os.path.join(self.data_dir, identity_name, f"{cloth_name}_{animation_name}")
                assert os.path.isdir(animation_dir), animation_dir

                # Real scans animation dir
                has_real_scans = False
                real_scans_animation_dir = os.path.join(self.scans_dir, identity_name, f"{cloth_name}_{animation_name}")
                if os.path.isdir(real_scans_animation_dir):
                    has_real_scans = True

                # Create output dir for current animation
                output_animation_dir = os.path.join(output_identity_cloth_dir, animation_name)
                os.makedirs(output_animation_dir, exist_ok=True)

                # List all the frames in the current animatioin
                frame_id_list = sorted(os.listdir(animation_dir))

                # ---------------------------------------------------------------------------------------------------- #
                # Create tpose if we haven't created it yet
                # ---------------------------------------------------------------------------------------------------- #
                if requires_tpose:
                    print("\t\tCreating tpose")
                    requires_tpose = False

                    # Load any sample
                    while True: 
                        random_idx = randrange(len(frame_id_list))
                        any_sample_path = os.path.join(animation_dir, frame_id_list[random_idx])

                        # Get tpose mesh
                        tpose_mesh = self.get_tpose_body_mesh(any_sample_path)

                        if viz:
                            o3d.visualization.draw_geometries([tpose_mesh])
                        
                        if tpose_mesh is not None:
                            break

                    # Save tpose
                    output_tpose_dir = os.path.join(output_identity_cloth_dir, "a_t_pose", "000000")
                    os.makedirs(output_tpose_dir, exist_ok=True)
                    tpose_mesh_path = os.path.join(output_tpose_dir, PROCESSED_MESH_FILENAME)
                    if overwrite or not os.path.isfile(tpose_mesh_path):
                        print("\t\tWriting", tpose_mesh_path)
                        o3d.io.write_triangle_mesh(tpose_mesh_path, tpose_mesh)
                    else:
                        print("\t\tNot overwriting", tpose_mesh_path)
                # ---------------------------------------------------------------------------------------------------- #
                    
                # ---------------------------------------------------------------------------------------------------- #
                # Go over all the frames in the current animation                 
                # ---------------------------------------------------------------------------------------------------- #
                for frame_id_name in tqdm.tqdm(frame_id_list):
                    frame_id = frame_id_name.split('.')[1]

                    # Output file
                    output_frame_dir = os.path.join(output_animation_dir, frame_id)

                    # If only specific samples, check this is one of those
                    if self.only_specific_samples and output_frame_dir not in self.specific_sample_paths:
                        # print(f"\t\t\tSkipping {output_frame_dir}, since not in list of specific samples...")
                        continue
                        
                    os.makedirs(output_frame_dir, exist_ok=True)
                    
                    clothed_posed_mesh_path = os.path.join(output_frame_dir, PROCESSED_MESH_FILENAME)
                    body_posed_mesh_path    = os.path.join(output_frame_dir, PROCESSED_MESH_BODY_FILENAME)
                    
                    # If files exist and we don't need to overwrite, continue
                    if not overwrite and os.path.isfile(clothed_posed_mesh_path) and os.path.isfile(body_posed_mesh_path):
                        continue

                    print(f"\t\t\tProcessing {output_frame_dir}")
                    
                    # Input npz
                    sample_path = os.path.join(animation_dir, frame_id_name)

                    # Get clothed and body meshes
                    result_get_posed_body_mesh = self.get_posed_body_mesh(sample_path, identity_name)
                    
                    # If we could not load the registered mesh, skip it and go to the next one
                    if result_get_posed_body_mesh is None:
                        print("Skipping", sample_path)
                        skipped_frames_due_to_loading_issue += 1
                        continue
                    
                    clothed_posed_mesh, body_posed_mesh, trans = result_get_posed_body_mesh
                    
                    # Save registered SMPLD mesh
                    if overwrite or not os.path.isfile(clothed_posed_mesh_path):
                        o3d.io.write_triangle_mesh(clothed_posed_mesh_path, clothed_posed_mesh)

                    # Save registered SMPL mesh
                    if overwrite or not os.path.isfile(body_posed_mesh_path):
                        o3d.io.write_triangle_mesh(body_posed_mesh_path, body_posed_mesh)

                    # Maybe save real scan as well is available
                    if has_real_scans:

                        real_scans_seq_computed += 1

                        output_real_scan_mesh_path = os.path.join(output_frame_dir, REAL_SCAN_MESH_FILENAME)
                        if not overwrite and os.path.isfile(output_real_scan_mesh_path):
                            continue

                        real_scan_sample_path = os.path.join(real_scans_animation_dir, os.path.splitext(frame_id_name)[0] + '.ply')
                        assert os.path.isfile(real_scan_sample_path), real_scan_sample_path

                        # Load the real scan, and translate it such that it is centered
                        real_scan_mesh = o3d.io.read_triangle_mesh(real_scan_sample_path)
                        real_scan_mesh.scale(1/1000., center=(0, 0, 0)) # scans are given in millimeters
                        real_scan_mesh.translate(-trans, relative=True)

                        #################################################################################################
                        if False:
                            real_scan_mesh.paint_uniform_color([1, 0, 0])
                            real_scan_mesh.compute_vertex_normals()
                            o3d.visualization.draw_geometries([self.world_frame, real_scan_mesh])
                        #################################################################################################

                        # Remove noise in input scan using the registered mesh as a mask
                        clean_real_scan_mesh = self.filter_real_scan(real_scan_mesh, clothed_posed_mesh)

                        #################################################################################################
                        if False:
                            posed_mesh.compute_vertex_normals()
                            clean_real_scan_mesh.paint_uniform_color([1, 0, 0])
                            clean_real_scan_mesh.compute_vertex_normals()

                            # o3d.visualization.draw_geometries([self.world_frame, clean_real_scan_mesh, posed_mesh])
                            o3d.visualization.draw_geometries([self.world_frame, clean_real_scan_mesh])
                        #################################################################################################

                        # Save the cleaned real scan
                        o3d.io.write_triangle_mesh(output_real_scan_mesh_path, real_scan_mesh)


        assert real_scans_seq_computed == 20, f"We should have computed 20 sequences with real scans, but we have {real_scans_seq_computed}."

        print()
        print(f"Skipped {skipped_frames_due_to_loading_issue} frames")

if __name__ == '__main__':

    OVERWRITE = False
    viz = False

    ################################################################################################
    # This is the name we'll give the processed dataset that we'll compute from the raw CAPE data
    PROCESSED_DATASET_NAME = "cape"
    ################################################################################################

    data_root = f"{cfg.ROOT}/data"
    datasets_root = f"{cfg.ROOT}/datasets"

    os.makedirs(datasets_root, exist_ok=True)

    ################################################################################################
    ## If dataset_name is not None, process only the samples in the file
    ################################################################################################
    specific_sample_paths = []
    
    # ----------------------------------------------------------------------------------------------
    dataset_name = None
    # dataset_name = "CAPE-POSE-TRAIN-35id-subsampled-10119ts"
    # ----------------------------------------------------------------------------------------------
    
    if dataset_name is not None:
        import json
        labels_json = f"{datasets_root}/ZSPLITS_HUMAN/{dataset_name}/labels.json"
        with open(labels_json, 'r') as f:
            labels = json.loads(f.read())

        for label in labels:
            sample_path = os.path.join(datasets_root, label['dataset'], label['identity_name'], label['animation_name'], label['sample_id'])
            specific_sample_paths.append(sample_path)

    ################################################################################################
    ################################################################################################

    # Data
    data_root_dir = os.path.join(data_root, "cape_release")

    # Dataset
    processed_data_dir = os.path.join(datasets_root, PROCESSED_DATASET_NAME)
    
    # Name for the meshes we're gonna create
    PROCESSED_MESH_FILENAME      = "mesh_raw.ply"
    PROCESSED_MESH_BODY_FILENAME = "mesh_body_raw.ply"
    REAL_SCAN_MESH_FILENAME      = "mesh_real_scan.ply"

    print("PROCESSED_DATASET_NAME", PROCESSED_DATASET_NAME)
    input("Continue?")
    os.makedirs(processed_data_dir, exist_ok=True)

    ################################################################################################
    ################################################################################################

    processor = Processor(
        data_root_dir, processed_data_dir, specific_sample_paths
    )
    
    # ################################################################################################
    # # TRAIN set
    # ################################################################################################
    processor.process(overwrite=OVERWRITE)
