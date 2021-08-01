import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import trimesh
import numpy as np
import glob

from multiprocessing import Pool

import argparse
import traceback

import utils.pcd_utils as pcd_utils
from data_scripts import config_data
import config as cfg


def boundary_sampling_flow(in_path):
    try:

        if "SPLITS" in in_path or in_path.endswith("json") or in_path.endswith("txt") or in_path.endswith("npz"):
            return

        out_file = os.path.join(in_path, 'flow_samples_{}.npz'.format(args.sigma))

        if not OVERWRITE and os.path.exists(out_file):
            print("Skipping", in_path)
            return

        # Load mesh
        mesh = trimesh.load(os.path.join(in_path, 'mesh_normalized.ply'), process=False, maintain_order=True)

        ############################################################################################################
        # Sample points on the mesh using the reference barycentric coordinates computed at the beginning
        ############################################################################################################

        if args.sigma > 0:
            # Along the normal
            points_source, points_triangles = pcd_utils.sample_points_give_bary(
                mesh, faces_ref, bary_coords_ref
            )

            normals, is_valid_normal = trimesh.triangles.normals(points_triangles)
            assert np.all(is_valid_normal), "Not all normals were valid"

            points = points_source + random_noise * normals

        else:
            # On the surface
            points, _ = pcd_utils.sample_points_give_bary(
                mesh, faces_ref, bary_coords_ref
            )

        np.savez(out_file, points=points)
        print('Done with {}'.format(out_file))

    except:
        print('\t------------ Error with {}: {}'.format(in_path, traceback.format_exc()))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Run boundary sampling'
    )
    parser.add_argument('-sigma', type=float, required=True)
    parser.add_argument('-t', '-max_threads', dest='max_threads', type=int, default=-1)
    args = parser.parse_args()

    try:
        n_jobs = int(os.environ['SLURM_CPUS_ON_NODE'])
        assert args.max_threads != 0
        if args.max_threads > 0:
            n_jobs = args.max_threads
    except:
        n_jobs = 1

    #####################################################################
    # Set up
    #####################################################################

    OVERWRITE = False
    use_stored_sampling = True
    only_aug = False

    sample_num = 100000

    # ----------------------------------------------------------------- #
    datasets_name = "datasets"
    # ----------------------------------------------------------------- #
    dataset = "cape"
    # ----------------------------------------------------------------- #

    print("DATASET:", dataset)
    #####################################################################

    dataset_dir = f'{cfg.ROOT}/{datasets_name}/{dataset}'
    
    #####################################################################
    # Characters
    #####################################################################
    character_names = [
        d for d in os.listdir(dataset_dir) 
        if "ZSPLITS" not in d
        and not d.endswith("json")
        and not d.endswith("txt")
    ]
    character_names = sorted(character_names)

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
    # For mano
    if "mano" in dataset:
        character_names = [c for c in character_names if "test" not in c]

    # For cape
    if "cape" in dataset:
        character_names = [c for c in character_names if c not in config_data.test_identities_cape]

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #

    # Print
    print()
    for c in character_names:
        print(c)
    
    input(f"Continue? {len(character_names)} characters")

    is_first = True

    #####################################################################
    # Process
    #####################################################################
    for character_name in character_names:

        print()
        print("character", character_name)

        ROOT_CHARACTER = f'{cfg.ROOT}/datasets/{dataset}/{character_name}'
            
        #####################################################################
        # Get reference sampling
        #####################################################################
        sampling_info_path = os.path.join(ROOT_CHARACTER, f"sampling_info_flow_{args.sigma}.npz")

        if not OVERWRITE and use_stored_sampling and os.path.exists(sampling_info_path):
            if is_first:
                input("Reading old sampling")
                is_first = False

            sampling_info_npz = np.load(sampling_info_path)
            faces_ref       = sampling_info_npz['faces_ref']
            bary_coords_ref = sampling_info_npz['bary_coords_ref']
            random_noise    = sampling_info_npz['random_noise']

        else:
            if is_first:
                input("Sampling new")
                is_first = False
                
            ref_shape_path = os.path.join(ROOT_CHARACTER, "a_t_pose", "000000")
            ref_mesh_path = os.path.join(ref_shape_path, "mesh_normalized.ply")
            ref_mesh = trimesh.load_mesh(ref_mesh_path, process=False, maintain_order=True)
            
            # Sample points in tpose, then use the barycentric coords to sample corresponding points in other meshes
            _, faces_ref, bary_coords_ref, _ = pcd_utils.sample_points(
                ref_mesh, sample_num, return_barycentric=True
            )

            # Random noise along the normals (has to be the same for all samples to get corresp)
            random_noise = 2.0 * np.random.rand(sample_num, 1) - 1.0 # [-1, 1]
            random_noise = args.sigma * random_noise
            
            # Store
            np.savez(
                sampling_info_path,
                faces_ref=faces_ref,
                bary_coords_ref=bary_coords_ref, 
                random_noise=random_noise, 
            )

        #####################################################################
        # Sample points
        #####################################################################

        try:
            p = Pool(n_jobs)
            p.map(boundary_sampling_flow, glob.glob(ROOT_CHARACTER + '/*/*/'))
        finally:
            p.close()
            p.join()
        