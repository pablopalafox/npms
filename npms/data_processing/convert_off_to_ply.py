import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import glob
import shutil
import trimesh
import numpy as np

import multiprocessing as mp
from multiprocessing import Pool

# from joblib import Parallel, delayed

import argparse
import glob
import traceback

from data_scripts import config_data as cfg


def remove_data(in_path):
    try:

        for in_mesh_filename in os.listdir(in_path):
            if not in_mesh_filename.endswith('.off'):
                continue

            print("Processing", in_path)

            # Read off
            in_mesh_path = os.path.join(in_path, in_mesh_filename)
            mesh = trimesh.load(in_mesh_path, process=False)
            vertices_A, faces_A = mesh.vertices, mesh.triangles
            
            # Export ply
            in_mesh_filename_raw = os.path.splitext(in_mesh_filename)[0]
            out_mesh_path = os.path.join(in_path, f"{in_mesh_filename_raw}.ply")
            mesh.export(out_mesh_path)

            # Load ply and assert
            mesh_B = trimesh.load(out_mesh_path, process=False)
            vertices_B, faces_B = mesh_B.vertices, mesh.triangles
            assert np.allclose(vertices_A, vertices_B)
            assert np.allclose(faces_A, faces_B)

            # Remove off
            os.remove(in_mesh_path)
            
    except:
        print('\t------------ Error with {}: {}'.format(in_path, traceback.format_exc()))


if __name__ == '__main__':

    #####################################################################
    # Set up
    #####################################################################

    try:
        n_jobs = int(os.environ['SLURM_CPUS_ON_NODE'])
    except:
        n_jobs = 1
        
    print()
    print(f"Using {n_jobs} ...")
    print()

    # character_names = [
    #     'ninja',
    #     # 'olivia',
    #     # 'racer',
    #     # 'crypto',
    #     # 'alien'
    # ]
    character_names = cfg.identities + cfg.identities_augmented

    dataset_type = "datasets_multi"

    for character_name in character_names:

        ROOT = f'/cluster/lothlann/ppalafox/{dataset_type}/mixamo/{character_name}'

        #####################################################################
        # Copy data
        #####################################################################

        # DEBUG ONE
        # remove_data(os.path.join(ROOT, "a_t_pose", "000000"))
        # exit()
        # DEBUG ONE
        
        try:
            p = Pool(n_jobs)
            p.map(remove_data, glob.glob(ROOT + '/*/*'))
        finally:
            p.close()
            p.join()