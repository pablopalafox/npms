import sys, os
import json
import open3d as o3d
import numpy as np
from tqdm import tqdm


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.parsing_utils import get_dataset_type_from_dataset_name

#######################################################################################################

world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=0.5, origin=[0, 0, 0]
)

#######################################################################################################

ROOT = "/cluster_HDD/lothlann/ppalafox"
data_dir = f"{ROOT}/datasets_mix"

dataset_name = "POSE_amass_subsampled-train-20000ts"

dataset_type = get_dataset_type_from_dataset_name(dataset_name)
splits_dir = f"ZSPLITS_{dataset_type}"

labels_json       = os.path.join(data_dir, splits_dir, dataset_name, "labels.json")
print("Reading from:")
print(labels_json)

#######################################################################################################
# Data
#######################################################################################################
with open(labels_json, "r") as f:
    labels = json.loads(f.read())


for label in tqdm(labels):

    sample_dir = os.path.join(data_dir, label['dataset'], label['identity_name'], label['animation_name'], label['sample_id'])
    assert os.path.isdir(sample_dir), sample_dir

    params_path = os.path.join(sample_dir, 'params.npz')
    params_npz = np.load(params_path)
    pose = params_npz['pose']

    root_orient = pose[:3]
    y_rot = np.abs(root_orient[1])
    y_rot_deg = y_rot / np.pi * 180.


    if y_rot_deg < 0.1:

        print(y_rot_deg)
        print(label['dataset'], label['identity_name'], label['animation_name'], label['sample_id'])

        
        mesh_path = os.path.join(sample_dir, "mesh_normalized.ply")
        mesh_o3d = o3d.io.read_triangle_mesh(mesh_path)
        mesh_o3d.compute_vertex_normals()
        o3d.visualization.draw_geometries([mesh_o3d, world_frame])

