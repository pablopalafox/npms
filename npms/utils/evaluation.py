import numpy as np
from pykdtree.kdtree import KDTree
from data_processing.implicit_waterproofing import implicit_waterproofing

# mostly apdopted from occupancy_networks/im2mesh/common.py and occupancy_networks/im2mesh/eval.py

def eval_mesh(mesh_pred, mesh_gt, bb_min, bb_max, n_points=100000):

    while True:
        try:
            pointcloud_pred, idx = mesh_pred.sample(n_points, return_index=True)
        except IndexError:
            n_points = int(n_points / 2)
        else:
            break

    pointcloud_pred = pointcloud_pred.astype(np.float32)
    normals_pred = mesh_pred.face_normals[idx]

    pointcloud_gt, idx = mesh_gt.sample(n_points, return_index=True)
    pointcloud_gt = pointcloud_gt.astype(np.float32)
    normals_gt = mesh_gt.face_normals[idx]

    assert pointcloud_pred.shape == pointcloud_gt.shape, f"{pointcloud_pred.shape} vs {pointcloud_gt.shape}"

    out_dict = eval_pointcloud(pointcloud_pred, pointcloud_gt, normals_pred, normals_gt)


    bb_len = bb_max - bb_min
    bb_samples = np.random.rand(n_points*10, 3) * bb_len + bb_min

    occ_pred = implicit_waterproofing(mesh_pred, bb_samples)[0]
    occ_gt = implicit_waterproofing(mesh_gt, bb_samples)[0]

    area_union = (occ_pred | occ_gt).astype(np.float32).sum()
    area_intersect = (occ_pred & occ_gt).astype(np.float32).sum()

    out_dict['iou'] =  (area_intersect / area_union)

    return out_dict


def eval_pointcloud(pointcloud_pred, pointcloud_gt,
                    normals_pred=None, normals_gt=None):

    pointcloud_pred = np.asarray(pointcloud_pred)
    pointcloud_gt = np.asarray(pointcloud_gt)

    # Completeness: how far are the points of the target point cloud
    # from the predicted point cloud
    completeness, completeness_normals = distance_p2p(
        pointcloud_gt, pointcloud_pred,
        normals_gt, normals_pred
    )
    completeness2 = completeness ** 2
    
    # Accuracy: how far are th points of the predicted pointcloud
    # from the target pointcloud
    accuracy, accuracy_normals = distance_p2p(
        pointcloud_pred, pointcloud_gt,
        normals_pred, normals_gt
    )
    accuracy2 = accuracy**2

    completeness = completeness.mean()
    completeness2 = completeness2.mean()

    accuracy = accuracy.mean()
    accuracy2 = accuracy2.mean()

    # Chamfer distance
    chamfer_l2 = 0.5 * completeness2 + 0.5 * accuracy2

    if not normals_pred is None:
        accuracy_normals     = accuracy_normals.mean()
        completeness_normals = completeness_normals.mean()
        normals_correctness = (
            0.5 * completeness_normals + 0.5 * accuracy_normals
        )
    else:
        accuracy_normals = np.nan
        completeness_normals = np.nan
        normals_correctness = np.nan


    out_dict = {
        'completeness': completeness,
        'accuracy': accuracy,
        'normals completeness': completeness_normals,
        'normals accuracy': accuracy_normals,
        'normals': normals_correctness,
        'completeness2': completeness2,
        'accuracy2': accuracy2,
        'chamfer_l2': chamfer_l2,
        'iou': np.nan
    }

    return out_dict


def distance_p2p(pointcloud_pred, pointcloud_gt,
                    normals_pred, normals_gt):
    ''' Computes minimal distances of each point in points_src to points_tgt.
    Args:
        points_src (numpy array): source points
        normals_src (numpy array): source normals
        points_tgt (numpy array): target points
        normals_tgt (numpy array): target normals
    '''
    kdtree = KDTree(pointcloud_gt)
    dist, idx = kdtree.query(pointcloud_pred)

    if normals_pred is None:
        return dist, None

    normals_pred = normals_pred / np.linalg.norm(normals_pred, axis=-1, keepdims=True)
    normals_gt = normals_gt / np.linalg.norm(normals_gt, axis=-1, keepdims=True)

    normals_dot_product = (normals_gt[idx] * normals_pred).sum(axis=-1)
    # Handle normals that point into wrong direction gracefully
    # (mostly due to mehtod not caring about this in generation)
    normals_dot_product = np.abs(normals_dot_product)

    return dist, normals_dot_product


# def compute_chamfer_per_vertex(mesh_pred, mesh_gt, n_points=100000):

#     mesh_vertices_pred = mesh_pred.vertices.astype(np.float32)

#     while True:
#         try:
#             pointcloud_pred = mesh_pred.sample(n_points)
#         except IndexError:
#             n_points = int(n_points / 2)
#         else:
#             break
#     pointcloud_pred = pointcloud_pred.astype(np.float32)

#     pointcloud_gt = mesh_gt.sample(n_points)
#     pointcloud_gt = pointcloud_gt.astype(np.float32)

#     assert pointcloud_pred.shape == pointcloud_gt.shape, f"{pointcloud_pred.shape} vs {pointcloud_gt.shape}"

#     # Completeness: how far are the points of the target point cloud
#     # from the predicted point cloud
#     completeness, _ = distance_p2p(
#         pointcloud_gt, pointcloud_pred,
#         None, None
#     )
#     completeness2 = completeness ** 2
    
#     # Accuracy: how far are the points of the predicted pointcloud
#     # from the target pointcloud
#     accuracy, _ = distance_p2p(
#         pointcloud_pred, pointcloud_gt,
#         None, None
#     )
#     accuracy2 = accuracy**2

#     # chamfer = 0.5 * completeness2 + 0.5 * accuracy2
#     chamfer = completeness

#     # For each vertex, find the nearest sampled point and assign that vertex the chamfer of the sample
#     kdtree = KDTree(pointcloud_pred)
#     _, idx = kdtree.query(mesh_vertices_pred)

#     chamfer_per_vertex = chamfer[idx]

#     return chamfer_per_vertex

def compute_accuracy_per_predicted_vertex(mesh_pred, mesh_gt, n_points=100000):

    pointcloud_pred = mesh_pred.vertices
    pointcloud_pred = pointcloud_pred.astype(np.float32)

    pointcloud_gt = mesh_gt.sample(n_points)
    pointcloud_gt = pointcloud_gt.astype(np.float32)

    # assert pointcloud_pred.shape == pointcloud_gt.shape, f"{pointcloud_pred.shape} vs {pointcloud_gt.shape}"

    accuracy, _ = distance_p2p(
        pointcloud_pred, pointcloud_gt,
        None, None
    )

    return accuracy

def compute_completeness_per_gt_vertex(mesh_pred, mesh_gt, n_points=100000):

    pointcloud_pred = mesh_pred.sample(n_points)
    pointcloud_pred = pointcloud_pred.astype(np.float32)

    pointcloud_gt = mesh_gt.vertices
    pointcloud_gt = pointcloud_gt.astype(np.float32)

    # assert pointcloud_pred.shape == pointcloud_gt.shape, f"{pointcloud_pred.shape} vs {pointcloud_gt.shape}"

    completeness, _ = distance_p2p(
        pointcloud_gt, pointcloud_pred,
        None, None
    )

    return completeness