import os
import numpy as np
import trimesh
import open3d as o3d
import json
from tqdm import tqdm
import math
import csv
import glob
from utils.utils import query_yes_no
from scipy.spatial import cKDTree as KDTree
# from sklearn.neighbors import KDTree

import utils.deepsdf_utils as deepsdf_utils 
from utils.evaluation import eval_mesh
import utils.pcd_utils as pcd_utils

from utils.pcd_utils import (BBox,
                            transform_pointcloud_to_opengl_coords,
                            rotate_around_axis,
                            origin, normalize_transformation)

import config as cfg


def compute_reconstruction_error(
    data_base_dir, labels, predicted_meshes_dir,
    viz=False,
    start_from_frame=0,
    end_before_frame=-1,
):

    iou = []
    chamfer = []
    accuracy = []
    completeness = []
    normal_consis = []

    ########################################################################################
    ########################################################################################
    # Go over all frames in the sequence
    ########################################################################################
    ########################################################################################
    for frame_t, label in enumerate(tqdm(labels)):

        if frame_t < start_from_frame:
            continue

        if frame_t == end_before_frame:
            break

        ############################################################
        # Load groundtruth vertices of current frame.
        ############################################################
        label = labels[frame_t]

        gt_dir = os.path.join(data_base_dir, label['dataset'], label['identity_name'], label['animation_name'], label['sample_id'])
        assert os.path.isdir(gt_dir), gt_dir
        gt_mesh_path = os.path.join(gt_dir, "mesh_normalized.ply")
        gt_mesh_trimesh = trimesh.load(gt_mesh_path, process=False)
        if viz:
            gt_mesh_o3d = o3d.io.read_triangle_mesh(gt_mesh_path)
            gt_mesh_o3d.paint_uniform_color([0, 1, 0])
            gt_mesh_o3d.compute_vertex_normals()

        ############################################################
        # Load predicted vertices of current keyframe
        ############################################################
        frame_dir = os.path.join(predicted_meshes_dir, label['sample_id'])
        assert os.path.isdir(frame_dir), frame_dir
        mesh_path = os.path.join(frame_dir, "ref_warped.ply")
        mesh_trimesh = trimesh.load(mesh_path, process=False)
        if viz:
            mesh_o3d = o3d.io.read_triangle_mesh(mesh_path)
            mesh_o3d.paint_uniform_color([1, 0, 0])
            mesh_o3d.compute_vertex_normals()

        ############################################################
        # Compute metrics
        ############################################################
        eval_res = eval_mesh(gt_mesh_trimesh, mesh_trimesh, -0.5, 0.5)
            
        iou.append(eval_res['iou'])
        chamfer.append(eval_res['chamfer_l2'])
        accuracy.append(eval_res['accuracy'])
        completeness.append(eval_res['completeness'])
        normal_consis.append(eval_res['normals'])

    # Compute average metrics.
    iou_avg           = np.mean(np.array(iou))
    chamfer_avg       = np.mean(np.array(chamfer))
    accuracy_avg      = np.mean(np.array(accuracy))
    completeness_avg  = np.mean(np.array(completeness))
    normal_consis_avg = np.mean(np.array(normal_consis))

    return iou_avg, chamfer_avg, accuracy_avg, completeness_avg, normal_consis_avg


"""
Tracking error
"""
def compute_tracking_error(
    data_base_dir, labels, predicted_meshes_dir,
    sample_num, knn, coverage,
    keyframe_every=10,
    debug=False, viz=False,
    start_from_frame=0,
    end_before_frame=-1,
):
    print()
    print("#"*120)
    print("Computing tracking error...")
    print("#"*120)
    print()

    num_gt_meshes = len(labels)

    num_keyframes = int(num_gt_meshes / keyframe_every)

    keyframe_inc = math.floor(float(num_gt_meshes) / num_keyframes) 

    print()
    print(f"Num keyframes {num_keyframes} (inc: {keyframe_inc})")
    print()

    # For densely sampling points on the gt and pred meshes
    gt_faces, gt_bary_coords     = None, None
    pred_faces, pred_bary_coords = None, None

    epe3d_array = []

    ########################################################################################
    ########################################################################################
    # Go over the (groundtruth) keyframes
    ########################################################################################
    ########################################################################################
    for keyframe_idx in range(num_keyframes):
        keyframe_t = keyframe_idx * keyframe_inc

        if keyframe_t < start_from_frame:
            continue

        if keyframe_t == end_before_frame:
            continue

        print()
        print("#"*60)
        print("Keyframe time: {}".format(keyframe_t))
        print("#"*60)

        ############################################################
        # Load groundtruth vertices of current keyframe.
        ############################################################
        label = labels[keyframe_t]

        gt_dir = os.path.join(data_base_dir, label['dataset'], label['identity_name'], label['animation_name'], label['sample_id'])
        assert os.path.isdir(gt_dir), gt_dir
        gt_mesh_path = os.path.join(gt_dir, "mesh_normalized.ply")
        gt_mesh_trimesh = trimesh.load(gt_mesh_path, process=False)
        # if viz:
        #     gt_mesh_o3d = o3d.io.read_triangle_mesh(gt_mesh_path)
        #     gt_mesh_o3d.paint_uniform_color([0, 1, 0])
        #     gt_mesh_o3d.compute_vertex_normals()

        # ----------------------------------------
        # Sample points in mesh more densely
        # ----------------------------------------
        if gt_faces is None:
            gt_vertices, gt_faces, gt_bary_coords, _ = pcd_utils.sample_points(
                gt_mesh_trimesh, sample_num, return_barycentric=True
            )
            gt_vertices_ref = gt_vertices.astype(np.float32)
        else:
            gt_vertices, _ = pcd_utils.sample_points_give_bary(gt_mesh_trimesh, gt_faces, gt_bary_coords)
        gt_vertices = gt_vertices.astype(np.float32)

        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        if False:
            gt_vertices_ref_o3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gt_vertices_ref))
            gt_vertices_ref_o3d.paint_uniform_color([1, 1, 0])

            gt_vertices_o3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gt_vertices))
            gt_vertices_o3d.paint_uniform_color([0, 1, 0])

            corresp = [(k, k) for k in range(0, gt_vertices_ref.shape[0])]
            lines = o3d.geometry.LineSet.create_from_point_cloud_correspondences(gt_vertices_ref_o3d, gt_vertices_o3d, corresp)
            lines.paint_uniform_color([0.2, 0.8, 0.8])
            
            o3d.visualization.draw_geometries([unit_bbox, gt_vertices_ref_o3d, gt_vertices_o3d, lines])
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        ############################################################
        # Load predicted vertices of current keyframe
        ############################################################
        frame_dir = os.path.join(predicted_meshes_dir, label['sample_id'])
        assert os.path.isdir(frame_dir), frame_dir
        mesh_path = os.path.join(frame_dir, "ref_warped.ply")
        mesh_trimesh = trimesh.load(mesh_path, process=False)
        # if viz:
        #     mesh_o3d = o3d.io.read_triangle_mesh(mesh_path)
        #     mesh_o3d.paint_uniform_color([1, 0, 0])
        #     mesh_o3d.compute_vertex_normals()

        # # debug
        if debug: mesh_trimesh = gt_mesh_trimesh
        # # debug

        # ----------------------------------------
        # Sample points in mesh more densely
        # ----------------------------------------
        if pred_faces is None:
            pred_vertices, pred_faces, pred_bary_coords, _ = pcd_utils.sample_points(
                mesh_trimesh, sample_num, return_barycentric=True
            )
        else:
            pred_vertices, _ = pcd_utils.sample_points_give_bary(mesh_trimesh, pred_faces, pred_bary_coords)
        pred_vertices = pred_vertices.astype(np.float32)

        if viz:
            o3d.visualization.draw_geometries([unit_bbox, gt_mesh_o3d, mesh_o3d])
        
        ############################################################
        # Compute mapping from gt vertices to pred vertices
        ############################################################
        kdtree = KDTree(pred_vertices)
        dist, ind = kdtree.query(gt_vertices, k=knn)
        if len(dist.shape) == 1: dist, ind = dist[..., None], ind[..., None]
        # Compute weights from distances
        weights = np.exp(- dist*dist / (2*coverage*coverage))
        # Normalize weights
        sum_weights = np.sum(weights, axis=-1, keepdims=True)
        if not np.all(sum_weights != 0.0):
            continue

        weights = weights / sum_weights
        # Replicate across point dimension
        weights = weights[..., None]
        weights = np.repeat(weights, 3, axis=-1)

        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        if False:
            # GT vertices
            gt_vertices_o3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gt_vertices))
            gt_vertices_o3d.paint_uniform_color([0, 1, 0])

            # PRED vertices
            interp_pred_vertices = pred_vertices[ind]
            interp_pred_vertices = interp_pred_vertices * weights
            interp_pred_vertices = np.sum(interp_pred_vertices, axis=1)

            interp_pred_vertices_o3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(interp_pred_vertices))
            interp_pred_vertices_o3d.paint_uniform_color([1, 0, 1])

            corresp = [(k, k) for k in range(0, gt_vertices.shape[0])]
            lines = o3d.geometry.LineSet.create_from_point_cloud_correspondences(gt_vertices_o3d, interp_pred_vertices_o3d, corresp)
            lines.paint_uniform_color([0.2, 0.8, 0.8])

            o3d.visualization.draw_geometries([unit_bbox, gt_vertices_o3d, interp_pred_vertices_o3d, lines])
            # exit()
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        ########################################################################################
        ########################################################################################
        # Now go over all frames in the sequence
        ########################################################################################
        ########################################################################################
        for frame_t, label in enumerate(tqdm(labels)):
            if frame_t == keyframe_t:
                continue 

            if frame_t < start_from_frame:
                continue

            if frame_t == end_before_frame:
                continue

            # print(f"{keyframe_t}-{frame_t}")

            ############################################################
            # Load groundtruth vertices of current frame.
            ############################################################
            label = labels[frame_t]

            gt_dir = os.path.join(data_base_dir, label['dataset'], label['identity_name'], label['animation_name'], label['sample_id'])
            assert os.path.isdir(gt_dir), gt_dir
            gt_mesh_path = os.path.join(gt_dir, "mesh_normalized.ply")
            gt_mesh_trimesh = trimesh.load(gt_mesh_path, process=False)
            if viz:
                gt_mesh_o3d = o3d.io.read_triangle_mesh(gt_mesh_path)
                gt_mesh_o3d.paint_uniform_color([0, 1, 0])
                gt_mesh_o3d.compute_vertex_normals()

            gt_vertices, _ = pcd_utils.sample_points_give_bary(gt_mesh_trimesh, gt_faces, gt_bary_coords)
            gt_vertices = gt_vertices.astype(np.float32)

            ############################################################
            # Load predicted vertices of current keyframe
            ############################################################
            frame_dir = os.path.join(predicted_meshes_dir, label['sample_id'])
            assert os.path.isdir(frame_dir), frame_dir
            mesh_path = os.path.join(frame_dir, "ref_warped.ply")
            mesh_trimesh = trimesh.load(mesh_path, process=False)
            if viz:
                mesh_o3d = o3d.io.read_triangle_mesh(mesh_path)
                mesh_o3d.paint_uniform_color([1, 0, 0])
                mesh_o3d.compute_vertex_normals()

            # # debug
            if debug: mesh_trimesh = gt_mesh_trimesh
            # # debug

            pred_vertices, _ = pcd_utils.sample_points_give_bary(mesh_trimesh, pred_faces, pred_bary_coords)
            pred_vertices = pred_vertices.astype(np.float32)
            
            ############################################################
            # Interpolate predicted vertices
            ############################################################
            interp_pred_vertices = pred_vertices[ind]
            interp_pred_vertices = interp_pred_vertices * weights
            interp_pred_vertices = np.sum(interp_pred_vertices, axis=1)

            # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            if False:
                # GT vertices
                gt_vertices_o3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gt_vertices))
                gt_vertices_o3d.paint_uniform_color([0, 1, 0])

                # PRED vertices
                pred_vertices_o3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pred_vertices))
                pred_vertices_o3d.paint_uniform_color([1, 0, 1])

                # PRED interpolated vertices
                interp_pred_vertices_o3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(interp_pred_vertices))
                interp_pred_vertices_o3d.paint_uniform_color([1, 0, 0])

                corresp = [(k, k) for k in range(0, gt_vertices.shape[0])]
                lines = o3d.geometry.LineSet.create_from_point_cloud_correspondences(gt_vertices_o3d, interp_pred_vertices_o3d, corresp)
                lines.paint_uniform_color([0.2, 0.8, 0.8])

                # o3d.visualization.draw_geometries([unit_bbox, gt_vertices_o3d, pred_vertices_o3d])
                o3d.visualization.draw_geometries([unit_bbox, gt_vertices_o3d, pred_vertices_o3d, interp_pred_vertices_o3d, lines])
                # o3d.visualization.draw_geometries([unit_bbox, gt_vertices_o3d, interp_pred_vertices_o3d])
            # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

            ############################################################
            # Compute EPE
            ############################################################
            l2_dist = gt_vertices - interp_pred_vertices
            l2_dist = l2_dist * l2_dist
            l2_dist = np.sum(l2_dist, axis=1)
            l2_dist = np.sqrt(l2_dist)

            # Compute metrics.
            epe3d = l2_dist.mean()
            epe3d_array.append(epe3d)

        # Compute average metrics.
        epe3d_array_tmp = np.array(epe3d_array)
        epe3d_avg_tmp = np.mean(epe3d_array_tmp)
        print("epe3d_avg_tmp: {}".format(epe3d_avg_tmp))

    # Compute average metrics.
    epe3d_array = np.array(epe3d_array)
    epe3d_avg = np.mean(epe3d_array)

    return epe3d_avg


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(
        description='Run Model'
    )
    parser.add_argument('-n', '--run_name', default=None)
    args = parser.parse_args()

    debug = False
    if debug: input("Really want to debug?")

    sample_num = 100000
    knn = 1
    coverage = 0.005 # np.max(dist)
    keyframe_every = 50

    #############################
    
    viz = False
    
    # unit bbox
    p_min = -0.5
    p_max =  0.5
    unit_bbox = BBox.compute_bbox_from_min_point_and_max_point(
        np.array([p_min]*3), np.array([p_max]*3)
    )
    unit_bbox = rotate_around_axis(unit_bbox, axis_name="x", angle=-np.pi) 
    
    
    #############################
    # Experiment
    #############################

    start_from_frame = 0
    end_before_frame = -1

    # ---------------------------------------------

    exp_name = "2021-02-10__NESHPOD__nss0.7_uni0.3__bs4__lr-0.0005-0.0005-0.001-0.001__s128-256__p256-512__wSE3__wShapePosEnc__wPosePosEnc__ON__MIX-POSE__AMASS-50id-10349__MIXAMO-165id-40000__CAPE-35id-20533"
    
    if args.run_name is None:
        # run_name = "2021-02-15__MIXAMO_TRANS_ALL-POSE-TEST-olivia-female_samba_ijexa_break-1id-300ts-1seqs__partial__icp0.0005-500__iters1000__sreg0.1_preg0.0001__slr0.0005_plr0.001__interval500_factor0.5__clamp0.1__sigma0.015__tmpreg100__cptFromTrain92"
        # run_name = "2021-02-15__CAPE-POSE-TEST-00032_shortshort-shoulders_mill-1id-207ts-1seqs__partial__icp0.0005-500__iters1000__sreg0.1_preg0.0001__slr0.0005_plr0.001__interval500_factor0.5__clamp0.1__sigma0.015__tmpreg100__cptFromTrain92"
        # run_name = "2021-02-19__DFAUST-POSE-TEST-50021-running_on_spot-1id-17ts-1seqs__icp0.0001-500__iters1000__sreg0.1_preg0.0001__slr0.0005_plr0.001__interval500_factor0.5__clamp0.1__sigma0.015__tmpreg100__cptFromTrain92__woALT__woClampRed__part"
        
        ####################################################################
        # CAPE
        ####################################################################
        # run_name = "2021-03-01__CAPE-POSE-TEST-00032_shortlong-hips-1id-293ts-1seqs__icp0.0005-500__iters1000__sreg0.1_preg0.0001__slr0.0005_plr0.001__interval500_factor0.5__clamp0.1__sigma0.015__tmpreg100__cptFromTrain92__woALT__woClampRed__compIFNet"
        run_name = "2021-03-01__CAPE-POSE-TEST-03223_shortshort-tilt_twist_left-1id-349ts-1seqs__icp0.0005-500__iters1000__sreg0.1_preg0.0001__slr0.0005_plr0.001__interval500_factor0.5__clamp0.1__sigma0.015__tmpreg100__cptFromTrain92__woALT__woClampRed__compIFNet"

        # Test 2 - CAPE-POSE-TEST-03223_shortshort-tilt_twist_left-1id-349ts-1seqs
        # start_from_frame = 50
        # end_before_frame = 150
        # run_name = "2021-02-15__CAPE-POSE-TEST-00032_shortshort-shoulders_mill-1id-207ts-1seqs__partial__icp0.0005-500__iters1000__sreg0.1_preg0.0001__slr0.0005_plr0.001__interval500_factor0.5__clamp0.1__sigma0.015__tmpreg100__cptFromTrain92"
        
        # Test 3 - CAPE-POSE-TEST-03223_shortshort-tilt_twist_left-1id-349ts-1seqs
        # start_from_frame = 30
        # end_before_frame = 323
        # run_name = "2021-02-15__CAPE-POSE-TEST-03223_shortlong-shoulders_mill-1id-378ts-1seqs__partial__icp0.0005-500__iters1000__sreg0.1_preg0.0001__slr0.0005_plr0.001__interval500_factor0.5__clamp0.1__sigma0.015__tmpreg100__cptFromTrain92"
        
        # Test 4 - CAPE-POSE-TEST-03223_shortshort-tilt_twist_left-1id-349ts-1seqs
        # start_from_frame = 50
        # end_before_frame = -1
        # run_name = "2021-02-15__CAPE-POSE-TEST-03223_shortshort-tilt_twist_left-1id-349ts-1seqs__partial__icp0.0005-500__iters1000__sreg0.1_preg0.0001__slr0.0005_plr0.001__interval500_factor0.5__clamp0.1__sigma0.015__tmpreg100__cptFromTrain92"
        

        # run_name = "2021-03-01__CAPE-POSE-TEST-03223_shortshort-tilt_twist_left-1id-30ts-1seqs__icp0.001-50__iters100__sreg0.1_preg0.0001__slr0.0001_plr0.0001__interval50_factor0.5__clamp0.1__sigma0.015__tmpreg100__cptFromTrain92__woALT__woClampRed__part"

        # run_name = "2021-03-01__CAPE-POSE-TEST-03223_shortshort-tilt_twist_left-1id-349ts-1seqs__icp0.0005-500__iters1000__sreg0.1_preg0.0001__slr0.0005_plr0.001__interval500_factor0.5__clamp0.1__sigma0.015__tmpreg100__cptFromTrain92__woALT__woClampRed__comp"

        # run_name = "2021-03-01__CAPE-POSE-TEST-00032_shortshort-shoulders_mill-1id-207ts-1seqs__icp0.0005-500__iters1000__sreg0.1_preg0.0001__slr0.0005_plr0.001__interval500_factor0.5__clamp0.1__sigma0.015__tmpreg100__cptFromTrain92__woALT__woClampRed__compIFNet"
        
        # CHALLENGING CAPE
        # run_name = "2021-03-02__CAPE-POSE-TEST-00032_shortlong-tilt_twist_left-1id-100ts-1seqs-50to149__bs8__icp0.0005-500__iters1000__sreg0.1_preg0.0001__slr0.0005_plr0.001__interval500_factor0.5__clamp0.1__sigma0.015__tmpreg100__cptFromTrain92__woALT__woClampRed__part"
        # run_name = "2021-03-02__CAPE-POSE-TEST-00032_shortlong-tilt_twist_left-1id-100ts-1seqs-170to269__bs3__icp0.0005-500__iters1000__sreg0.1_preg0.0001__slr0.0005_plr0.001__interval500_factor0.5__clamp0.1__sigma0.015__tmpreg100__cptFromTrain92__woALT__woClampRed__part"

    else:
        run_name = args.run_name

    assert run_name is not None
    # ---------------------------------------------
    
    exp_version = "sdf"
    exps_dir  = os.path.join(cfg.exp_dir, exp_version)
    optimization_dir = os.path.join(exps_dir, exp_name, "optimization", run_name)
    predicted_meshes_dir_list = os.path.join(optimization_dir, f"predicted_meshes*")
    print("Which one do you want?")
    predicted_meshes_dir = None
    for tmp in sorted(glob.glob(predicted_meshes_dir_list)):
        answer = query_yes_no(tmp, default="no")
        if answer:
            predicted_meshes_dir = tmp
            break
    assert predicted_meshes_dir is not None
    assert os.path.isdir(predicted_meshes_dir), predicted_meshes_dir

    #############################

    dataset_name = run_name.split('__')[1]

    #############################
    # Groundtruth data    
    #############################
    from utils.parsing_utils import get_dataset_type_from_dataset_name
    dataset_type = get_dataset_type_from_dataset_name(dataset_name)
    splits_dir = f"{cfg.splits_dir}_{dataset_type}"

    data_base_dir = "/cluster/lothlann/ppalafox/datasets"
    data_dir = f"{data_base_dir}/{splits_dir}/{dataset_name}"
    assert os.path.isdir(data_dir), data_dir

    with open(os.path.join(data_dir, "labels.json"), 'r') as f:
        labels = json.loads(f.read())

    #############################
    #############################

    keyframe_every = min(keyframe_every, len(labels))
    start_from_frame= min(start_from_frame, len(labels))

    print()
    print("#"*60)
    print("TRACKING")
    print("#"*60)
    epe3d_avg = compute_tracking_error(
        data_base_dir, labels, predicted_meshes_dir, sample_num, knn, coverage, keyframe_every=keyframe_every,
        start_from_frame=start_from_frame,
        end_before_frame=end_before_frame,
    )
    
    print()
    print("#"*60)
    print("RECONSTRUCTION")
    print("#"*60)
    iou_avg, chamfer_avg, accuracy_avg, completeness_avg, normal_consis_avg = compute_reconstruction_error(
        data_base_dir, labels, predicted_meshes_dir, 
        start_from_frame=start_from_frame,
        end_before_frame=end_before_frame
    )

    print()
    print("#"*60)
    print("iou_avg:           {}".format(iou_avg))
    print("chamfer_avg:       {}".format(chamfer_avg))
    print("accuracy_avg:      {}".format(accuracy_avg))
    print("completeness_avg:  {}".format(completeness_avg))
    print("normal_consis_avg: {}".format(normal_consis_avg))
    print("epe3d_avg:         {}".format(epe3d_avg))
    print("#"*60)

    print()
    print("exp_name")
    print(exp_name)
    print("run_name")
    print(run_name)

    # Write results
    results_summary_path = os.path.join(optimization_dir, f"results_summary__{sample_num}samples__{knn}knn__{coverage}coverage__{keyframe_every}kf_every.csv")
    with open(results_summary_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["iou_avg", iou_avg])
        csvwriter.writerow(["chamfer_avg", chamfer_avg])
        csvwriter.writerow(["normal_consis_avg", normal_consis_avg])
        csvwriter.writerow(["epe3d_avg", epe3d_avg])