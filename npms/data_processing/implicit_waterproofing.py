import os, sys

import numpy as np

import math
from multiprocessing import Pool

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from external.libmesh.inside_mesh import check_mesh_contains


def to_rotation_matrix(euler_angles):
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(euler_angles[0]), -math.sin(euler_angles[0])],
                    [0, math.sin(euler_angles[0]), math.cos(euler_angles[0])]
                    ])

    R_y = np.array([[math.cos(euler_angles[1]), 0, math.sin(euler_angles[1])],
                    [0, 1, 0],
                    [-math.sin(euler_angles[1]), 0, math.cos(euler_angles[1])]
                    ])

    R_z = np.array([[math.cos(euler_angles[2]), -math.sin(euler_angles[2]), 0],
                    [math.sin(euler_angles[2]), math.cos(euler_angles[2]), 0],
                    [0, 0, 1]
                    ])

    R = np.dot(R_z, np.dot(R_y, R_x))

    return R


def implicit_waterproofing(mesh_source, query_points):
    occ_list, holes_list = check_mesh_contains(mesh_source, query_points)

    for euler_angles in np.array([[0, np.pi / 2, 0], [np.pi / 2, 0, 0], [0, 0, np.pi / 2]]):

        if not holes_list.any():
            # print("\tNo holes! :)")
            break

        mesh = mesh_source.copy()
        r = to_rotation_matrix(euler_angles)
        r = np.pad(r, [(0, 1), (0, 1)], 'constant', constant_values=0)
        mesh.apply_transform(r)
        points = np.dot(r[:3, :3], query_points[holes_list].T).T

        occ_list_rot, holes_list_rot = check_mesh_contains(mesh, points)

        occ_list[holes_list] = occ_list_rot
        holes_list_updated = np.full(len(query_points), False)
        holes_list_updated[holes_list] = holes_list_rot
        holes_list = holes_list_updated

    return occ_list, holes_list


def implicit_waterproofing_with_normal_direction(mesh_source, query_points, is_inwards):
    occ_list, holes_list = check_mesh_contains(mesh_source, query_points)

    # For "bad" points, check their corresponding normal direction and mark them 
    # as occupied if the normal sign is negative
    bad_points_but_inwards = np.logical_and(holes_list, is_inwards)
    holes_list[bad_points_but_inwards] = False
    occ_list[bad_points_but_inwards] = True

    for euler_angles in np.array([[0, np.pi / 2, 0], [np.pi / 2, 0, 0], [0, 0, np.pi / 2]]):

        if not holes_list.any():
            # print("\tNo holes! :)")
            break

        mesh = mesh_source.copy()
        r = to_rotation_matrix(euler_angles)
        r = np.pad(r, [(0, 1), (0, 1)], 'constant', constant_values=0)
        mesh.apply_transform(r)
        points = np.dot(r[:3, :3], query_points[holes_list].T).T

        occ_list_rot, holes_list_rot = check_mesh_contains(mesh, points)

        occ_list[holes_list] = occ_list_rot
        holes_list_updated = np.full(len(query_points), False)
        holes_list_updated[holes_list] = holes_list_rot
        holes_list = holes_list_updated

    return occ_list, holes_list


# specifically for .binvox format (every cube has same hight,width and depth)
def create_grid_points(mesh, res):
    bottom_cotner, upper_corner = mesh.bounds
    minimun = min(bottom_cotner)
    maximum = max(upper_corner)
    x = np.linspace(minimun, maximum, res)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    X = X.reshape((np.prod(X.shape),))
    Y = Y.reshape((np.prod(Y.shape),))
    Z = Z.reshape((np.prod(Z.shape),))

    points_list = np.column_stack((X, Y, Z))
    del X, Y, Z, x
    return points_list

def create_grid_points_from_bounds(minimun, maximum, res):
    x = np.linspace(minimun, maximum, res)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    X = X.reshape((np.prod(X.shape),))
    Y = Y.reshape((np.prod(Y.shape),))
    Z = Z.reshape((np.prod(Z.shape),))

    points_list = np.column_stack((X, Y, Z))
    del X, Y, Z, x
    return points_list


# # Converting to occupancy grid
def to_occ(mesh, res):
    occ, holes = implicit_waterproofing(mesh, create_grid_points(mesh, res))
    occ = np.reshape(occ,(res,res,res))
    return occ