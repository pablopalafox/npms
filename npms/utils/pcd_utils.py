import sys

import numpy as np
import torch
import trimesh
from timeit import default_timer as timer

import open3d as o3d


T_opengl_cv = np.array([
    [1.0,  0.0, 0.0],
    [0.0, -1.0, 0.0],
    [0.0, 0.0, -1.0]
])

T_opengl_cv_homogeneous = np.array([
    [1.0,  0.0,  0.0, 0.0],
    [0.0, -1.0,  0.0, 0.0],
    [0.0,  0.0, -1.0, 0.0],
    [0.0,  0.0,  0.0, 1.0],
])

origin = np.array([0, 0, 0])
z_axis = np.array([0, 0, 1])

unit_p_min = np.array([-1, -1, -1])
unit_p_max = np.array([ 1,  1,  1])


def transform_pointcloud_to_opengl_coords(points_cv):
    assert len(points_cv.shape) == 2 and points_cv.shape[1] == 3, points_cv.shape

    # apply 180deg rotation around 'x' axis to transform the mesh into OpenGL coordinates
    point_opengl = np.matmul(points_cv, T_opengl_cv.transpose())

    return point_opengl


class UnitBBox():
    unit_bbox_points = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1]
        ], dtype=np.float32)

    bbox_edges = [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 4],
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7]
        ]


class BBox:
    def __init__(self, points, percentage_of_diagonal_to_add=None):

        points_copy = np.copy(points)
        len_points_shape = len(points_copy.shape) 
        if len_points_shape == 3:
            if points_copy.shape[0] == 3:
                points_copy = np.moveaxis(points_copy, 0, -1)
            assert points_copy.shape[2] == 3
            points_copy = points_copy.reshape(-1, 3)
        elif len_points_shape == 2:
            if points_copy.shape[0] == 3:
                points_copy = np.moveaxis(points_copy, 0, -1)
            assert points_copy.shape[1] == 3
        else:
            raise Exception("Input shape not valid: {}".format(points.shape))
            
        self.min_point = np.min(points_copy, axis=0)[:, np.newaxis]
        self.max_point = np.max(points_copy, axis=0)[:, np.newaxis]
        self._enlarge_bbox(percentage_of_diagonal_to_add)
        self._compute_extent()

        #################################### debug ####################################
        # debug = False
        # if debug:
        #     referece_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        #         size=1, origin=[0, 0, 0])

        #     # tranform to opengl
        #     points_copy = transform_pointcloud_to_opengl_coords(points_copy)

        #     pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_copy))
        #     pcd.paint_uniform_color([1, 0.8, 0.506])

        #     bbox = self.compute_bbox_from_min_point_and_extent(
        #         self.min_point,
        #         self.extent
        #     )
        #     # tranform to opengl
        #     bbox.points = o3d.utility.Vector3dVector(transform_pointcloud_to_opengl_coords(np.asarray(bbox.points)))
                
        #     viz_and_exit([pcd, referece_frame, bbox])        
        #################################### debug ####################################

    def _compute_extent(self):
        self.extent = self.max_point - self.min_point
        
    def _enlarge_bbox(self, percentage_of_diagonal_to_add):
        if percentage_of_diagonal_to_add is None:
            return 

        diagonal = self.max_point - self.min_point
        diagonal_length = np.linalg.norm(diagonal)
    
        self.min_point = self.min_point - percentage_of_diagonal_to_add * diagonal_length
        self.max_point = self.max_point + percentage_of_diagonal_to_add * diagonal_length

    def get_bbox_as_array(self):
        return np.concatenate((self.min_point, self.extent), axis=1)
    
    def get_bbox_as_line_set(self):
        return BBox.compute_bbox_from_min_point_and_extent(self.min_point, self.extent)

    def get_bbox_center(self):
        return (self.min_point + self.max_point) / 2.

    @staticmethod
    def compute_extent(p_min, p_max):
        return p_max - p_min

    @staticmethod
    def compute_bbox_from_min_point_and_extent(p_min, extent, color=[1, 0, 0]):
        bbox_points = p_min.squeeze() + UnitBBox.unit_bbox_points * extent.squeeze()

        colors = [color for i in range(len(UnitBBox.bbox_edges))]
        bbox = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(bbox_points),
            lines=o3d.utility.Vector2iVector(UnitBBox.bbox_edges),
        )
        bbox.colors = o3d.utility.Vector3dVector(colors)

        return bbox

    @staticmethod
    def compute_bbox_from_min_point_and_max_point(p_min, p_max, color=[1, 0, 0]):
        extent = BBox.compute_extent(p_min, p_max)
        bbox = BBox.compute_bbox_from_min_point_and_extent(p_min, extent, color)
        return bbox

    @staticmethod
    def enlarge_bbox(p_min, p_max, displacement):
        p_min = p_min - displacement
        p_max = p_max + displacement
        return p_min, p_max

    @staticmethod
    def convert_bbox_to_cube(p_min, p_max):
        current_extent = BBox.compute_extent(p_min, p_max)
        cube_extent = np.max(current_extent) * np.ones_like(current_extent)

        delta = cube_extent - current_extent
        half_delta = delta / 2.0

        p_min = p_min - half_delta
        p_max = p_max + half_delta

        return p_min, p_max


def transform_to_noc_space(points, p_min, extent):
    translated_points = np.copy(points) - p_min
    scaled_points = translated_points / extent
    nocs = 2.0 * scaled_points - 1.0
    return nocs


def normalize_transformation(R, t, p_min, scale):
    R_normalized = R
    t_normalized = scale * (np.matmul(R, p_min) + t - p_min)
    return R_normalized, t_normalized


def align_vector_to_another(a=np.array([0, 0, 1]), b=np.array([1, 0, 0])):
    """
    Aligns vector a to vector b with axis angle rotation
    """
    if np.array_equal(a, b):
        return None, None
    axis_ = np.cross(a, b)
    axis_ = axis_ / np.linalg.norm(axis_)
    angle = np.arccos(np.dot(a, b))

    return axis_, angle

    
def normalize(a, axis=-1, order=2):
    """Normalizes a numpy array of points"""
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis), l2


def transform_mesh_to_unit_cube(
        mesh, p_min, extent, return_wireframe=False, to_opengl=False, return_only_vertices=False
    ):
    vertices = np.asarray(mesh.vertices)

    # The mesh is given in opengl coordinates.
    # So we'll transform first to vision coordinates and work there.
    vertices_cv = transform_pointcloud_to_opengl_coords(vertices)

    # Move point dimensions to the front
    vertices_cv = np.moveaxis(vertices_cv, -1, 0)
    assert vertices_cv.shape[0] == 3

    # The mesh's coordinates are already in world space
    # So we only need to transform them to noc space
    vertices_noc = transform_to_noc_space(vertices_cv, p_min=p_min, extent=extent)

    # Move point dimensions again to the last axis
    vertices_noc = np.moveaxis(vertices_noc, -1, 0)

    if return_only_vertices:
        return vertices_noc

    # Update mesh vertices
    mesh.vertices = o3d.utility.Vector3dVector(vertices_noc)

    if to_opengl:
        mesh.transform(T_opengl_cv_homogeneous)
    
    # Get a wireframe from the mesh
    if return_wireframe:
        mesh_wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
        mesh_wireframe.colors = o3d.utility.Vector3dVector(
            np.array([0.5, 0.1, 0.0]) * np.ones_like(vertices_noc)
        )
        # mesh_wireframe.paint_uniform_color([0.5, 0.1, 0.0])
        return mesh_wireframe

    mesh.compute_vertex_normals()

    return mesh


def load_and_transform_mesh_to_unit_cube(
        mesh_path, p_min, extent, return_wireframe=False, to_opengl=False
    ):
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    return transform_mesh_to_unit_cube(mesh, p_min, extent, return_wireframe, to_opengl)
    

def rotate_around_axis(obj, axis_name, angle):
    axis_name = axis_name.lower()
    if axis_name == "x":
        axis = np.array([1, 0, 0])[:, None]
    elif axis_name == "y":
        axis = np.array([0, 1, 0])[:, None]
    elif axis_name == "z":
        axis = np.array([0, 0, 1])[:, None]
    else:
        raise Exception(f"Axis {axis_name} is not a valid axis!")
    
    axis_angle = axis * angle
    R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis_angle)
    obj.rotate(R, center=[0, 0, 0])
    return obj


def initialize_surface_voxels(grid, unique_voxel_coords, inv_ind, points_cur, surface_flow, res, voxel_coverage_2):

    flow_field_mask = np.zeros((res, res, res), dtype=np.float32)
    flow_field      = np.zeros((res, res, res, 3), dtype=np.float32)
    
    
    for i in range(len(unique_voxel_coords)):
        v = unique_voxel_coords[i].astype(np.int32)
        grid_point = grid[v[0], v[1], v[2], :]

        mask = inv_ind == i

        pts      = points_cur[mask]
        pts_flow = surface_flow[mask]

        num_pts = pts.shape[0]

        if num_pts > 1:
    
            # assert pts.shape[-1] == 3
            diff = pts - grid_point
            # assert diff.shape[-1] == 3
            dist = np.linalg.norm(diff, axis=-1)

            weights = np.exp(-dist**2 / (2.0 * voxel_coverage_2))[:, None]
            weights /= np.sum(weights)
            weights = np.repeat(weights, 3, axis=-1)

            pts_flow = weights * pts_flow
            pts_flow = np.sum(pts_flow, axis=0, keepdims=True)

        assert pts_flow.shape == (1, 3), pts_flow.shape
        flow_field[v[0], v[1], v[2], :] = pts_flow
        flow_field_mask[v[0], v[1], v[2]] = 1.0

    return flow_field_mask, flow_field


def get_grid_coords(points):
    coords = points.copy()
    coords[..., 0], coords[..., 2] = points[..., 2], points[..., 0]
    return 2 * coords


def sample_points(mesh_source, num_samples, return_barycentric=True, sample_even=False):

    mesh = mesh_source.copy()

    if sample_even:
        samples, face_indices = trimesh.sample.sample_surface_even(mesh, num_samples)
    else:
        samples, face_indices = mesh.sample(num_samples, return_index=True)
    
    if not return_barycentric:
        return samples

    samples_triangles = mesh.triangles[face_indices]
    bary_coords = trimesh.triangles.points_to_barycentric(samples_triangles, samples)
    return samples, face_indices, bary_coords, samples_triangles


def sample_points_give_bary(mesh_source, face_indices, bary_coords):
    mesh = mesh_source.copy()
    samples_triangles = mesh.triangles[face_indices]
    p = bary_coords[..., None] * samples_triangles
    p = np.sum(p, axis=1)
    return p, samples_triangles


def trimesh_to_open3d(mesh_trimesh, color=[0.3, 0.3, 0.3]):
    mesh_o3d = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(mesh_trimesh.vertices),
        o3d.utility.Vector3iVector(mesh_trimesh.faces),
    )
    mesh_o3d.compute_vertex_normals()
    mesh_o3d.paint_uniform_color(color)
    return mesh_o3d

def open3d_to_trimesh(mesh_o3d):
    mesh_trimesh = trimesh.Trimesh(
        np.array(mesh_o3d.vertices),
        np.array(mesh_o3d.triangles),
        process=False
    )
    return mesh_trimesh

def filter_mesh(mesh, cylinder_radius=0.4):
    import numpy.linalg as linalg
    # Filter meshes outside of a vertical cylinder (along the y axis, in practice)
    
    vertices = mesh.vertices
    faces = mesh.faces

    # Find the indices of invalid vertices
    vertices_xz_plane = np.delete(vertices, 1, axis=1)
    dist_to_cylinder = linalg.norm(vertices_xz_plane, axis=1)
    invalid_vertices_mask = dist_to_cylinder > cylinder_radius
    invalid_vertices_inds = np.argwhere(invalid_vertices_mask)

    # Return the input mesh if no invalid vertices
    num_invalid_vertices = np.sum(invalid_vertices_mask)
    if num_invalid_vertices == 0:
        return mesh
    
    print(f"Removing {num_invalid_vertices} vertices")

    # Create open3d mesh
    mesh_o3d = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices),
        o3d.utility.Vector3iVector(faces),
    )
    mesh_o3d.remove_vertices_by_index(invalid_vertices_inds)
    
    if False:
        mesh_o3d.compute_vertex_normals()
        mesh_o3d.paint_uniform_color([1, 1, 0])
        o3d.visualization.draw_geometries([mesh_o3d])

    filtered_mesh = trimesh.Trimesh(
        np.array(mesh_o3d.vertices),
        np.array(mesh_o3d.triangles),
        process=False
    )

    return filtered_mesh


def filter_mesh_o3d(mesh, cylinder_radius=0.4):
    import numpy.linalg as linalg
    # Filter meshes outside of a vertical cylinder (along the y axis, in practice)
    
    vertices = np.array(mesh.vertices)

    # Find the indices of invalid vertices
    vertices_xz_plane = np.delete(vertices, 1, axis=1)
    dist_to_cylinder = linalg.norm(vertices_xz_plane, axis=1)
    invalid_vertices_mask = dist_to_cylinder > cylinder_radius
    invalid_vertices_inds = np.argwhere(invalid_vertices_mask)

    # Return the input mesh if no invalid vertices
    num_invalid_vertices = np.sum(invalid_vertices_mask)
    if num_invalid_vertices == 0:
        return mesh
    
    mesh.remove_vertices_by_index(invalid_vertices_inds)
    print(f"Removed {num_invalid_vertices} vertices")
    
    return mesh