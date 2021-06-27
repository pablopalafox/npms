import numpy as np

from NPMs._C import backproject_depth_ushort as backproject_depth_ushort_c
from NPMs._C import backproject_depth_float as backproject_depth_float_c
from NPMs._C import compute_validity_and_sign_mask as compute_validity_and_sign_mask_c
from NPMs._C import compute_normals as compute_normals_c
from NPMs._C import compute_normals_via_pca as compute_normals_via_pca_c


def backproject_depth_py(depth_image, fx, fy, cx, cy, normalizer = 1000.0):
    assert len(depth_image.shape) == 2
    width = depth_image.shape[1]
    height = depth_image.shape[0]

    point_image = np.zeros((3, height, width))
    for y in range(height):
        for x in range(width):
            depth = depth_image[y, x] / normalizer
            if depth > 0:
                pos_x = depth * (x - cx) / fx
                pos_y = depth * (y - cy) / fy
                pos_z = depth

                point_image[0, y, x] = pos_x
                point_image[1, y, x] = pos_y
                point_image[2, y, x] = pos_z

    return point_image


def backproject_depth(depth_image, fx, fy, cx, cy, default="zeros", normalizer=1000.0):
    assert len(depth_image.shape) == 2
    width = depth_image.shape[1]
    height = depth_image.shape[0]

    if default == "zeros":
        point_image = np.zeros((3, height, width), dtype=np.float32)
    elif default == "NaN":
        point_image = np.empty((3, height, width), dtype=np.float32)
        point_image[:] = np.NaN
    else:
        raise Exception("Default value for point_image is not implemented")

    if depth_image.dtype == np.float64:
        depth_image = depth_image.astype(np.float32)
        
    if depth_image.dtype == np.float32:
        backproject_depth_float_c(depth_image, point_image, fx, fy, cx, cy)
    elif depth_image.dtype == np.uint16:
        backproject_depth_ushort_c(depth_image, point_image, fx, fy, cx, cy, normalizer)
    else:
        raise Exception("Not implemented for Type {}".format(depth_image.dtype))

    assert point_image.dtype == np.float32

    return point_image


def compute_validity_and_sign_mask(
    grid_points_cam, valid_depth_mask_image, depth_image, fx, fy, cx, cy, w, h, truncation=0.01
):
    assert len(depth_image.shape) == 2
    if depth_image.dtype == np.float64:
        depth_image = depth_image.astype(np.float32)

    # By default, all grid points are valid
    validity_grid_mask = np.ones((grid_points_cam.shape[0]), dtype=np.bool)

    # By default, all grid points have a positive SDF
    sign_grid_mask = np.ones((grid_points_cam.shape[0]), dtype=np.float32)

    compute_validity_and_sign_mask_c(
        grid_points_cam,
        valid_depth_mask_image,
        depth_image,
        fx, fy, cx, cy, w, h,
        truncation,
        validity_grid_mask,
        sign_grid_mask
    )

    return validity_grid_mask, sign_grid_mask


def compute_validity_and_sign_mask_np(
    grid_points_cam, valid_depth_mask_image, depth_image, fx, fy, cx, cy, w, h, truncation=0.01
):
    assert len(depth_image.shape) == 2
    if depth_image.dtype == np.float64:
        depth_image = depth_image.astype(np.float32)
        
    # By default, all grid points are valid
    validity_grid_mask = np.ones((grid_points_cam.shape[0]), dtype=np.bool)

    # By default, all grid points have a positive SDF
    sign_grid_mask = np.ones((grid_points_cam.shape[0]), dtype=np.float32)

    for pt_id, grid_point_cam in enumerate(grid_points_cam):
    
        # Project point to camera plane
        x, y, z = grid_point_cam[0], grid_point_cam[1], grid_point_cam[2]

        px = int(round(fx * x / z + cx))
        py = int(round(fy * y / z + cy))

        if px < 0 or px >= w or py < 0 or py >= h:
            continue

        is_valid_pixel = valid_depth_mask_image[py, px]

        if not is_valid_pixel:
            continue
        
        d = depth_image[py, px]

        # If the z coordinate of the grid point is further away from the camera than the surface + some 
        # truncation, then we invalidate the grid point
        if z > (d + truncation):
            validity_grid_mask[pt_id] = False

        if z > d:
            sign_grid_mask[pt_id] = -1

    return validity_grid_mask, sign_grid_mask


def compute_normals_py(point_image_original, default="zeros", epsilon=1e-8):
    point_image = np.copy(point_image_original)
    
    assert len(point_image.shape) == 3

    if point_image.shape[0] == 3:
        point_image = np.moveaxis(point_image, 0, -1)
    
    assert point_image.shape[-1] == 3
    
    height = point_image.shape[0]
    width  = point_image.shape[1]

    if default == "zeros":
        normals = np.zeros((height, width, 3), dtype=np.float32)
    elif default == "NaN":
        normals = np.empty((height, width, 3), dtype=np.float32)
        normals[:] = np.NaN
    else:
        raise Exception("Default value for normals is not implemented")
    
    if point_image.dtype == np.float64:
        point_image = point_image.astype(np.float32)

    if point_image.dtype == np.float32:
        # Normals
        A = point_image[1:-1, 2:] - point_image[1:-1, 0:-2]
        B = point_image[2:, 1:-1] - point_image[0:-2, 1:-1]
        normal_matrix = np.cross(B, A, axis=2)

        valid_normals_tmp_3d = np.isfinite(normal_matrix)
        valid_normals_tmp = np.all(valid_normals_tmp_3d, axis=2)
        
        valid_normals = np.zeros((height, width), dtype=np.bool)
        valid_normals[1:-1, 1:-1] = valid_normals_tmp
        
        # set invalid pixels to 0
        normal_matrix[~valid_normals_tmp_3d] = 0.0
        assert np.isfinite(normal_matrix).all()
        
        # compute norm
        norm = np.linalg.norm(normal_matrix, axis=2)
        
        valid_norm_tmp = norm > epsilon
        valid_norm = np.zeros((height, width), dtype=np.bool)
        valid_norm[1:-1, 1:-1] = valid_norm_tmp

        valid_normals = valid_normals & valid_norm

        norm = norm[..., np.newaxis]
        norm = np.repeat(norm, 3, axis=2)

        normals[valid_normals] = normal_matrix[valid_normals_tmp] / norm[valid_normals_tmp]
    else:
        raise Exception("Not implemented for Type {}".format(normals.dtype))

    return normals


def compute_normals(point_image_original, default="zeros", epsilon=1e-8):
    point_image = np.copy(point_image_original)
    
    assert len(point_image.shape) == 3

    if point_image.shape[0] == 3:
        point_image = np.moveaxis(point_image, 0, -1)
    
    assert point_image.shape[-1] == 3

    height = point_image.shape[0]
    width  = point_image.shape[1]

    if default == "zeros":
        normals = np.zeros((height, width, 3), dtype=np.float32)
    elif default == "NaN":
        normals = np.empty((height, width, 3), dtype=np.float32)
        normals[:] = np.NaN
    else:
        raise Exception("Default value for normals is not implemented")
    
    if point_image.dtype == np.float64:
        point_image = point_image.astype(np.float32)

    if point_image.dtype == np.float32:
        compute_normals_c(point_image, normals, epsilon)
    else:
        raise Exception("Not implemented for Type {}".format(normals.dtype))

    return normals


def compute_normals_via_pca(point_image_original, default="zeros", kernel_size=5, max_distance=0.05):
    point_image = np.copy(point_image_original)
    
    assert len(point_image.shape) == 3

    if point_image.shape[0] == 3:
        point_image = np.moveaxis(point_image, 0, -1)

    assert point_image.shape[-1] == 3

    height = point_image.shape[0]
    width  = point_image.shape[1]

    if default == "zeros":
        normals = np.zeros((height, width, 3), dtype=np.float32)
    elif default == "NaN":
        normals = np.empty((height, width, 3), dtype=np.float32)
        normals[:] = np.NaN
    else:
        raise Exception("Default value for normals is not implemented")

    if point_image.dtype == np.float64:
        point_image = point_image.astype(np.float32)

    if point_image.dtype == np.float32:
        compute_normals_via_pca_c(point_image, normals, kernel_size, max_distance)
    else:
        raise Exception("Not implemented for Type {}".format(normals.dtype))

    return normals