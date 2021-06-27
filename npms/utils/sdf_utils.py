import numpy as np


def scale_grid(xyz, x_scale, y_scale, z_scale, bbox_min=-0.5, bbox_max=0.5):
    X, Y, Z = xyz
    X = X*x_scale
    Y = Y*y_scale
    Z = Z*z_scale
    points = np.concatenate((X[np.newaxis, ...], Y[np.newaxis, ...], Z[np.newaxis, ...]), axis=0)
    points = points.reshape(3, -1).astype(np.float32)
    points = (bbox_max - bbox_min) * points + bbox_min
    return points


def sample_scaled_grid_points(dims):

    # Generate regular input.
    grid_x = np.mgrid[:dims[0][0], :dims[0][1], :dims[0][2]]
    grid_y = np.mgrid[:dims[1][0], :dims[1][1], :dims[1][2]]
    grid_z = np.mgrid[:dims[2][0], :dims[2][1], :dims[2][2]]

    grids = [grid_x, grid_y, grid_z]

    points = []

    for i, grid in enumerate(grids):
        coords = scale_grid(grid, 1.0/(dims[i][0]-1.0), 1.0/(dims[i][1]-1.0), 1.0/(dims[i][2]-1.0))
        points.append(coords.astype(np.float32))

    return points


def sample_grid_points(dim):
    # Generate regular input.
    coords = scale_grid(np.mgrid[:dim, :dim, :dim], 1.0/(dim-1.0), 1.0/(dim-1.0), 1.0/(dim-1.0))
    return coords.astype(np.float32) # [3, N]
    

def sphere_tsdf(X, Y, Z, radius=0.5, truncate=True):
    sdf = X**2 + Y**2 + Z**2 - radius**2

    if truncate:
        truncation = 0.001 # relatively small truncation
        return np.clip(sdf, -truncation, truncation)

    else:
        return sdf


def predict_sdf(points, seq_idx, time, t_idx, point_images, model, dim, num_chunks=1, linear_interpolation=True):
    num_points = points.shape[-1]
    assert num_points % num_chunks == 0, "The number of points in the grid must be divisible by the number of chunks"
    points_per_chunk = int(num_points / num_chunks)

    sdf_pred = np.empty((1, 1, num_points), dtype=np.float32)
    for i in range(num_chunks):
        points_i = points.clone()[..., i*points_per_chunk:(i+1)*points_per_chunk]

        # Run forward pass.
        if seq_idx is not None:
            sdf_pred_i = model(points_i, seq_idx, t_idx, linear_interpolation).cpu().detach().numpy()
        else:
            sdf_pred_i = model(points_i, time, t_idx, point_images, linear_interpolation).cpu().detach().numpy()

        # Concatenate
        sdf_pred[..., i*points_per_chunk:(i+1)*points_per_chunk] = sdf_pred_i

    return sdf_pred.reshape(dim, dim, dim)


def predict_supersampled_sdf(points, time, t_idx, model, dims, num_chunks):
    num_points = points.shape[-1]
    assert num_points % num_chunks == 0, "The number of points in the grid must be divisible by the number of chunks"
    points_per_chunk = int(num_points / num_chunks)

    sdf_pred = np.empty((1, 1, num_points), dtype=np.float32)
    for i in range(num_chunks):
        points_i = points.clone()[..., i*points_per_chunk:(i+1)*points_per_chunk]

        # Run forward pass.
        sdf_pred_i = model(points_i, time, t_idx).cpu().detach().numpy()

        # Concatenate
        sdf_pred[..., i*points_per_chunk:(i+1)*points_per_chunk] = sdf_pred_i

    return sdf_pred.reshape(*dims)