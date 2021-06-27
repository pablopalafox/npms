import torch
import numpy as np

import kornia
from timeit import default_timer as timer


def quaternion_multiply(quaternion1, quaternion2):
    """Multiply quaternion1 by quaternion2.

    Args:
        quaternion1: N x 4 quaternions
        quaternion2: N x 4 quaternions
    Returns:
        res: N x 4
    """

    x1, y1, z1, w1 = quaternion1[:, 0], quaternion1[:, 1], quaternion1[:, 2], quaternion1[:, 3]
    x2, y2, z2, w2 = quaternion2[:, 0], quaternion2[:, 1], quaternion2[:, 2], quaternion2[:, 3]

    res = torch.empty_like(quaternion1)
    res[:, 0] = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    res[:, 1] = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    res[:, 2] = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    res[:, 3] = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2

    return res


def quaternion_conj(q):
    """Compute conjugate quaternion
    Args:
        q: N x 4 quaternions

        We assume above quaternions to be in the form [x, y, z, w]
    Returns:
        q: N x 4 quaternions
    """
    return torch.cat([-1 * q[:, :-1], q[:, [-1]]], dim=1)


def rotate_points_with_quaternions(p, q):
    """Multiply points p with quaternions q.

    Args:
        p: N x 4 points (pure quaternions, with w = 0)
        q: N x 4 quaternions
    
        We assume above quaternions to be in the form [x, y, z, w]

    Returns:
        p_rot: N x 3 rotated points
    """

    N = p.shape[0]


    # Compute conjugate quaternion
    q_conj = quaternion_conj(q)

    # Reshape points
    p_quat = torch.zeros((N, 4), dtype=p.dtype, device=p.device)
    p_quat[:, :-1] = p

    assert p_quat.shape == q.shape, f"{p_quat.shape} vs {q.shape}"
    
    p_quat = quaternion_multiply(q, p_quat)
    p_rot  = quaternion_multiply(p_quat, q_conj)

    return p_rot[:, :-1] 


if __name__ == "__main__":
    pass