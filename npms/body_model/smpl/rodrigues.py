import torch


def skew_matrix(vectors):
    x, y, z = torch.split(vectors, 1, dim=1)
    zero = torch.zeros_like(x)
    matrix = torch.cat([zero, -z, y, z, zero, -x, -y, x, zero], dim=1).reshape((-1, 3, 3))
    return matrix


def compact_rodrigues(v):
    dtype = v.dtype
    device = v.device
    
    angle = torch.norm(v, dim=1, keepdim=True)
    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)
    K = skew_matrix(v / (angle + 1e-8))
    
    ident = torch.eye(3, dtype=dtype, device=device)[None]
    rot_mats = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mats


def compact_derive_rodrigues(v):
    # https://arxiv.org/pdf/1312.0788.pdf
    dtype = v.dtype
    device = v.device

    rot = compact_rodrigues(v)
    angle = torch.norm(v, dim=1)[:, None, None]
    e = torch.eye(3, dtype=dtype, device=device)
    a = torch.einsum('bi,bjk->bijk', v, skew_matrix(v))
    bx = skew_matrix(torch.cross(v, (e - rot) @ e[0]))
    by = skew_matrix(torch.cross(v, (e - rot) @ e[1]))
    bz = skew_matrix(torch.cross(v, (e - rot) @ e[2]))

    drdx = torch.einsum('bij,bjk->bik', (a[:, 0] + bx) / (angle + 1e-8) ** 2, rot)
    drdy = torch.einsum('bij,bjk->bik', (a[:, 1] + by) / (angle + 1e-8) ** 2, rot)
    drdz = torch.einsum('bij,bjk->bik', (a[:, 2] + bz) / (angle + 1e-8) ** 2, rot)

    drotation = torch.stack([drdx, drdy, drdz], dim=-1).reshape(-1, 9, 3)
    return drotation
    

def rodrigues(angle_axis):
    # https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    dtype = angle_axis.dtype
    batch_size = angle_axis.shape[0]
    device = angle_axis.device
    
    # Bx1 arrays
    angle, rx, ry, rz = torch.split(angle_axis, 1, dim=1)
    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)
    
    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))
    
    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mats = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mats


def inverse_rodrigues(rot_mats, eps=1e-8):
    # https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToAngle/index.htm
    batch_size = rot_mats.shape[0]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.split(rot_mats.view(batch_size, -1), 1, dim=1)
    angle = torch.acos(torch.clamp((m00 + m11 + m22 - 1) / 2, -1, 1))
    x = (m21 - m12) / (torch.sqrt((m21 - m12)**2 + (m02 - m20)**2 + (m10 - m01)**2 + eps**2))
    y = (m02 - m20) / (torch.sqrt((m21 - m12)**2 + (m02 - m20)**2 + (m10 - m01)**2 + eps**2))
    z = (m10 - m01) / (torch.sqrt((m21 - m12)**2 + (m02 - m20)**2 + (m10 - m01)**2 + eps**2))
    angle_axis = torch.cat([angle, x, y, z], dim=1)
    return angle_axis


def stable_inverse_rodrigues(rot_mats, eps=1e-8):
    # https://math.stackexchange.com/questions/83874/efficient-and-accurate-numerical-implementation-of-the-inverse-rodrigues-rotatio?rq=1
    batch_size = rot_mats.shape[0]
    device = rot_mats.device
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = \
        map(lambda x: x.reshape(-1), torch.split(rot_mats.view(batch_size, -1), 1, dim=1))

    v = torch.stack([m21 - m12, m02 - m20, m10 - m01], dim=-1)
    trace = (m00 + m11 + m22).clamp(-1, 1)
    rot_vecs = torch.zeros_like(v)

    # case x
    ix = trace >= 3 - eps
    rot_vecs[ix] = (0.5 * (trace[ix] - 3) / 12) @ v[ix] if ix.sum() > 1 else torch.empty_like(v[ix])

    # case y
    iy = (3 - eps > trace) & (trace > -1 + eps)
    theta = torch.acos((trace[iy] - 1) / 2)
    rot_vecs[iy] = torch.div(theta, 2 * torch.sin(theta)) @ v[iy] if iy.sum() > 1 else torch.empty_like(v[iy])

    # case z
    iz = ~(ix | iy)
    a = torch.ones((batch_size,), dtype=torch.int64, device=device)
    a[(m00 >= m11) & (m00 >= m22)] *= 0
    a[(m11 > m00) & (m11 >= m22)] *= 1
    a[(m22 > m00) & (m22 > m00)] *= 2
    b = torch.fmod(a + 1, 3)
    c = torch.fmod(a + 2, 3)

    s = torch.sqrt(rot_mats[iz, a, a] - rot_mats[iz, b, b] - rot_mats[iz, c, c] + 1)
    v[iz, a] = s / 2
    v[iz, b] = torch.div(rot_mats[iz, b, a] + rot_mats[iz, a, b], (2 * s))
    v[iz, c] = torch.div(rot_mats[iz, c, a] + rot_mats[iz, a, c], (2 * s))
    rot_vecs[iz] = 3.14159265359 * v[iz] / torch.norm(v[iz], dim=-1, keepdim=True)

    return rot_vecs


def derive_rodrigues(rot_vecs, eps=1e-8):
    dtype = rot_vecs.dtype
    device = rot_vecs.device
    batch_size = rot_vecs.shape[0]
    
    # Bx1 arrays
    angle = torch.norm(rot_vecs + eps, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    
    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)
    ze = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    on = torch.ones((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([ze, -rz, ry, rz, ze, -rx, -ry, rx, ze], dim=1).view((batch_size, 3, 3))
    Kx = torch.cat([ze, ry, rz, ry, -2 * rx, ze, rz, ze, -2 * rx], dim=1).view((batch_size, 3, 3))
    Ky = torch.cat([-2 * ry, rx, ze, rx, ze, rz, ze, rz, -2 * ry], dim=1).view((batch_size, 3, 3))
    Kz = torch.cat([-2 * rz, ze, rx, ze, -2 * rz, ry, rx, ry, ze], dim=1).view((batch_size, 3, 3))
    
    drot_dangle = cos * K + sin * torch.bmm(K, K)
    drot_drx = sin * torch.cat([ze, ze, ze, ze, ze, -on, ze, on, ze], dim=1).view((batch_size, 3, 3)) + (1 - cos) * Kx
    drot_dry = sin * torch.cat([ze, ze, on, ze, ze, ze, -on, ze, ze], dim=1).view((batch_size, 3, 3)) + (1 - cos) * Ky
    drot_drz = sin * torch.cat([ze, -on, ze, on, ze, ze, ze, ze, ze], dim=1).view((batch_size, 3, 3)) + (1 - cos) * Kz
    
    drot_mat = torch.stack((drot_dangle, drot_drx, drot_dry, drot_drz), dim=-1).reshape(batch_size, 9, 4)
    return drot_mat


def axis_angle(rot_vecs, eps=1e-8):
    angle = torch.norm(rot_vecs + eps, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle
    result = torch.cat((angle, rot_dir), dim=1)
    return result


def inverse_axis_angle(angle_axis, eps=1e-8):
    angle = angle_axis[:, 0:1] - eps
    rot_vecs = angle_axis[:, 1:] * angle
    return rot_vecs


def derive_axis_angle(rot_vecs, eps=1e-8):
    dtype = rot_vecs.dtype
    device = rot_vecs.device
    batch_size = rot_vecs.shape[0]
    
    rx, ry, rz = torch.split(rot_vecs, 1, dim=1)
    norm = torch.pow((rx + eps) ** 2 + (ry + eps) ** 2 + (rz + eps) ** 2, 0.5)
    
    daxis_angle = torch.zeros((batch_size, 4, 3), dtype=dtype, device=device)
    daxis_angle[:, 0, 0] = ((rx + eps) / norm).squeeze()
    daxis_angle[:, 0, 1] = ((ry + eps) / norm).squeeze()
    daxis_angle[:, 0, 2] = ((rz + eps) / norm).squeeze()
    daxis_angle[:, 1, 0] = (1 / norm - rx * (rx + eps) / norm ** 3).squeeze()
    daxis_angle[:, 1, 1] = (-rx * (ry + eps) / norm ** 3).squeeze()
    daxis_angle[:, 1, 2] = (-rx * (rz + eps) / norm ** 3).squeeze()
    daxis_angle[:, 2, 0] = (-ry * (rx + eps) / norm ** 3).squeeze()
    daxis_angle[:, 2, 1] = (1 / norm - ry * (ry + eps) / norm ** 3).squeeze()
    daxis_angle[:, 2, 2] = (-ry * (rz + eps) / norm ** 3).squeeze()
    daxis_angle[:, 3, 0] = (-rz * (rx + eps) / norm ** 3).squeeze()
    daxis_angle[:, 3, 1] = (-rz * (ry + eps) / norm ** 3).squeeze()
    daxis_angle[:, 3, 2] = (1 / norm - rz * (rz + eps) / norm ** 3).squeeze()
    
    return daxis_angle.reshape(batch_size, 4, 3)