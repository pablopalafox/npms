import struct
import os
import numpy as np

from utils import base_util
from utils import file_util


def read_pts_file(path):
    """Reads a .pts or a .sdf point samples file."""
    _, ext = os.path.splitext(path)
    assert ext in ['.sdf', '.pts']
    l = 4 if ext == '.sdf' else 6
    with file_util.open_file(path, 'rb') as f:
        points = np.fromfile(f, dtype=np.float32)
    points = np.reshape(points, [-1, l])
    return points


def read_depth_im(path):
    """Loads a GAPS depth image stored as a 16-bit monochromatic PNG."""
    return file_util.read_image(path) / 1000.0


def depth_path_name(depth_dir, idx):
  """Generates the GAPS filename for a depth image from its index and dir."""
  return os.path.join(depth_dir, '%s_depth.png' % str(idx).zfill(6))


def read_depth_directory(depth_dir, im_count):
    """Reads the images in a directory of depth images made by scn2img.
    Args:
    depth_dir: Path to the root directory containing the scn2img output images.
    im_count: The number of images to read. Will read images with indices
        range(im_count).
    Returns:
    Numpy array with shape [im_count, height, width]. Dimensions determined from
        file.
    """
    depth_ims = []
    for i in range(im_count):
        path = depth_path_name(depth_dir, i)
        depth_ims.append(read_depth_im(path))
    depth_ims = np.stack(depth_ims)
    assert len(depth_ims.shape) == 3
    return depth_ims
    

def read_grd(path):
    """Reads a GAPS .grd file into a (tx, grd) pair."""
    with base_util.FS.open(path, 'rb') as f:
        content = f.read()
    res = struct.unpack('iii', content[:4 * 3])
    vcount = res[0] * res[1] * res[2]
    # if res[0] != 32 or res[1] != 32 or res[2] != 32:
    #   raise ValueError(f'Expected a resolution of 32^3 but got '
    #                    f'({res[0]}, {res[1]}, {res[2]}) for example {path}.')
    content = content[4 * 3:]
    tx = struct.unpack('f' * 16, content[:4 * 16])
    tx = np.array(tx).reshape([4, 4]).astype(np.float32)
    content = content[4 * 16:]
    grd = struct.unpack('f' * vcount, content[:4 * vcount])
    grd = np.array(grd).reshape(res).astype(np.float32)
    return tx, grd


def write_grd(path, volume, world2grid=None):
  """Writes a GAPS .grd file containing a voxel grid and world2grid matrix."""
  volume = np.squeeze(volume)
  assert len(volume.shape) == 3
  header = [int(s) for s in volume.shape]
  if world2grid is not None:
    header += [x.astype(np.float32) for x in np.reshape(world2grid, [16])]
  else:
    header += [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
  header = struct.pack(3*'i' + 16*'f', *header)
  content = volume.astype('f').tostring()
  with base_util.FS.open(path, 'wb') as f:
    f.write(header)
    f.write(content)