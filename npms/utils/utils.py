import sys
from datetime import datetime
import numpy as np
from PIL import Image
import math
import json

import config as cfg


def build_exp_name(
    initial_exp_name="NPMs",
    extra_name="",
):
    exp_name = initial_exp_name

    # Date
    date = datetime.now().strftime('%Y-%m-%d')
    exp_name = f'{date}__{exp_name}'

    # SDF samples
    if cfg.only_shape:
        exp_name += "__SHAPE"
        exp_name += f"_nss{cfg.sdf_samples_types['near']}"
        exp_name += f"_uni{cfg.sdf_samples_types['uniform']}"
    else:
         exp_name += "__POSE"
        # exp_name += f"_lbdaFlow{cfg.lambdas_sdf['flow']}"

    # Batch size
    exp_name += f'__bs{cfg.batch_size}'

    # Learning rate
    lr_text = "lr"
    for _, lr in cfg.lr_dict.items():
        lr_text += "-" + str(lr)
    exp_name += f'__{lr_text}'

    # Interval
    exp_name += f"_intvl{cfg.interval}"

    # Shape code info
    exp_name += f"__s{cfg.shape_codes_dim}-{cfg.shape_network_specs['dims'][0]}-{len(cfg.shape_network_specs['dims'])}l"

    # Pose code info
    exp_name += f"__p{cfg.pose_codes_dim}-{cfg.pose_network_specs['dims'][0]}-{len(cfg.pose_network_specs['dims'])}l"

    # Use SE3
    exp_name += "__wSE3" if cfg.use_se3 else "__woSE3"

    # Positional encoding
    exp_name += "__wShapePosEnc" if cfg.shape_network_specs['positional_enc'] else "__woShapePosEnc"
    exp_name += "__wPosePosEnc"  if cfg.pose_network_specs['positional_enc']  else "__woPosePosEnc"

    # Dropout
    exp_name += f"__wDroutS{cfg.shape_network_specs['dropout_prob']}" if cfg.shape_network_specs['dropout'] is not None else "__woDroutS"
    exp_name += f"__wDroutP{cfg.shape_network_specs['dropout_prob']}" if cfg.pose_network_specs['dropout'] is not None else "__woDroutP"

    # BN
    exp_name += f"__wWNormS" if cfg.shape_network_specs['weight_norm'] else "__woWNormS"
    exp_name += f"__wWNormP" if cfg.pose_network_specs['weight_norm'] else "__woWNormP"

    # Dataset name
    exp_name += f'__ON__{cfg.train_dataset_name}'

    # Extra name
    if len(extra_name) > 0:
        exp_name += f"__{extra_name}"

    return exp_name


def save_grayscale_image(filename, image_numpy):
    image_to_save = np.copy(image_numpy)
    image_to_save = (image_to_save * 255).astype(np.uint8)
    
    if len(image_to_save.shape) == 2:
        io.imsave(filename, image_to_save)
    elif len(image_to_save.shape) == 3:
        assert image_to_save.shape[0] == 1 or image_to_save.shape[-1] == 1
        image_to_save = image_to_save[0]
        io.imsave(filename, image_to_save)


def show_mask_image(image_numpy):
    assert image_numpy.dtype == np.bool
    image_to_show = np.copy(image_numpy)
    image_to_show = (image_to_show * 255).astype(np.uint8)
    
    img = Image.fromarray(image_to_show)
    img.show()


def unnorm_gaussian_kernel_3d(k, sigma=1.0):                                                                                                                                                                                                                                                                                             
    """
    Creates gaussian kernel of size k and a sigma of sigma.
    """

    ax = np.arange(-k//2 + 1.0, k//2 + 1.0)
    xx, yy, zz = np.meshgrid(ax, ax, ax)

    kernel = np.exp(-(xx**2 + yy**2 + zz**2) / (2.0 * sigma**2))

    return np.asarray(kernel, dtype=np.float32)


def gaussian_kernel_3d(k, sigma=1.0):                                                                                                                                                                                                                                                                                             
    """
    Creates gaussian kernel of size k and a sigma of sigma.
    """

    ax = np.arange(-k//2 + 1.0, k//2 + 1.0)
    xx, yy, zz = np.meshgrid(ax, ax, ax)

    kernel = (1.0 / math.sqrt(2.0 * math.pi * sigma**2)) * np.exp(-(xx**2 + yy**2 + zz**2) / (2.0 * sigma**2))

    return np.asarray(kernel, dtype=np.float32)
     

# From https://stackoverflow.com/questions/3041986/apt-command-line-interface-like-yes-no-input
def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")


import os, shutil
def copytree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)


def compute_dataset_mapping(data_dir, dataset_orig_name, dataset_new_name):
    # Original
    if isinstance(dataset_orig_name, list):
        orig_labels_tpose = dataset_orig_name
    else:
        orig_labels_tpose_json = os.path.join(data_dir, dataset_orig_name, "labels_tpose.json")
        assert os.path.isfile(orig_labels_tpose_json), orig_labels_tpose_json

        with open(orig_labels_tpose_json) as f:
            orig_labels_tpose = json.loads(f.read())

    # New
    if isinstance(dataset_new_name, list):
        new_labels_tpose = dataset_new_name
    else:
        new_labels_tpose_json = os.path.join(data_dir, dataset_new_name, "labels_tpose.json")
        assert os.path.isfile(new_labels_tpose_json), new_labels_tpose_json

        with open(new_labels_tpose_json) as f:
            new_labels_tpose = json.loads(f.read())

    new_to_orig = {}

    for i, new_label in enumerate(new_labels_tpose):
        for j, orig_label in enumerate(orig_labels_tpose):
            if new_label['identity_name'] == orig_label['identity_name']:
                new_to_orig[i] = j

    return new_to_orig


def filter_identities(target_identities, exclude_identities):
    target_identities_tmp = []
    for ti in target_identities:
        discard = False
        for exclude_identity in exclude_identities:
            if exclude_identity in ti:
                discard = True
                break
        if discard:
            continue
        target_identities_tmp.append(ti)

    return target_identities_tmp