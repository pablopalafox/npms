import os


#####################################################################################################################
# SET ME!!!
ROOT = "/cluster_HDD/lothlann/ppalafox"
# SET ME!!!
#####################################################################################################################

data_dir = f"{ROOT}/datasets_SSD@20_04_2021"
exp_dir  = f"{ROOT}/experiments"

exp_version = "npms"
exps_dir  = os.path.join(exp_dir, exp_version)

########################################################################################################################
# Options
########################################################################################################################
cache_data = True

# if not cache_data:
#     input("We are not caching the dataset!!")

# For MarchingCubes
mult = 2
reconstruction_res = mult * 256
max_batch = (mult * 32)**3
if reconstruction_res != 512:
    print("!"*60)
    print("THINK ABOUT INCREASING THE RESOLUTION FOR THE FINAL RESULTS")
    print("!"*60)

# Shape
use_shape_encoder = True
freeze_shape_codes = False

# Pose
use_pose_encoder = True
if not use_pose_encoder:
    init_from_pretrained_pose_codes = False
    init_from_mean = True; add_noise_to_initialization = False; sigma_noise = 0.1

warp_reconstructed_mesh = True

#######################################################################################################
# Solver iterations
#######################################################################################################
do_optim = True
num_iterations = 500 #1000
num_snapshots = 5
code_snapshot_every = min(num_iterations, max(1, int(num_iterations / num_snapshots)))

interval = int(num_iterations // 4)
factor = 0.5

if freeze_shape_codes:
    pose_bootstrap_iterations = None
else:
    pose_bootstrap_iterations = None

# Alternate between shape and pose code optim
alternate_shape_pose = False

############################################################################################################
# Learning rates for the shape and pose codes
############################################################################################################
lr_dict = {
    "shape_codes": 0.0005,
    "pose_codes":  0.001, 
}

############################################################################################################
# Code reg
############################################################################################################
code_reg_lambdas = {
    'shape': 1e-1,
    'pose':  1e-4,
}
shape_code_reg_scale = 1e-6 # if 'code_reg_lambdas['shape]' is , then 'shape_code_reg_scale' variable is useless

############################################################################################################
# Temporal reg
############################################################################################################
temporal_reg_lambda = 100 # 100
code_consistency_lambda = 0 # 10000

############################################################################################################
# Schedules
############################################################################################################
learning_rate_schedules = {
    "shape_codes": {
        "type": "step",
        "initial": lr_dict['shape_codes'],
        "interval": interval,
        "factor": factor,
    },
    "pose_codes": {
        "type": "step",
        "initial": lr_dict['pose_codes'],
        "interval": interval,
        "factor": factor,
    },
}

############################################################################################################
# Optim dict
############################################################################################################
optim_dict = {
    # shape
    's': {
        'code_reg_lambda': code_reg_lambdas['shape'],
        'code_bound': None
    },
    # pose
    'p': {
        'code_reg_lambda': code_reg_lambdas['pose'],
        'code_bound': None
    },
    'icp': {
        'lambda': 0.001, #0.0005,
        'iters': int(num_iterations / 2),
        'ns_eps': 0.001,
    }
}

############################################################################################################
# Use single-view or complete input
############################################################################################################
use_partial_input = True
use_sdf_from_ifnet = None
if not use_partial_input:
    input("Will use complete inputs. Continue?")
    use_sdf_from_ifnet = True
    if use_sdf_from_ifnet:
        input("Will use IFNet predictions. Continue?")


############################################################################################################
# resolutions
############################################################################################################
# Input res of the partial voxel grid that we feed to the encoders
encoder_res = 256
# Input res of the partial SDF grid that we use to optimize over
res_sdf = [256]

############################################################################################################
# For sampling points around the predicted tpose
############################################################################################################
total_sample_num = 500000
sigma = 0.015
num_samples = 20000

clamping_distance = 0.1
reduce_clamping = False

batch_size  = 3
num_workers = 10
shuffle = True
radius = 1

num_to_eval = -1
load_every = 1
from_frame_id = 0

##############################################
# MODELS
##############################################
shape_codes_dim = 256
pose_codes_dim = 256
shape_fs = 512
pose_fs = 512
positional_enc_shape = False
positional_enc_pose  = False
use_se3 = False
norm_layers_shape = None #[0, 1, 2, 3, 4, 5, 6, 7]
norm_layers_pose = None # [0, 1, 2, 3, 4, 5, 6, 7]
weight_norm_shape = False
weight_norm_pose = False

########################################
# EXPERIMENT
########################################

exp_name = "2021-03-15__NESHPOD__bs4__lr-0.0005-0.0005-0.001-0.001_intvl30__s256-512-8l__p256-1024-8l__woSE3__wShapePosEnc__wPosePosEnc__woDroutS__woDroutP__wWNormS__wWNormP__ON__MIX-POSE__AMASS-50id-5000__MIXAMO-165id-20000__CAPE-35id-20533"
checkpoint_epoch = 150
use_se3 = False
shape_codes_dim = 256
pose_codes_dim = 256
shape_fs = 512; pose_fs = 1024
positional_enc_shape = True
positional_enc_pose = True
norm_layers_shape = [0, 1, 2, 3, 4, 5, 6, 7]
norm_layers_pose = [0, 1, 2, 3, 4, 5, 6, 7]
weight_norm_shape = True
weight_norm_pose = True
exp_name_shape_encoder = "SHAPE_ENCODER_lr1e-05__cpt4000_p256_fs512_res256_adj0.1every2__ON__MIX-POSE__AMASS-50id-10349__MIXAMO-165id-40000__CAPE-35id-20533"
checkpoint_shape_encoder = "best"
exp_name_pose_encoder = "POSE_ENCODER__lr1e-06_bs4__cpt150_p256_fs1024_res256_adj0.5every15_encTypev3__ON__MIX-POSE__AMASS-50id-5000__MIXAMO-165id-20000__CAPE-35id-20533"
checkpoint_pose_encoder = "best"


print()
print("exp_name")
print(exp_name)
print()

# ----------------------------------------------------------------------------------------------------------------

shape_network_specs = {
    "dims" : [shape_fs] * 8,
    "dropout" : None, # [0, 1, 2, 3, 4, 5, 6, 7],
    "dropout_prob" : 0.2,
    "norm_layers" : norm_layers_shape,
    "latent_in" : [4],
    "xyz_in_all" : False,
    "latent_dropout" : False,
    "weight_norm" : weight_norm_shape,
    "positional_enc": positional_enc_shape, 
}
pose_network_specs = {
    "dims" : [pose_fs] * 8,
    "dropout" : None, # [0, 1, 2, 3, 4, 5, 6, 7],
    "dropout_prob" : 0.2,
    "norm_layers" : norm_layers_pose,
    "latent_in" : [4],
    "xyz_in_all" : False,
    "latent_dropout" : False,
    "weight_norm" : weight_norm_pose,
    "positional_enc": positional_enc_pose,
    "n_positional_freqs": 8,
    "n_alpha_epochs": 0, 
}

########################################################################################################################
########################################################################################################################

# ------------------------------------------------------------------------

##################
# DATASET NAME
##################

dataset_name = ""

# ------------------------------------------------------------------------
