#####################################################################################################################
# SET ME!!!
ROOT = "/cluster_HDD/lothlann/ppalafox"
# SET ME!!!
#####################################################################################################################

data_base_dir = f"{ROOT}/datasets_SSD@20_04_2021"
exp_dir       = f"{ROOT}/experiments"


#####################################################################################################################
# MODELS
#####################################################################################################################

shape_network_specs = {
    "dims" : [512] * 8,
    "dropout" : None, #[0, 1, 2, 3, 4, 5, 6, 7],
    "dropout_prob" : 0.05,
    "norm_layers" : [0, 1, 2, 3, 4, 5, 6, 7],
    "latent_in" : [4],
    "xyz_in_all" : False,
    "latent_dropout" : False,
    "weight_norm" : True,
    "positional_enc": True, 
    "n_positional_freqs": 8,
}

pose_network_specs = {
    "dims" : [1024] * 8,
    "dropout" : None, #[0, 1, 2, 3, 4, 5, 6, 7],
    "dropout_prob" : 0.2,
    "norm_layers" : [0, 1, 2, 3, 4, 5, 6, 7],
    "latent_in" : [4],
    "xyz_in_all" : False,
    "latent_dropout" : False,
    "weight_norm" : True,
    "positional_enc": True,
    "n_positional_freqs": 8,
    "n_alpha_epochs": 0, 
}

shape_codes_dim = 384
pose_codes_dim = 384

#####################################################################################################################
# DATA
#####################################################################################################################

############################################################################################
############################################################################################
only_shape = False # Set to True when bootstrapping the ShapeMLP
############################################################################################
############################################################################################

if only_shape:
    cache_data = True
else:
    cache_data = False

# --------------------------------------------------------------------------------------- #
# Choose a name for the folder where we will be storing experiments
exp_version="npms"
# --------------------------------------------------------------------------------------- #

############################################################################################
############################################################################################
#############
### SHAPE ###
#############
shape_dataset_name = ""

#############
### POSE ###
#############
pose_dataset_name = ""

train_dataset_name = shape_dataset_name if only_shape else pose_dataset_name
############################################################################################
############################################################################################

#####################################################################################################################
#####################################################################################################################

batch_size  = 8 if only_shape else 8
num_workers = 10 if only_shape else 20

## SDF samples
sdf_samples_types = {
    "surface": 0.0,
    "near": 0.7,
    "uniform": 0.3,
}
num_sdf_samples = 50000 if only_shape else 0

## Flow samples
sample_flow_dist   = [1, 1]
sample_flow_sigmas = [0.002, 0.01] # [0.01, 0.1] #[0.002, 0.01]
num_flow_points    = 50000 if not only_shape else 0

sample_info = {
    'sdf': {
        'types': sdf_samples_types,
        'num_points': num_sdf_samples,
    },
    'sdf_all': {
        'num_points': num_sdf_samples,
    },
    'flow': {
        'dist': sample_flow_dist,
        'sigmas': sample_flow_sigmas,
        'num_points': num_flow_points,
    }
}

weight_correspondences = False
weights_dict = {
    'use_procrustes': False,
    'min_rigidity_distance': 0.1,
    'rigidity_scale':        2.0,
    'rigidity_max_weight':   2.0,
}

#####################################################################################################################
# TRAINING OPTIONS
#####################################################################################################################

use_se3 = False

use_curriculum = False
curriculum_dist = [0.1, 0.3, 0.5] # Curriculum distribution within the total number of steps

# SDF OPTIONS
enforce_minmax = True
clamping_distance = 0.1

# Set it to None to disable this "sphere" normalization | 1.0
code_bound = None

epochs       = 4000 if only_shape else 200
epochs_extra =    0 if only_shape else 0
eval_every   =  100 if only_shape else 1
interval     =  500 if only_shape else 30

optimizer = 'Adam'

##################################################################################
##################################################################################
# Set to True if you wanna start training from a given checkpoint, which
# you need to specify below
# If False, we check whether we "continue_from" 
init_from = True
##################################################################################
##################################################################################

# If we're only training the shape latent space, we typically wanna start from scratch
if only_shape:
    init_from = False

# Set the init checkpoint if necessary
if init_from:
    continue_from = False # Since we will initialize from a checkpoint, we set continue_from to False

    # By default, init_from_pose is set to False, but you can set it to a certain checkpoint if you want to start the Pose MLP from a checkpoint different than the Shape MLP
    init_from_pose = False
    
    init_from = ""
    checkpoint = 0

else:
    continue_from = ""

    continue_from = False if only_shape else continue_from

##################################################################################
# Load modules
if continue_from:
    assert not only_shape
    # In the current implementation, we only allow to "continue_from" if we're learning the pose space
    load_shape_decoder   = True
    load_shape_codes     = True
    freeze_shape_decoder = True
    freeze_shape_codes   = True

    load_pose_decoder    = True
    load_pose_codes      = True
    freeze_pose_decoder  = False
    freeze_pose_codes    = False
else:
    load_shape_decoder   = True if not only_shape else False
    load_shape_codes     = True if not only_shape else False
    freeze_shape_decoder = True if not only_shape else False
    freeze_shape_codes   = True if not only_shape else False

    load_pose_decoder    = False
    load_pose_codes      = False
    freeze_pose_decoder  = False
    freeze_pose_codes    = False
##################################################################################

lambdas_sdf = {
    'ref':  0 if not only_shape else 1,
    "flow": 1 if not only_shape else 0,
}

if lambdas_sdf['ref'] > 0:
    assert num_sdf_samples > 0

if lambdas_sdf['flow'] > 0:
    assert num_flow_points > 0

do_code_regularization = True
shape_reg_lambda       = 1e-4
pose_reg_lambda        = 1e-4

lr_dict = {
    "shape_decoder": 0.0005, #0.00025, #0.0005,    
    "pose_decoder":  0.0005, #0.00025, #0.0005,    
    "shape_codes":   0.001, #0.0005 , #0.001,    
    "pose_codes":    0.001, #0.0005 , #0.001,    
}

learning_rate_schedules = {
    "shape_decoder": {
        "type": "step",
        "initial": lr_dict['shape_decoder'],
        "interval": interval,
        "factor": 0.5,
    },
    "pose_decoder": {
        "type": "step",
        "initial": lr_dict['pose_decoder'],
        "interval": interval,
        "factor": 0.5,
    },
    "shape_codes": {
        "type": "step",
        "initial": lr_dict['shape_codes'],
        "interval": interval,
        "factor": 0.5,
    },
    "pose_codes": {
        "type": "step",
        "initial": lr_dict['pose_codes'],
        "interval": interval,
        "factor": 0.5,
    },
}