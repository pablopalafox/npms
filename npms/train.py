import os
import argparse
import torch
import datasets.sdf_dataset as sdf_dataset
from models import training
from data_scripts.compute_mapping import compute_train_to_augmented_mapping
import utils.utils as utils

import config as cfg
print(f"Loaded {cfg.config_dataset} config. Continue?")


parser = argparse.ArgumentParser(
    description='Run Model'
)

parser.add_argument('-d', '--debug', action='store_true')
parser.add_argument('-n', '--extra_name', default="", type=str)

try:
    args = parser.parse_args()
except:
    args = parser.parse_known_args()[0]

###############################################################################################
# Dataset and Experiment dirs
###############################################################################################

data_dir = cfg.data_base_dir
exp_dir  = os.path.join(cfg.exp_dir, cfg.exp_version)

print()
print("DATA_DIR:", data_dir)
print("EXP_DIR:", exp_dir)
print()

from utils.parsing_utils import get_dataset_type_from_dataset_name
dataset_type = get_dataset_type_from_dataset_name(cfg.train_dataset_name)
splits_dir = f"{cfg.splits_dir}_{dataset_type}"

train_labels_json = os.path.join(data_dir, splits_dir, f"{cfg.train_dataset_name}", "labels.json")

# Mapping from training identities to the (possibly augmented) identities we trained the shape MLP on
train_to_augmented = {}
if not cfg.only_shape and cfg.init_from: 
    assert cfg.shape_dataset_name in cfg.init_from, "The pre-trained shape model you are using was not trained on the specified 'shape_dataset_name'"
    assert cfg.pose_dataset_name == cfg.train_dataset_name
    
    train_to_augmented = compute_train_to_augmented_mapping(
        data_dir, splits_dir,
        cfg.shape_dataset_name,
        cfg.train_dataset_name,
    )

###############################################################################################
###############################################################################################

assert os.path.isfile(train_labels_json), train_labels_json

labels_json = {
    'train': train_labels_json,
}

###############################################################################################
###############################################################################################

print("TRAIN DATASET...")
train_dataset = sdf_dataset.SDFDataset(
    data_dir=data_dir,
    labels_json=labels_json['train'],
    batch_size=cfg.batch_size, 
    num_workers=cfg.num_workers,
    sample_info=cfg.sample_info,
    cache_data=cfg.cache_data
)

if not cfg.continue_from:
    exp_name = utils.build_exp_name(
        extra_name=args.extra_name,
    )
else:
    print("#"*20)
    print("Continuing from...",)
    print("#"*20)
    exp_name = cfg.continue_from

# Initialize trainer
trainer = training.Trainer(
    args.debug,
    torch.device("cuda"), 
    train_dataset,
    exp_dir, exp_name,
    train_to_augmented
)

########################################################################################################################
###################################################### PRINT PARAMS ####################################################
########################################################################################################################

print()
print("#"*60)
print("#"*60)
print()
print("exp_name:")
print(exp_name)
print()
if cfg.only_shape:
    print("LEARNING SHAPE SPACE")
else:    
    print("LEARNING POSE SPACE")
print()

print()
print("Use SE(3)", cfg.use_se3)
print()

print()
print("SHAPE ------------------------------")
print()
for k, v in cfg.shape_network_specs.items(): print(f"{k:<40}, {v}")
print()
print("POSE  ------------------------------")
print()
for k, v in cfg.pose_network_specs.items(): print(f"{k:<40}, {v}")

print()
print("#"*60)
print("#"*60)

# if not args.debug:
#     print()
#     input("Verify the params above and press enter if you want to go ahead...")

########################################################################################################################
########################################################################################################################

trainer.train_model()

print()
print("Training done!")
print()