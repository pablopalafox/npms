#################################################
# SET ME
#################################################
config_dataset = "HUMAN"
#################################################

splits_dir = "ZSPLITS"

if config_dataset == "HUMAN":
    from configs_train.config_train_HUMAN import *
elif config_dataset == "MANO":
    from configs_train.config_train_MANO import *
else:
    raise Exception('bad config dataset')

