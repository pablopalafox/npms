import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import shutil

from data_scripts import config_data as cfg
import config as cfg_general
from utils.utils import filter_identities


if __name__ == "__main__":

    OVERWRITE = True

    # ------------------------------------------------------------------------
    # 1
    # ------------------------------------------------------------------------
    special_case = True
    split = "test"

    # ------------------------------------------------------------------------
    # 2
    # ------------------------------------------------------------------------
    only_identities_from_dataset = None
    # only_identities_from_dataset = "AMASS-SHAPE-TRAIN-50id"
    
    if only_identities_from_dataset is not None:
        input(f"Only identities from {only_identities_from_dataset}?")

    # ------------------------------------------------------------------------
    # 3
    # ------------------------------------------------------------------------
    cluster = "cluster_HDD"
    dataset = "cape" #"amass"
    
    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------

    from utils.parsing_utils import get_dataset_type_from_dataset_name
    dataset_type = get_dataset_type_from_dataset_name(dataset)
    splits_dir = f"{cfg_general.splits_dir}_{dataset_type}"

    data_base_dir = f"/{cluster}/lothlann/ppalafox/datasets"
    dataset_dir = f"{data_base_dir}/{dataset}"

    labels_tpose = []
    labels = []

    ################################################################################################
    # Maybe remove some extreme examples
    ################################################################################################
    extreme_samples = []
    filtered_extreme = False
    if "mixamo" in dataset:
        num_extreme_samples = 3866
        extreme_samples_json = os.path.join(dataset_dir, f"mixamo_extreme_samples_{num_extreme_samples}.json")
        print(extreme_samples_json)
        if os.path.isfile(extreme_samples_json):
            with open(extreme_samples_json, "r") as f:
                extreme_samples = json.loads(f.read())
            filtered_extreme = True

    print()
    if filtered_extreme:
        print(f"Filtered {len(extreme_samples)} samples!")
    else:
        print("No filtered samples!")
    ################################################################################################

    ################################################################################################
    target_identities = []
    target_animations = []

    if split == "train":

        if only_identities_from_dataset is not None:

            from_labels_tpose_json = os.path.join(data_base_dir, splits_dir, only_identities_from_dataset, "labels_tpose.json")
            with open(from_labels_tpose_json, 'r') as f:
                from_labels_tpose = json.loads(f.read())

            for l in from_labels_tpose:
                target_identities.append(l['identity_name'])
            
        else:
            target_identities = [
                d for d in os.listdir(dataset_dir) 
                if "SPLITS" not in d
                and not d.endswith("json")
                and not d.endswith("txt")
            ]
            target_identities = sorted(target_identities)

            if "mixamo" in dataset:
                target_identities = filter_identities(target_identities, cfg.test_identities_mixamo)
                target_identities = filter_identities(target_identities, cfg.identities_augmented)
            elif "amass" in dataset:
                target_identities = filter_identities(target_identities, cfg.test_identities_amass)
            elif "mano" in dataset:
                target_identities = filter_identities(target_identities, cfg.test_identities_mano)
            elif "cape" in dataset:
                target_identities = filter_identities(target_identities, cfg.test_identities_cape)
            else:
                raise Exception("Dataset not implemented")

    elif split == "test":
        if "mixamo" in dataset:
            target_identities = cfg.test_identities_mixamo
        elif "amass" in dataset:
            if False:
                target_identities = cfg.test_identities_amass
            else:
                target_identities = ['Transitionsmocap_s003']
                target_animations = ['motion010']
        elif "mano" in dataset:
            target_identities = cfg.test_identities_mano
        elif "cape" in dataset:
            target_identities = cfg.test_identities_cape
        elif "dfaust" in dataset:
            target_identities = None
        else:
            raise Exception("Dataset not implemented")

    ################################################################################################

    # Name we give our resulting dataset
    dataset_name = f"{dataset.upper()}-POSE-{split.upper()}"

    # If it's a special case...
    if special_case and split == "test":

        if "mixamo" in dataset:
            test_animations_by_identity = cfg.test_animations_mixamo_by_identity
        elif "cape" in dataset:
            test_animations_by_identity = cfg.test_animations_cape_by_identity
        elif "dfaust" in dataset:
            test_animations_by_identity = cfg.test_animations_dfaust_by_identity
        else:
            raise Exception("Dataset not implemented")
        
        target_identities = list(test_animations_by_identity.keys())
        target_animations = []
        for ident, anim in test_animations_by_identity.items():
            if ident in target_identities:
                target_animations.extend(anim)

        # dataset_name = f"{dataset.upper()}-POSE-{split.upper()}-MOVING_FOR_TRANSLATION"
        dataset_name = f"{dataset.upper()}-POSE-{split.upper()}-{target_identities[0]}-{target_animations[0]}"
        # dataset_name = f"{dataset.upper()}-POSE-{split.upper()}-REAL_SCANS_SELECTION"

    print()
    print("TARGET IDENTITIES:")
    print(target_identities, f"({len(target_identities)})")
    print(target_animations, f"({len(target_animations)})")
    print()
    input(f"Dataset name: {dataset_name}?")

    ################################################################################################

    num_identities = len(target_identities)
    num_samples = 0
    num_seqs    = 0

    info_by_identity = {}

    real_num_identities = 0

    for identity_id, identity_name in enumerate(sorted(target_identities)):

        identity_num_animations = 0
        identity_num_frames = 0

        sample_tpose = {
            'dataset': dataset,
            "identity_id": identity_id,
            "identity_name": identity_name,
            "animation_name": "a_t_pose",
            "sample_id": "000000"
        }

        labels_tpose.append(sample_tpose)

        identity_path = os.path.join(dataset_dir, identity_name)

        print("identity", identity_path)

        all_animation_names = [
            m for m in sorted(os.listdir(identity_path))
            if m != cfg_general.splits_dir and not m.endswith('json') and not m.endswith('npz')
        ]

        # Go over all animations for our character
        for animation_name in sorted(all_animation_names):

            if "a_t_pose" in animation_name:
                continue

            if len(target_animations) > 0 and animation_name not in target_animations:
                continue
            
            # Identity name for query --> we get rid of the "aug_small_0"
            identity_name_query = identity_name.split('_')[0]
            
            if "mixamo" in dataset: 
                assert animation_name in cfg.animations_by_identity[identity_name_query]
                assert animation_name != cfg_general.splits_dir

            animation_dir = os.path.join(identity_path, animation_name)

            assert os.path.isdir(animation_dir), animation_dir

            # Go over all samples in current animation
            for sample_id in sorted(os.listdir(animation_dir)):

                sample_dir = os.path.join(animation_dir, sample_id)

                # Skip if sample is in the extreme_samples list
                animation_sample_name = f"{animation_name}_{sample_id}"

                if animation_sample_name in extreme_samples:
                    print("Skipping")
                    continue
                
                if not os.path.isdir(sample_dir):
                    continue

                try:
                    sample_id_int = int(sample_id)
                except:
                    print(f"Skipping {sample_id}")
                    continue

                sample = {
                    'dataset': dataset,
                    'identity_id': identity_id,
                    'identity_name': identity_name,
                    'animation_name': animation_name,
                    'sample_id': sample_id 
                }

                labels.append(sample)

                num_samples += 1
                identity_num_frames += 1

            num_seqs += 1
            identity_num_animations += 1
        
        real_num_identities += 1

        info_by_identity[identity_name] = {
            'num_animations': identity_num_animations,
            'num_frames': identity_num_frames,
        }

    print()
    print("real_num_identities", real_num_identities)

    n_anims = 0
    n_frames = 0 

    for id_name, info in info_by_identity.items():
        print(f"id_name {id_name} - num animations {info['num_animations']} - num frames {info['num_frames']}")
        n_anims += info['num_animations']
        n_frames += info['num_frames']

    # print(num_identities, num_samples, num_seqs)
    # print(num_identities, n_frames, n_anims)

    ###############################################################################################################
    ###############################################################################################################
    from utils.parsing_utils import get_dataset_type_from_dataset_name
    splits_dir = cfg_general.splits_dir + f"_{get_dataset_type_from_dataset_name(dataset_name)}"
    
    extra_name = ""
    if filtered_extreme:
        extra_name = "-filt"
    
    labels_dir = os.path.join(data_base_dir, splits_dir, f"{dataset_name}-{num_identities}id-{num_samples}ts-{num_seqs}seqs{extra_name}")
    
    if OVERWRITE or not os.path.exists(labels_dir) and len(labels) > 0:

        if os.path.exists(labels_dir):
            shutil.rmtree(labels_dir)
        
        os.makedirs(labels_dir)
        
        labels_json             = os.path.join(labels_dir, "labels.json")
        train_labels_tpose_json = os.path.join(labels_dir, "labels_tpose.json")
        
        with open(labels_json, 'w') as f:
            json.dump(labels, f, indent=4)

        with open(train_labels_tpose_json, 'w') as f:
            json.dump(labels_tpose, f, indent=4)

        print()
        print("Done:", labels_json)

    else:
        print(f"Labels {labels_dir} already exists! Didn't overwrite it.")

    print("Total num train samples:", num_samples)
    print("Label generation done!")