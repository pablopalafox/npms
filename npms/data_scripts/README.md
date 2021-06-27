Within [data_scripts](npms/data_scripts) you can find helpful scripts to generate your own `labels.json` and `labels_tpose.json` from a dataset:

- [prepare_labels_multi_tpose.py](npms/data_scripts/prepare_labels_multi_tpose.py): generate labels for training the shape latent space.

- [prepare_labels_multi_tpose.py](npms/data_scripts/prepare_labels_multi.py): generate labels for training the pose latent space.

- [prepare_labels_shape_encoder.py](npms/data_scripts/prepare_labels_shape_encoder.py): generate labels for training a shape encoder for initializing shape codes before test-time optimization (have a look at the [paper](https://pablopalafox.github.io/npms/palafox2021npms.pdf)) if you don't know what I'm talking about.

- [prepare_labels_pose_encoder.py](npms/data_scripts/prepare_labels_pose_encoder.py): generate labels for training a pose encoder for initializing pose codes before test-time optimization.

- [prepare_labels_subsample.py](npms/data_scripts/prepare_labels_subsample.py): subsample an existing labels folder.

- [prepare_labels_merge_multi*.py](npms/data_scripts/prepare_labels_merge_multi.py): merge existing labels.