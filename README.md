# NPMs: Neural Parametric Models

### [Project Page](https://pablopalafox.github.io/npms/) | [Paper](https://pablopalafox.github.io/npms/palafox2021npms.pdf) | [ArXiv](https://arxiv.org/abs/2104.00702) | [Video](https://youtu.be/muZXXgkkMPY)
<br />

> NPMs: Neural Parametric Models for 3D Deformable Shapes <br />
> [Pablo Palafox](https://pablopalafox.github.io/), [Aljaz Bozic](https://aljazbozic.github.io/), [Justus Thies](https://justusthies.github.io/), [Matthias Niessner](https://www.niessnerlab.org/members/matthias_niessner/profile.html), [Angela Dai](https://www.3dunderstanding.org/team.html)

<p align="center">
    <img width="100%" src="resources/teaser.gif"/>
</p>


#### Citation
    @article{palafox2021npms
        author        = {Palafox, Pablo and Bo{\v{z}}i{\v{c}}, Alja{\v{z}} and Thies, Justus and Nie{\ss}ner, Matthias and Dai, Angela},
        title         = {NPMs: Neural Parametric Models for 3D Deformable Shapes},
        journal       = {arXiv preprint arXiv:2104.00702},
        year          = {2021},
    }


## Install


You can either pull our docker image, build it yourself with the provided [Dockerfile](Dockerfile) or build the project from source.

#### Pull Docker Image
```
docker pull ppalafox/npms:latest
```

You can now run an interactive container of the image you just built (before that, navigate to [npms](npsm)):
```
cd npms
docker run --ipc=host -it --name npms --gpus=all -v $PWD:/app -v /cluster:/cluster npms:latest bash
```

#### Build Docker Image

Run the following from within the root of this project (where [Dockerfile](Dockerfile) lives) to build a docker image with all required dependencies.
```
docker build . -t npms
```

You can now run an interactive container of the image you just built (before that, navigate to [npms](npms)):
```
cd npms
docker run --ipc=host -it --name npms --gpus=all -v $PWD:/app -v /cluster:/cluster npms:latest bash
```

Of course, you'll have to specify you're own paths to the volumes you'd like to mount using the `-v` flag.


#### Build from source
A linux system with cuda is required for the project.

The [npms_env.yml](npms_env.yml) file contains (hopefully) all necessary python dependencies for the project.
To conveniently install them automatically with [anaconda](https://www.anaconda.com/) you can use:

```
conda env create -f npms_env.yml
conda activate npms
```

##### Other dependencies
We need some other dependencies. Starting from the root folder of this project, we'll do the following...

<!-- - Let's start by cloning [Eigen](https://gitlab.com/libeigen/eigen.git):
```
cd external
git clone https://gitlab.com/libeigen/eigen.git
``` -->

<!-- - Let's install [ChamferDistancePytorch](https://github.com/ThibaultGROUEIX/ChamferDistancePytorch.git). Since it uses JIT compilation, we don't need to do anything else other than cloning it:
```
git clone https://github.com/ThibaultGROUEIX/ChamferDistancePytorch.git
cd ..
``` -->

- Compile the [csrc]('csrc') folder: 
```
cd csrc 
python setup.py install
cd ..
```

- We need some libraries from [IFNet](https://github.com/jchibane/if-net). In particular, we need `libmesh` and `libvoxelize` from that repo. They are already placed within [external](external). (Check the corresponding [LICENSE](external/libmesh/LICENSE)). To build these, proceed as follows:

```
cd libmesh/
python setup.py build_ext --inplace
cd ../libvoxelize/
python setup.py build_ext --inplace
cd ..
```

- Install [gaps](https://github.com/tomfunkhouser/gaps.git). For this, we are using a couple of scripts from [LDIF](https://github.com/google/ldif), namely [external/build_gaps.sh](external/build_gaps.sh) and [external/gaps_is_installed.sh](external/gaps_is_installed.sh). We also need the folder [external/qview](external/qview), which also belongs to the [LDIF](https://github.com/google/ldif) project (it's already place within our [external](external)) folder. To build `gaps`, make sure you're within [external](external) and run:
```
chmod +x build_gaps.sh
./build_gaps.sh
```

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; You can make sure it's built properly by running:

```
chmod +x gaps_is_installed.sh
./gaps_is_installed.sh
```

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; You should get a "Ready to go!" as output.

- We already have [npms/data_processing/implicit_waterproofing.py](npms/data_processing/implicit_waterproofing.py), which belongs to the [IFNet](https://github.com/jchibane/if-net) project, so nothing to do here (same [IFNet LICENSE](external/libmesh/LICENSE) applies to this file).


- We also need some helper functions from [LDIF](https://github.com/google/ldif). Namely, [base_util.py](https://github.com/google/ldif/blob/master/ldif/util/base_util.py) and [file_util.py](https://github.com/google/ldif/blob/master/ldif/util/file_util.py). We have placed them already under [npms/utils](npms/utils).

You can now navigate back to the root folder: `cd ..`

## Data Preparation
As an example, let's have a quick overview of what the process would look like in order to generate training data from the [CAPE](https://github.com/qianlim/cape_utils.git) dataset.

Download their dataset, by [registering](https://cape.is.tue.mpg.de/en/sign_up) and accepting their terms. Once you've followed their steps to download the dataset, you should have a folder named `cape_release`. 

In [npms/configs_train/config_train_HUMAN.py](npms/configs_train/config_train_HUMAN.py), set the variable `ROOT` to point to the folder where you want your data to live in. Then:

```
cd <ROOT>
mkdir data
```

And place `cape_release` within `data`.

#### Download SMPL models
Register [here](https://smpl.is.tue.mpg.de/register) to get access to SMPL body models. Then, under the [downloads](https://smpl.is.tue.mpg.de/downloads) tab, download the models. Refer to https://github.com/vchoutas/smplx#model-loading for more details. 

From within the root folder of this project, run:

```
cd npms/body_model
mkdir smpl
```

And place the `.pkl` files you just downloaded under `npms/body_model/smpl`. Now change their names, such that you have something like: 

body_models<br/>
│── smpl<br/>
│  │── smpl<br/>
│  │  └── SMPL_FEMALE.pkl<br/>
│  │  └── SMPL_MALE.pkl<br/>
│  │  └── SMPL_NEUTRAL.pkl<br/>

#### Preprocess the raw CAPE

Now let's process the raw data in order to generate training samples for our NPM. 

```
cd npms/data_processing
python prepare_cape_data.py
```

Then, we normalize the preprocessed dataset, such that the meshes reside within a bounding box with boundaries `bbox_min=-0.5` and `bbox_max=0.5`.

```
# We're within npms/data_processing
python normalize_dataset.py
```

At this point, we can generate training samples for both the shape and the pose MLP. An extra step would be required if our t-poses (`<ROOT>/datasets/cape/a_t_pose/000000/mesh_normalized.ply`) were not watertight. We'd need to run [multiview_to_watertight_mesh.py](npms/data_processing/multiview_to_watertight_mesh.py). Since CAPE is already watertight, we don't need to worry about this.

##### About `labels.json` and `labels_tpose.json`
One last thing before actually generating the samples is to create some "labels" files that specify the paths to the dataset we wanna create. Under the folder [ZSPLITS_HUMAN](ZSPLITS_HUMAN) we have copied some examples.

Within it, you can find other folders containing datasets in the form of the paths to the actual data. For example, [CAPE-SHAPE-TRAIN-35id](ZSPLITS_HUMAN/CAPE-SHAPE-TRAIN-35id), which in turn contains two files: [labels_tpose](ZSPLITS_HUMAN/CAPE-SHAPE-TRAIN-35id/labels_tpose.json) and [labels](ZSPLITS_HUMAN/CAPE-SHAPE-TRAIN-35id/labels.json). They define datasets in a flexible way, by means of a list of dictionaries, where each dictionary holds the paths to a particular sample. You'll get a feeling of why we have a `labels.json` and `labels_tpose.json` by running the following sections to generate data, as well as when you dive into actually training a new NPM from scratch.

Go ahead and copy the folder [ZSPLITS_HUMAN](ZSPLITS_HUMAN) into `<ROOT>/datasets`, where `ROOT` is a path to your datasets that you can specify in [npms/configs_train/config_train_HUMAN.py](npms/configs_train/config_train_HUMAN.py). If you followed along until now, within `<ROOT>/datasets` you should already have the preprocessed `<ROOT>/datasets/cape` dataset.

```
# Assuming you're in the root folder of the project
cp -r ZSPLITS_HUMAN <ROOT>/datasets
```

> Note: within [data_scripts](npms/data_scripts) you can find helpful scripts to generate your own `labels.json` and `labels_tpose.json` from a dataset. Check out the [npms/data_scripts/README.md](npms/data_scripts/README.md) for a brief overview on these scripts.


##### SDF samples
Generate SDF samples around our identities in their t-pose in order to train the shape latent space. 
```
# We're within npms/data_processing
python sample_boundary_sdf_gaps.py
```

##### Flow samples
Generate correspondences from an identity in its t-pose to its posed instances. 
```
# We're within npms/data_processing
python sample_flow.py -sigma 0.01
python sample_flow.py -sigma 0.002
```

We're done with generating data for CAPE! This was just an example using CAPE, but as you've seen, the only thing you need to have is a dataset of meshes:
- we need t-pose meshes for each identity in the dataset, and we can use [multiview_to_watertight_mesh.py](npms/data_processing/multiview_to_watertight_mesh.py) to make these t-pose meshes watertight, to then sample points and their SDF values.
- for a given identity, we need to have surface correspondences between the t-pose and the posed meshes (but note that these posed meshes don't need to be watertight).

## Training an NPM

#### Shape Latent Space

Set `only_shape=True` in [config_train_HUMAN.py](npms/configs_train/config_train_HUMAN.py). Then, from within the [npms](npms) folder, start the training:

```
python train.py
```

#### Pose Latent Space

Set `only_shape=False` in [config_train_HUMAN.py](npms/configs_train/config_train_HUMAN.py). We now need to load the best checkpoint from training the shape MLP. For that, go to [config_train_HUMAN.py](npms/configs_train/config_train_HUMAN.py), make sure `init_from = True` in its first appearance in the file, and then set this same variable to your pretrained model name later in the file:

```
init_from = "<model_name>"
checkpoint = <the_epoch_number_you_want_to_load>
```

Then, from within the [npms](npms) folder, start the training:

```
python train.py
```

Once we reach convergence, you're done. You know have latent spaces of shape and pose that you can play with.

You could:
- fit your learned model to an monocular depth sequence ([Fitting an NPM to a Monocular Depth Sequence](#fitting-an-npm-to-a-monocular-depth-sequence))

- interpolate between two shape codes, or between two pose codes ([Latent-space Interpolation](#latent-space-interpolation))

- transfer poses from one identity to another ([Shape and Pose Transfer](#shape-and-pose-transfer))

## Fitting an NPM to a Monocular Depth Sequence

#### Code Initialization
When fitting an NPM to monocular depth sequence, it is recommended that we have a relatively good initialization of our shape and pose codes to avoid falling into local minima. To this end, we are gonna learn a shape and a pose encoder that map an input depth map to a shape and pose code, respectively. 

We basically use the shape and pose codes that we've learned during training time as targets for training the shape and pose encoders. You can use [prepare_labels_shape_encoder.py](npms/data_scripts/prepare_labels_shape_encoder.py) and [prepare_labels_pose_encoder.py](npms/data_scripts/prepare_labels_pose_encoder.py) to generate the dataset labels for this encoder training.

You basically have to train them like so: 

```
python encode_shape_codes.py
python encode_pose_codes.py
```

And regarding the data you need for training the encoder...

**Data preparation**: Take a look at the scripts [voxelize_multiview.py](npms/data_processing/voxelize_multiview.py) to prepare the single-view voxel grids that we require to train our encoders.


#### Test-time Optimization
Now you can fit NPMs to an input monocular depth sequence:

```
python fit_npm.py -o -d HUMAN -e <EXTRA_NAME_IF_YOU_WANT>
```

The `-o` flag for `optimize`; the `-d` flag for the kind of dataset (`HUMAN`, `MANO`) and the `-e` flag for appending a string to the name of the current optimization run.

You'll have to take a look at [config_eval_HUMAN.py](npms/configs_viz/config_eval_HUMAN.py) and set the name of your trained model (`exp_model`) and its hyperparameters, as well as the dataset name `dataset_name` you want to evaluate on.

It's definitely not the cleanest and easiest config file, sorry for that!

**Data preparation**: Take a look at the scripts [compute_partial_sdf_grid.py](npms/data_processing/compute_partial_sdf_grid.py) to prepare the single-view SDF grid that we assume as input at test-time.

#### Visualization

With the following script you can visualize your fitting. Have a look at [config_viz_OURS.py](npms/configs_viz/config_viz_OURS.py) and set the name of your trained model (`exp_model`) as well as the name of your optimization run (`run_name`) of test-time fitting you just computed.

```
python viz_all_methods.py -m NPM -d HUMAN
```

There are a bunch of other scripts for visualization. They're definitely not cleaned-up, but I kept them here anyways in case they might be useful for you as a starting point.

#### Compute metrics
```
python compute_errors.py -n <name_of_optimization_run>
```

## Latent-space Interpolation
Check out the files:
- [interpolate_shapes.py](npms/interpolate_shapes.py)
- [interpolate_poses.py](npms/interpolate_poses.py) 

## Shape and Pose Transfer
Check out the files:
- [transfer_shape.py](npms/transfer_shape.py)
- [transfer_pose.py](npms/transfer_pose.py)
- [transfer_pose_sequence.py](npms/transfer_pose_sequence.py)


## Pretrained Models
Download pre-trained models [here](https://drive.google.com/drive/folders/1nodmgLiPqvcv2enQWp3M0ptoDx6oou4p?usp=sharing)


## License
NPMs is relased under the MIT License. See the [LICENSE file](LICENSE) for more details.

Check the corresponding LICENSES of the projects under the [external](external) folder. 

For instance, we make use of [libmesh](external/libmesh) and [libvoxelize](external/libvoxelize), which come from [IFNets](https://github.com/jchibane/if-net). Please check [their LICENSE](https://github.com/jchibane/if-net#license). 

We need some helper functions from [LDIF](https://github.com/google/ldif/tree/b7060e4bd804ba3f93f46bd6fd1736a7c0dd92a7). Namely, [base_util.py](https://github.com/google/ldif/blob/b7060e4bd804ba3f93f46bd6fd1736a7c0dd92a7/ldif/util/base_util.py) and [file_util.py](https://github.com/google/ldif/blob/b7060e4bd804ba3f93f46bd6fd1736a7c0dd92a7/ldif/util/file_util.py), which should be already under [utils](utils). Check the license and copyright in those files.
