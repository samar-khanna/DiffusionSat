# DiffusionSat (ICLR 2024)
**[Website](https://samar-khanna.github.io/DiffusionSat/)** | 
**[Paper](https://arxiv.org/abs/2312.03606)**   |
**[Video](https://slideslive.com/39018155/diffusionsat-a-generative-foundation-model-for-satellite-imagery)**  |
**[Zenodo](https://zenodo.org/communities/diffusionsat)**  

This is the official repository for the ICLR 2024 paper 
"_DiffusionSat: A Generative Foundation Model For Satellite Imagery_".  

Authors: 
[Samar Khanna](https://www.samarkhanna.com) <sup>1</sup>, 
[Patrick Liu](https://web.stanford.edu/~pliu1/), 
[Linqi (Alex) Zhou](https://alexzhou907.github.io), 
[Chenlin Meng](https://chenlin9.github.io/), 
[Robin Rombach](https://github.com/rromb), 
[Marshall Burke](https://web.stanford.edu/~mburke/), 
[David B. Lobell](https://earth.stanford.edu/people/david-lobell#gs.5vndff), 
[Stefano Ermon](https://cs.stanford.edu/~ermon/).

## Installation
We use conda to create our environments. You will have to do the following:
```bash
cd DiffusionSat 
conda create -n diffusionsat python=3.10

# if you want cuda 11.8, replace the index url with https://download.pytorch.org/whl/cu118
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121
pip install -e ".[torch]"  # install editable diffusers
pip install -r requirements_remaining.txt
```

## Model checkpoint files
Model checkpoint files have been uploaded to Zenodo within the DiffusionSat community at [this link](https://zenodo.org/communities/diffusionsat).

**OLD**: Model checkpoint files were previously available [on Google Drive](https://drive.google.com/drive/u/2/folders/1p6nk4S5IpZEck3_xLo2hcI2Ha3P8LiA9).
Note that the files on Google Drive may not be forever available, and could be taken down at any moment.  
(While the files are on Google Drive, you can use [gdown](https://github.com/wkentaro/gdown) to download them).

## Single Image Generation
This section covers image-generation using single-image DiffusionSat, without control signal inputs.
The relevant jupyter notebook can be found in `notebooks/single-image.ipynb`. 

The relevant model checkpoints can be found here:  

| Resolution | Zenodo Page                                 | Download Link                                                                                                |
|------------|---------------------------------------------|--------------------------------------------------------------------------------------------------------------|
| 512 x 512  | [View](https://zenodo.org/records/13751498) | [Download](https://zenodo.org/records/13751498/files/finetune_sd21_sn-satlas-fmow_snr5_md7norm_bs64.zip)     |
| 256 x 256  | [View](https://zenodo.org/records/13756199) | [Download](https://zenodo.org/records/13756199/files/finetune_sd21_256_sn-satlas-fmow_snr5_md7norm_bs64.zip) |


## Conditional Image Generation

The Jupyter notebook that demonstrates generation with 3D ControlNets is shown
for the Texas housing dataset in `notebooks/controlnet_texas_samples.ipynb`. 
Generating with a ControlNet that accepts a single conditioning image + metadata is similar.

The relevant model checkpoints can be found here:

| Task                                  | Zenodo Page                                 | Download Link                                                                                     |
|---------------------------------------|---------------------------------------------|---------------------------------------------------------------------------------------------------|
| Texas Super-resolution                | [View](https://zenodo.org/records/13756211) | [Download](https://zenodo.org/records/13756211/files/controlnet3d-mixattn_sd21_md7norm_texas.zip) |
| fMoW Sentinel -> RGB Super-resolution | [View](https://zenodo.org/records/13756246) | [Download](https://zenodo.org/records/13756246/files/controlnet_sd21_md7norm_fmow_condres256.zip) |

## Training
These sections describe how to launch training using `accelerate`.


#### A note on `accelerate`
In this repository, we provide an example config file to use with `accelerate` in `launch_accelerate_configs`. 
You can also configure your own file by running `accelerate config` in your terminal and following the steps. 
This will save the config file in the  cache location (eg: `.cache/huggingface/accelerate/default_config.yaml`), 
and you can simply copy over the `.yaml` file to `launch_accelerate_configs/` or remove the 
`--config_file` argument from `accelerate launch` in the bash script.


#### A note on datasets
See [this section](#datasets) for more details on how to use `webdataset` for training. 
You will need to specify the dataset shardlist `.txt` files in `./datasets`.


### Single-Image Training
To train the `(text, metadata) -> single_image` model, use the following command:
```shell
./launch_scripts/launch_256_fmow_satlas_spacenet_img_txt_md.sh launch_accelerate_configs/single_gpu_accelerate_config.yaml
```

### Conditional (ControlNet) Training
To train the `(text, target_metadata, conditioning_metadata, conditioning_images) -> single_image` ControlNet model, use the following commands, 
detailed below.  

As a quick summary, these scripts use a frozen single-image model (see above) as a prior to 
train a ControlNet (which could be a 3D ControlNet for temporal conditioning images). 
This ControlNet can then generate a new image for the desired input text and metadata prompt, 
conditioned on additional metadata and images.

You will also need to provide the path to the single-image model checkpoint 
(by specifying this path in the `UNET_PATH` variable) that will remain frozen throughout training.

#### Texas Housing Super-resolution
```shell
./launch_scripts/launch_texas_md_controlnet.sh launch_accelerate_configs/single_gpu_accelerate_config.yaml
```
This task uses the Texas housing dataset from [satellite-pixel-synthesis-pytorch](https://github.com/KellyYutongHe/satellite-pixel-synthesis-pytorch).
The task is: given a low-res and high-res image of a location at time `T`, and a low-res image of the same location 
at time `T'`, generate a high-res image of the location at time `T'`.

#### fMoW-Sentinel -> fMoW-RGB Super-resolution
```shell
./launch_scripts/launch_fmow_md_superres.sh launch_accelerate_configs/single_gpu_accelerate_config.yaml
```
The task is: given a multi-spectral low-res image of a location (from [fMoW-Sentinel](https://github.com/sustainlab-group/SatMAE?tab=readme-ov-file#fmow-sentinel-dataset)), 
generate the corresponding high-res RGB image (from [fMoW-RGB](https://github.com/fMoW/dataset)).

#### fMoW Temporal Generation
```shell
./launch_scripts/launch_fmow_temporal_md_controlnet.sh launch_accelerate_configs/single_gpu_accelerate_config.yaml
```
This model conditions on a temporal sequence of input RGB images from fMoW-RGB to generate a single new image at a desired timestamp `T`.

#### xBD Temporal Inpainting
```shell
./launch_scripts/launch_xbd_md_controlnet.sh launch_accelerate_configs/single_gpu_accelerate_config.yaml
```
The task is: given a past (or future) image of a location affected by a natural disaster, generate the future (or past) image
after (or before) the natural disaster struck. We use the [xBD](https://github.com/DIUx-xView/xView2_baseline?tab=readme-ov-file#data-downloads) dataset.

## Datasets
The datasets we use are in [`webdataset`](https://github.com/webdataset/webdataset) format.
You will need to prepare your datasets in this format to be able to train using the given code,
or you can modify the data-loading to use your own custom dataset formats.

We have provided example shardlists in `datasets`. The training code will read the relevant file,
and load data using the data paths in this file. The advantage of using `webdataset` is that your data
does not need to only be on disk, and you can stream data from buckets in AWS S3 as well.  

We also provide a small sample `webdataset` in `datasets/texas_housing_val_10sample.tar`, sourced from 
the validation set of the Texas housing super-resolution task.


#### fMoW
Example format for each entry in the fMoW webdataset `.tar` file.
```
__key__: fmow-{cls_name}-{instance_id}  # eg: fmow-airport-airport_0
output.cls: label_idx  # eg: 32
input.npy: (h,w,c) numpy array
metadata.json: {'img_filename': ..., 'gsd': ..., 'cloud_cover': ..., 'timestamp': ..., 'country_code': ...}
```
Note that fMoW also requires metadata `.csv` files, which have been provided in `datasets/fmow-train-meta.csv` 
and `datasets/fmow-val-meta.csv`.

## Citation
If you find our project helpful, please cite our paper:
```
@inproceedings{
khanna2024diffusionsat,
title={DiffusionSat: A Generative Foundation Model for Satellite Imagery},
author={Samar Khanna and Patrick Liu and Linqi Zhou and Chenlin Meng and Robin Rombach and Marshall Burke and David B. Lobell and Stefano Ermon},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=I5webNFDgQ}
}
```
