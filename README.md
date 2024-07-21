# DiffusionSat (ICLR 2024)
**[Website](https://samar-khanna.github.io/DiffusionSat/)** | 
**[Paper](https://arxiv.org/abs/2312.03606)**   |
**[Video](https://recorder-v3.slideslive.com/?share=92102&s=22fca8d7-2deb-4bf0-af4a-02d1839dc69b)**

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
Please refer to `INSTALLATION.md`.

## Model checkpoint files
Model checkpoint files can be found at [this link](https://drive.google.com/drive/u/2/folders/1p6nk4S5IpZEck3_xLo2hcI2Ha3P8LiA9).
Note that this is a temporary location, and checkpoint files may be moved to another location in the future (I will update the README in this case).  
(While the files are on Google Drive, you can use [gdown](https://github.com/wkentaro/gdown) to download them).

## Single Image Generation
This section covers image-generation using single-image DiffusionSat, without control signal inputs.
The relevant jupyter notebook can be found in `notebooks/single-image.ipynb`. 

The relevant model checkpoints can be found here:  

| Resolution | Link     |
|------------|----------|
| 512 x 512  | [Download](https://drive.google.com/drive/u/2/folders/1zddxoEVNpbffIti8gUbhCRoL1w-tkMK2) |
| 256 x 256  | [Download](https://drive.google.com/drive/u/2/folders/1SZnVpIaYyWN7WbAM7Njn-DDOZ7gRCmQ9) |


## Conditional Image Generation
_Coming soon, stay tuned..._

## Training
These sections describe how to launch training using `accelerate`.

### Single-Image Training
To train the `(text, metadata) -> single_image` model, use the following command:
```shell
./launch_scripts/launch_256_fmow_satlas_spacenet_img_txt_md.sh launch_accelerate_configs/single_gpu_accelerate_config.yaml
```
Here we provide an example config file to use with `accelerate`, but you can also configure your own
file by running `accelerate config` and following the steps. This will save the config file in the 
cache location (eg: `.cache/huggingface/accelerate/default_config.yaml`), 
and you can simply copy over the `.yaml` file to `launch_accelerate_configs/` or remove the 
`--config_file` argument from `accelerate launch` in the bash script.

## Datasets
The datasets we use are in [`webdataset`](https://github.com/webdataset/webdataset) format.
You will need to prepare your datasets in this format to be able to train using the given code,
or you can modify the data-loading to use your own custom dataset formats.

We have provided example shardlists in `webdataset_shards`. The training code will read the relevant file,
and load data using the data paths in this file. The advantage of using `webdataset` is that your data
does not need to only be on disk, and you can stream data from buckets in AWS S3 as well.


#### fMoW
Example format for each entry in the fMoW webdataset `.tar` file.
```
__key__: fmow-{cls_name}-{instance_id}  # eg: fmow-airport-airport_0
output.cls: label_idx  # eg: 32
input.npy: (h,w,c) numpy array
metadata.json: {'img_filename': ..., 'gsd': ..., 'cloud_cover': ..., 'timestamp': ..., 'country_code': ...}
```
Note that fMoW also requires a metadata `.csv` file.

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
