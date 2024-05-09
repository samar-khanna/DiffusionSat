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
_Coming soon, stay tuned..._

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
