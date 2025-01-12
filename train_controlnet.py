###########################################################################
# References:
# https://github.com/huggingface/diffusers/
###########################################################################

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import copy
import logging
import math
import os
import random
import tarfile
import shutil
from pathlib import Path

import signal
import einops
import pandas as pd
import webdataset as wds
import braceexpand

import accelerate
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UniPCMultistepScheduler,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

from diffusionsat.sat_unet import SatUNet
from diffusionsat.controlnet import ControlNetModel
from diffusionsat.controlnet_3d import ControlNetModel3D
from diffusionsat.multicontrolnet import MultiControlNetModel
from diffusionsat.pipeline_controlnet import StableDiffusionControlNetPipeline
from diffusionsat.data_util import (
    # SampleEqually,
    fmow_tokenize_caption, fmow_numerical_metadata, fmow_temporal_images,
    spacenet_tokenize_caption, spacenet_numerical_metadata,
    satlas_tokenize_caption, satlas_numerical_metadata,
    texas_tokenize_caption, texas_numerical_metadata,
    xbd_tokenize_caption, xbd_numerical_metadata,
    metadata_normalize, combine_text_and_metadata,
    SentinelNormalize, IdentityTransform, SentinelDropBands, SentinelFlipBGR,
)

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.17.0.dev0")

logger = get_logger(__name__)


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def log_validation(vae, text_encoder, tokenizer, unet, controlnet, args, accelerator, weight_dtype, step):
    logger.info("Running validation... ")

    controlnet = accelerator.unwrap_model(controlnet)

    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        controlnet=controlnet,
        safety_checker=None,
        revision=args.revision,
        torch_dtype=weight_dtype,
    )
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    if len(args.validation_image) == len(args.validation_prompt):
        validation_images = args.validation_image
        validation_prompts = args.validation_prompt
    elif len(args.validation_image) == 1:
        validation_images = args.validation_image * len(args.validation_prompt)
        validation_prompts = args.validation_prompt
    elif len(args.validation_prompt) == 1:
        validation_images = args.validation_image
        validation_prompts = args.validation_prompt * len(args.validation_image)
    else:
        raise ValueError(
            "number of `args.validation_image` and `args.validation_prompt` should be checked in `parse_args`"
        )

    image_logs = []

    for validation_prompt, validation_image in zip(validation_prompts, validation_images):
        validation_image = Image.open(validation_image).convert("RGB")

        images = []

        for _ in range(args.num_validation_images):
            with torch.autocast("cuda"):
                image = pipeline(
                    validation_prompt, validation_image, num_inference_steps=20, generator=generator
                ).images[0]

            images.append(image)

        image_logs.append(
            {"validation_image": validation_image, "images": images, "validation_prompt": validation_prompt}
        )

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            for log in image_logs:
                images = log["images"]
                validation_prompt = log["validation_prompt"]
                validation_image = log["validation_image"]

                formatted_images = []

                formatted_images.append(np.asarray(validation_image))

                for image in images:
                    formatted_images.append(np.asarray(image))

                formatted_images = np.stack(formatted_images)

                tracker.writer.add_images(validation_prompt, formatted_images, step, dataformats="NHWC")
        elif tracker.name == "wandb":
            formatted_images = []

            for log in image_logs:
                images = log["images"]
                validation_prompt = log["validation_prompt"]
                validation_image = log["validation_image"]

                formatted_images.append(wandb.Image(validation_image, caption="Controlnet conditioning"))

                for image in images:
                    image = wandb.Image(image, caption=validation_prompt)
                    formatted_images.append(image)

            tracker.log({"validation": formatted_images})
        else:
            logger.warn(f"image logging not implemented for {tracker.name}")

        return image_logs


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a ControlNet training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--controlnet_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained controlnet model or model identifier from huggingface.co/models."
        " If not specified controlnet weights are initialized from unet.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained model identifier from huggingface.co/models. Trainable model components should be"
            " float32 precision."
        ),
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="controlnet-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. "
            "In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference."
            "Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."
            "See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by step"
            "instructions."
        ),
    )
    parser.add_argument(
        "--checkpoint_preempt_steps",
        type=int,
        default=None,
        help=(
            "Save a recent checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=(
            "Max number of checkpoints to store. Passed as `total_limit` to the `Accelerator` `ProjectConfiguration`."
            " See Accelerator::save_state https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.save_state"
            " for more details"
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing the target image."
    )
    parser.add_argument(
        "--conditioning_image_column",
        type=str,
        default="conditioning_image",
        help="The column of the dataset containing the controlnet conditioning image.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--proportion_empty_prompts",
        type=float,
        default=0,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of prompts evaluated every `--validation_steps` and logged to `--report_to`."
            " Provide either a matching number of `--validation_image`s, a single `--validation_image`"
            " to be used with all prompts, or a single prompt that will be used with all `--validation_image`s."
        ),
    )
    parser.add_argument(
        "--validation_image",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of paths to the controlnet conditioning image be evaluated every `--validation_steps`"
            " and logged to `--report_to`. Provide either a matching number of `--validation_prompt`s, a"
            " a single `--validation_prompt` to be used with all `--validation_image`s, or a single"
            " `--validation_image` that will be used with all `--validation_prompt`s."
        ),
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images to be generated for each `--validation_image`, `--validation_prompt` pair",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="train_controlnet",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )

    # CUSTOM: Added for DiffusionSat
    parser.add_argument("--wandb", type=str, default=None, help="Name of wandb project. None to not use wandb.")
    parser.add_argument("--dataset", type=str, default='texas', choices=['texas', 'fmow', 'fmow_temporal', 'xbd'])
    parser.add_argument(
        '--shardlist', type=str, default='./datasets/example_shardlist_disk.txt',
        help='Path to .txt file containing lists of shards (loaded in webdataset format).'
    )
    parser.add_argument("--texas_task", type=str, default='random', choices=['random', 'past', 'future'])
    parser.add_argument("--xbd_task", type=str, default='random', choices=['random', 'past', 'future'])
    parser.add_argument(
        "--unet_path", type=str, default=None,
    )
    parser.add_argument("--cond_resolution", type=int, default=None,
                        help="Whether to set different resolution for cond pixels vs output image.")
    parser.add_argument(
        "--num_metadata", type=int, default=5,
    )
    parser.add_argument(
        "--num_cond", type=int, default=3, help="Number of conditioning images to cat in channel dim."
    )
    parser.add_argument(
        "--drop_cond", type=float, default=0.0, help="Percentage of time to drop a conditioning image."
    )
    parser.add_argument(
        "--num_channels", type=int, default=3,
    )
    parser.add_argument(
        "--flip_bgr", action="store_true", default=False, help="Flip sentinel BGR to RGB"
    )
    parser.add_argument(
        "--dropped_bands", type=int, default=[], nargs="+", help="Which sentinel bands to drop"
    )
    parser.add_argument("--temporal", action="store_true", default=False, help="Use 3D controlnet")
    parser.add_argument("--temporal_attn", action="store_true", default=False, help="Use temporal attention for temporal models.")
    parser.add_argument("--multi", action="store_true", default=False, help="Train multiple controlnets, one for each cond image")
    parser.add_argument(
        "--text_metadata", action='store_true', default=False,
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.proportion_empty_prompts < 0 or args.proportion_empty_prompts > 1:
        raise ValueError("`--proportion_empty_prompts` must be in the range [0, 1].")

    if args.validation_prompt is not None and args.validation_image is None:
        raise ValueError("`--validation_image` must be set if `--validation_prompt` is set")

    if args.validation_prompt is None and args.validation_image is not None:
        raise ValueError("`--validation_prompt` must be set if `--validation_image` is set")

    if (
        args.validation_image is not None
        and args.validation_prompt is not None
        and len(args.validation_image) != 1
        and len(args.validation_prompt) != 1
        and len(args.validation_image) != len(args.validation_prompt)
    ):
        raise ValueError(
            "Must provide either 1 `--validation_image`, 1 `--validation_prompt`,"
            " or the same number of `--validation_prompt`s and `--validation_image`s"
        )

    if args.resolution % 8 != 0:
        raise ValueError(
            "`--resolution` must be divisible by 8 for consistently sized encoded images between the VAE and the controlnet encoder."
        )

    if args.unet_path is None:
        args.unet_path = args.pretrained_model_name_or_path
    print(f"Using unet from {args.unet_path}")
    if args.cond_resolution is None:
        args.cond_resolution = args.resolution
    if args.dropped_bands:
        args.num_channels = args.num_channels - len(args.dropped_bands)
        assert args.num_channels > 0, "Cannot have -ve channels"
    return args


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(total_limit=args.checkpoints_total_limit, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.wandb is not None:
            wandb.init(project=args.wandb)
            wandb.config.update(args)

        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

        # Define a function to handle SIGTERM
        def sigterm_handler(signum, frame):
            """
            The signal will always provide two function arguments to the handler:
            signum: the signal number (in this case the value of signal.SIGTERM)
            frame: the current execution frame
            """
            print("Received SIGTERM signal. Gracefully exiting...")
            # Add code here to save files or perform other cleanup tasks
            if accelerator.is_main_process:
                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                accelerator.save_state(save_path)
                logger.info(f"Saved state to {save_path}")

        # Register the SIGTERM handler
        signal.signal(signal.SIGTERM, sigterm_handler)
        signal.signal(signal.SIGINT, sigterm_handler)

    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, revision=args.revision, use_fast=False)
    elif args.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
        )

    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    unet = SatUNet.from_pretrained(
        args.unet_path, subfolder="unet", low_cpu_mem_usage=True,
        num_metadata=args.num_metadata, use_metadata=args.num_metadata > 0,
    )
    assert (unet.metadata_embedding is None) == (args.num_metadata == 0)

    if args.controlnet_model_name_or_path:
        logger.info("Loading existing controlnet weights")
        if args.temporal:
            controlnet = ControlNetModel3D.from_pretrained(args.controlnet_model_name_or_path)
        else:
            controlnet = ControlNetModel.from_pretrained(args.controlnet_model_name_or_path)
    else:
        logger.info("Initializing controlnet weights from unet")
        if args.temporal:
            controlnet = ControlNetModel3D.from_unet(unet, conditioning_in_channels=args.num_channels, use_temporal_transformer=args.temporal_attn)
        elif args.multi:
            controlnets = []
            for _ in range(args.num_cond):
                controlnets.append(
                    ControlNetModel.from_unet(unet, conditioning_in_channels=args.num_channels,
                                              conditioning_scale=args.resolution // args.cond_resolution)
                )
            controlnet = MultiControlNetModel(controlnets)
        else:
            controlnet = ControlNetModel.from_unet(unet, conditioning_in_channels=args.num_cond * args.num_channels,
                                                   conditioning_scale=args.resolution // args.cond_resolution)
            assert controlnet.controlnet_cond_embedding.conv_in.in_channels == args.num_cond * args.num_channels


    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            i = len(weights) - 1

            while len(weights) > 0:
                weights.pop()
                model = models[i]

                sub_dir = "controlnet"
                model.save_pretrained(os.path.join(output_dir, sub_dir))

                i -= 1

        def load_model_hook(models, input_dir):
            while len(models) > 0:
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                mod_class = ControlNetModel
                if args.temporal:
                    mod_class = ControlNetModel3D
                elif args.multi:
                    mod_class = MultiControlNetModel
                if mod_class is MultiControlNetModel:
                    load_model = mod_class.from_pretrained(input_dir)
                    for m_controlnet, l_controlnet in zip(model.nets, load_model.nets):
                        m_controlnet.register_to_config(**l_controlnet.config)
                else:
                    load_model = mod_class.from_pretrained(input_dir, subfolder="controlnet")
                    model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    # Only train the ControlNet
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    controlnet.train()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
            controlnet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.gradient_checkpointing:
        controlnet.enable_gradient_checkpointing()

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        " Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training, copy of the weights should still be float32."
    )

    if accelerator.unwrap_model(controlnet).dtype != torch.float32:
        raise ValueError(
            f"Controlnet loaded as datatype {accelerator.unwrap_model(controlnet).dtype}. {low_precision_error_string}"
        )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Optimizer creation
    params_to_optimize = controlnet.parameters()
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    normalizer = transforms.Normalize([0.5], [0.5])
    train_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
            transforms.CenterCrop(args.resolution),
            # transforms.RandomHorizontalFlip(),
            # transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    cond_image_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
            transforms.CenterCrop(args.resolution),
            # transforms.ToTensor(),
            # transforms.Normalize([0.5], [0.5])
        ]
    )

    dropped_bands = copy.deepcopy(args.dropped_bands)
    print(f"Dropping bands {dropped_bands}")
    low_res_transforms = transforms.Compose(
        [
            SentinelNormalize(channel_specific=True) if args.num_channels > 3 else IdentityTransform(),
            transforms.ToTensor(),
            SentinelFlipBGR() if args.flip_bgr else IdentityTransform(),
            SentinelDropBands(dropped_bands) if len(dropped_bands) > 0 else IdentityTransform(),
            transforms.Resize(args.cond_resolution, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
            transforms.CenterCrop(args.cond_resolution),  # cond img can be 64x64, 128x128, 256x256, or 512x512
            # transforms.ToTensor(),
        ]
    )


    def texas_preprocess_train(examples):
        for example in examples:
            if args.texas_task == 'past':
                target = 'tif.rgb-2016.npy'
            elif args.texas_task == 'future':
                target = 'tif.rgb-2018.npy'
            else:
                target = random.choice([k for k in example if 'rgb' in k and '.npy' in k])
            others = sorted([k for k in example if k != target and '.npy' in k])[:args.num_cond]
            cond_image = cond_image_transform(example[others[0]])
            cond_metadata = texas_numerical_metadata(example, others[0], args.resolution, args.num_metadata)
            cond_metadata = metadata_normalize(cond_metadata).unsqueeze(-1)
            for key in others[1:]:
                cond_image = torch.cat((cond_image, cond_image_transform(example[key])), dim=0)  # cat in channel dim
                cond_md = texas_numerical_metadata(example, key, args.resolution, args.num_metadata)
                cond_md = metadata_normalize(cond_md).unsqueeze(-1)
                cond_metadata = torch.cat((cond_metadata, cond_md), dim=-1)

            metadata = texas_numerical_metadata(example, target, args.resolution, args.num_metadata)
            output = {
                'pixel_values': train_transforms(example[target]),
                'input_ids': texas_tokenize_caption(example, tokenizer),
                'conditioning_pixel_values': cond_image,
                'conditioning_metadata': cond_metadata,
                'metadata': metadata_normalize(metadata),
            }
            yield output


    fmow_meta_df = pd.read_csv('datasets/fmow-train-meta.csv')
    def fmow_preprocess_train(examples):
        for example in examples:
            if args.text_metadata:
                metadata = fmow_numerical_metadata(example, fmow_meta_df, args.resolution, args.num_metadata,
                                                   base_year=0, rgb_key='rgb.npy')
                input_ids = fmow_tokenize_caption(example, tokenizer, return_text=True)
                input_ids = combine_text_and_metadata(input_ids, metadata, tokenizer)
            else:
                metadata = fmow_numerical_metadata(example, fmow_meta_df, args.resolution, args.num_metadata,
                                                   rgb_key='rgb.npy')
                input_ids = fmow_tokenize_caption(example, tokenizer)

            cond_metadata = metadata.clone().detach()  # (num_md)
            cond_metadata[2] = 10.
            output = {
                'pixel_values': train_transforms(example['rgb.npy']),
                'conditioning_pixel_values': low_res_transforms(example['multispectral.npy']),
                'input_ids': input_ids,
                'metadata': metadata_normalize(metadata),
                'conditioning_metadata': metadata_normalize(cond_metadata).unsqueeze(-1),
            }
            yield output

    def fmow_temporal_preprocess_train(examples):
        for example in examples:
            img_temporal, md_keys = fmow_temporal_images(example, cond_image_transform, num_frames=args.num_cond + 1)

            target_img = normalizer(img_temporal[0])  # (C, H, W)
            target_rgb_key = md_keys[0].replace('metadata', 'input').replace('json', 'npy')
            target_md = fmow_numerical_metadata(example, fmow_meta_df, args.resolution, args.num_metadata, rgb_key=target_rgb_key, md_key=md_keys[0])
            target_md = metadata_normalize(target_md)

            md_tensor = []
            for md_key in md_keys[1:]:
                rgb_key = md_key.replace('metadata', 'input').replace('json', 'npy')
                md = fmow_numerical_metadata(example, fmow_meta_df, args.resolution, args.num_metadata, rgb_key=rgb_key, md_key=md_key)
                md_tensor.append(metadata_normalize(md))

            cond_metadata = torch.stack(md_tensor, dim=-1)  # (num_md, T)
            cond_imgs = img_temporal[1:].view(-1, *img_temporal.shape[-2:])  # (T*C, H, W)
            output = {
                'pixel_values': target_img,  # (C, H, W)
                'conditioning_pixel_values': cond_imgs,
                'input_ids': fmow_tokenize_caption(example, tokenizer, md_key='metadata-0.json'),
                'metadata': target_md,
                'conditioning_metadata': cond_metadata,
            }
            yield output

    def xbd_preprocess_train(examples):
        for example in examples:
            if args.xbd_task == 'past':
                rgb_key = 'pre-input.npy'
            elif args.xbd_task == 'future':
                rgb_key = 'post-input.npy'
            else:
                rgb_key = random.choice(['pre-input.npy', 'post-input.npy'])
            md_key = 'pre-metadata.json' if 'pre' in rgb_key else 'post-metadata.json'
            cond_key = 'pre-input.npy' if 'post' in rgb_key else 'post-input.npy'
            cond_md_key = 'pre-metadata.json' if 'post' in rgb_key else 'post-metadata.json'

            if args.text_metadata:
                metadata = xbd_numerical_metadata(example, args.resolution, args.num_metadata, rgb_key=rgb_key, md_key=md_key, base_year=0,)
                input_ids = xbd_tokenize_caption(example, tokenizer, md_key=md_key, return_text=True)
                input_ids = combine_text_and_metadata(input_ids, metadata, tokenizer)
            else:
                metadata = xbd_numerical_metadata(example, args.resolution, args.num_metadata, rgb_key=rgb_key, md_key=md_key)
                input_ids = xbd_tokenize_caption(example, tokenizer, md_key=md_key)

            metadata = metadata_normalize(metadata)

            target_img = train_transforms(example[rgb_key])  # (C, H, W)
            cond_img = cond_image_transform(example[cond_key])  # (C, H, W)
            cond_metadata = xbd_numerical_metadata(example, args.resolution, args.num_metadata, rgb_key=cond_key, md_key=cond_md_key)
            cond_metadata = metadata_normalize(cond_metadata)

            output = {
                'pixel_values': target_img,  # (C, H, W)
                'conditioning_pixel_values': cond_img,
                'input_ids': input_ids,
                'metadata': metadata,
                'conditioning_metadata': cond_metadata.unsqueeze(-1),
            }
            yield output

    preprocess_fn = None
    if args.dataset == 'texas':
        preprocess_fn = texas_preprocess_train
    elif args.dataset == 'fmow':
        preprocess_fn = fmow_preprocess_train
    elif args.dataset == 'fmow_temporal':
        preprocess_fn = fmow_temporal_preprocess_train
    elif args.dataset == "xbd":
        preprocess_fn = xbd_preprocess_train
    else:
        raise NotImplementedError("bad dataset spec.")

    # Load dataset shards in webdataset format
    with open(args.shardlist, 'r') as shard_f:
        shardlist = shard_f.read().splitlines()

    full_shardlist = []
    for data_shard in shardlist:
        full_shardlist += list(braceexpand.braceexpand(data_shard))

    train_dataset = wds.DataPipeline(
        wds.ResampledShards('::'.join(full_shardlist)),
        # wds.shuffle(100),
        wds.tarfile_to_samples(),
        wds.shuffle(10000, initial=1000),
        wds.decode(),
        preprocess_fn,
    )

    def collate_fn(examples):
        dict_of_batches = {}

        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        dict_of_batches['pixel_values'] = pixel_values

        conditioning_pixel_values = torch.stack([example["conditioning_pixel_values"] for example in examples])
        conditioning_pixel_values = conditioning_pixel_values.to(memory_format=torch.contiguous_format).float()
        dict_of_batches['conditioning_pixel_values'] = conditioning_pixel_values

        input_ids = torch.stack([example["input_ids"] for example in examples])
        dict_of_batches['input_ids'] = input_ids

        metadata = torch.stack([example["metadata"] for example in examples])
        dict_of_batches['metadata'] = metadata

        cond_metadata = torch.stack([example["conditioning_metadata"] for example in examples])
        dict_of_batches['conditioning_metadata'] = cond_metadata

        return dict_of_batches

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        # shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Just a placeholder, don't bother (since we use webdataset)
    num_samples = 363570
    num_batches = num_samples // args.train_batch_size

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(num_batches / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    controlnet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        controlnet, optimizer, train_dataloader, lr_scheduler
    )

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae, unet and text_encoder to device and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(num_batches / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))

        # tensorboard cannot handle list types for config
        tracker_config.pop("validation_prompt")
        tracker_config.pop("validation_image")
        tracker_config.pop("dropped_bands")

        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = leave me alone")
    logger.info(f"  Num batches each epoch = no")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    image_logs = None
    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(controlnet):
                # Convert images to latent space
                in_c = batch["pixel_values"].shape[1]
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                controlnet_image = batch["conditioning_pixel_values"].to(dtype=weight_dtype)
                if args.temporal:
                    controlnet_image = einops.rearrange(controlnet_image, 'b (t c) h w -> b c t h w', t=args.num_cond)
                elif args.multi:
                    controlnet_image = einops.rearrange(controlnet_image, 'b (t c) h w -> t b c h w', t=args.num_cond)
                    controlnet_image = [im for im in controlnet_image]

                # Get the numerical metadata
                metadata = None
                cond_metadata = None
                if args.num_metadata > 0:
                    metadata = batch["metadata"]  # (N, num_md)

                    # NOTE: this version of dropping metadata might not be the correct way to do it.
                    # Another way would be to 0 out the metadata AFTER the timestep projection,
                    # but all models I trained didn't use this method, and used the version below.
                    # To try out the "correct" method, replace the 0. below with -1. and then uncomment lines 551, 553 in
                    # diffusionsat/sat_unet.py. I haven't tried this, but you can see if it works better.
                    # For controlnet, you'd have to implement a similar change to controlnet.py and controlnet_3d.py.
                    keep_mask = torch.rand_like(metadata) > 0.1
                    metadata = metadata * keep_mask + 0. * ~keep_mask * torch.ones_like(metadata)  # set drop metadata to -1

                    cond_metadata = batch["conditioning_metadata"]
                    keep_mask = torch.rand_like(cond_metadata) > 0.1
                    cond_metadata = cond_metadata * keep_mask + 0. * ~keep_mask * torch.ones_like(cond_metadata)

                # NOTE: This (experimental) portion drops out items in the conditioning input.
                # I didn't end up using it. Not sure if it's up to date.
                if args.num_cond > 1:
                    nc = args.num_channels
                    if random.random() < args.drop_cond:
                        num_drop = random.choice(list(range(args.num_cond - 1)))
                        drop_idxs = random.sample(list(range(args.num_cond)), k=num_drop)
                        for idx in drop_idxs:
                            controlnet_image[:, nc*idx:nc*(idx+1), :, :] = torch.zeros_like(controlnet_image[:, nc*idx:nc*(idx+1), :, :])
                            if args.num_metadata > 0:
                                cond_metadata[:, :, idx] = torch.zeros_like(cond_metadata[:, :, idx])

                if args.multi:
                    cond_metadata = einops.rearrange(cond_metadata, 'b m t -> t b m')
                    cond_metadata = [cond_md for cond_md in cond_metadata]

                down_block_res_samples, mid_block_res_sample = controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=controlnet_image,
                    metadata=metadata,
                    cond_metadata=cond_metadata,
                    return_dict=False,
                )

                # Predict the noise residual
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    metadata=metadata,
                    down_block_additional_residuals=[
                        sample.to(dtype=weight_dtype) for sample in down_block_res_samples
                    ],
                    mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype),
                ).sample

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = controlnet.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                #################################################################
                # CUSTOM: Log Wandb
                if args.wandb is not None and accelerator.is_main_process:
                    wandb.log(
                        {"Train/loss": loss.detach().item(), "Train/lr": lr_scheduler.get_last_lr()[0], "Train/step": global_step}
                    )
                #################################################################

                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                    elif args.checkpoint_preempt_steps is not None and global_step % args.checkpoint_preempt_steps == 0:
                        if accelerator.is_main_process:
                            save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                            accelerator.save_state(save_path)
                            logger.info(f"Saved state to {save_path}")

                            # Get the most recent checkpoint
                            prev_ckpt_dirs = os.listdir(args.output_dir)
                            prev_ckpt_dirs = [d for d in prev_ckpt_dirs if d.startswith("checkpoint")]
                            prev_ckpt_dirs = sorted(prev_ckpt_dirs, key=lambda x: int(x.split("-")[1]))
                            for prev_ckpt_path in prev_ckpt_dirs[:-1]:
                                prev_ckpt_step = int(prev_ckpt_path.split("-")[1])
                                # only delete if not a valid ckpt
                                if prev_ckpt_step < global_step and prev_ckpt_step % args.checkpointing_steps != 0:
                                    shutil.rmtree(os.path.join(args.output_dir, prev_ckpt_path))

                    if args.validation_prompt is not None and global_step % args.validation_steps == 0:
                        image_logs = log_validation(
                            vae,
                            text_encoder,
                            tokenizer,
                            unet,
                            controlnet,
                            args,
                            accelerator,
                            weight_dtype,
                            global_step,
                        )

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        controlnet = accelerator.unwrap_model(controlnet)
        controlnet.save_pretrained(args.output_dir)

        pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            text_encoder=text_encoder,
            vae=vae,
            unet=unet,
            controlnet=controlnet,
        )
        pipeline.save_pretrained(args.output_dir)

        if args.push_to_hub:
            # save_model_card(
            #     repo_id,
            #     image_logs=image_logs,
            #     base_model=args.pretrained_model_name_or_path,
            #     repo_folder=args.output_dir,
            # )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
