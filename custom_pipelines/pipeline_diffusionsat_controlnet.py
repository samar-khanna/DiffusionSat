"""
Self-contained DiffusionSat ControlNet pipeline that can be loaded directly from
the checkpoint folder without importing the project package.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import einops
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from torch import nn
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

from diffusers.loaders import TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    PIL_INTERPOLATION,
    logging,
    randn_tensor,
    replace_example_docstring,
    is_accelerate_available,
    is_accelerate_version,
)
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion.pipeline_output import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    StableDiffusionPipeline as DiffusersStableDiffusionPipeline,
)
from diffusers.pipelines.controlnet.pipeline_controlnet import (
    StableDiffusionControlNetPipeline as DiffusersControlNetPipeline,
)

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> # !pip install opencv-python transformers accelerate
        >>> from diffusers import DiffusionPipeline
        >>> from diffusers.utils import load_image
        >>> import numpy as np
        >>> import torch
        >>> import cv2
        >>> from PIL import Image
        >>>
        >>> image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png")
        >>> image = np.array(image)
        >>> image = cv2.Canny(image, 100, 200)
        >>> image = image[:, :, None]
        >>> image = np.concatenate([image, image, image], axis=2)
        >>> canny_image = Image.fromarray(image)
        >>>
        >>> pipe = DiffusionPipeline.from_pretrained("path/to/ckpt/diffusionsat", torch_dtype=torch.float16)
        >>> pipe = pipe.to("cuda")
        >>> pipe.enable_xformers_memory_efficient_attention()
        >>> generator = torch.manual_seed(0)
        >>> image = pipe(
        ...     "futuristic-looking woman", num_inference_steps=20, generator=generator, image=canny_image
        ... ).images[0]
        ```
"""


class DiffusionSatControlNetPipeline(DiffusionPipeline, TextualInversionLoaderMixin):
    """
    ControlNet-aware pipeline for DiffusionSat. This is a mostly direct copy of
    the project pipeline to avoid importing the `diffusionsat` package when
    loading from the checkpoint folder. Minimal tweaks:
    - auto-fills metadata/cond_metadata with zeros when the model expects them.
    """

    _optional_components = ["safety_checker", "feature_extractor"]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: Any,
        controlnet: Any,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        requires_safety_checker: bool = True,
    ):
        super().__init__()

        if safety_checker is None and requires_safety_checker:
            logger.warning(
                f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                " it only for use-cases that involve analyzing network behavior or auditing its results."
            )

        if safety_checker is not None and feature_extractor is None:
            raise ValueError(
                "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
                " checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
            )

        # Support MultiControlNetModel-like objects without importing the project module.
        if isinstance(controlnet, (list, tuple)):
            # defer to diffusers' MultiControlNetModel if available
            from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel

            controlnet = MultiControlNetModel(controlnet)

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            controlnet=controlnet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.register_to_config(requires_safety_checker=requires_safety_checker)

    # Reuse helpers from diffusers baseline pipelines.
    enable_vae_slicing = DiffusersStableDiffusionPipeline.enable_vae_slicing
    disable_vae_slicing = DiffusersStableDiffusionPipeline.disable_vae_slicing
    enable_vae_tiling = DiffusersStableDiffusionPipeline.enable_vae_tiling
    disable_vae_tiling = DiffusersStableDiffusionPipeline.disable_vae_tiling
    enable_sequential_cpu_offload = DiffusersControlNetPipeline.enable_sequential_cpu_offload
    enable_model_cpu_offload = DiffusersControlNetPipeline.enable_model_cpu_offload
    _execution_device = DiffusersStableDiffusionPipeline._execution_device
    _encode_prompt = DiffusersStableDiffusionPipeline._encode_prompt
    run_safety_checker = DiffusersStableDiffusionPipeline.run_safety_checker
    decode_latents = DiffusersStableDiffusionPipeline.decode_latents
    prepare_extra_step_kwargs = DiffusersStableDiffusionPipeline.prepare_extra_step_kwargs
    check_inputs = DiffusersControlNetPipeline.check_inputs
    check_image = DiffusersControlNetPipeline.check_image
    prepare_image = DiffusersControlNetPipeline.prepare_image
    prepare_latents = DiffusersStableDiffusionPipeline.prepare_latents

    def prepare_metadata(self, batch_size, metadata, ndims, do_classifier_free_guidance, device, dtype):
        has_metadata = getattr(self.unet.config, "use_metadata", False)
        num_metadata = getattr(self.unet.config, "num_metadata", 0)

        if metadata is None and has_metadata and num_metadata > 0:
            shape = (batch_size, num_metadata) if ndims == 2 else (batch_size, num_metadata, 1)
            metadata = torch.zeros(shape, device=device, dtype=dtype)

        if metadata is None:
            return None

        md = torch.as_tensor(metadata)
        if ndims == 2:
            assert (len(md.shape) == 1 and batch_size == 1) or (len(md.shape) == 2 and batch_size > 1)
            if len(md.shape) == 1:
                md = md.unsqueeze(0).expand(batch_size, -1)
        elif ndims == 3:
            assert (len(md.shape) == 2 and batch_size == 1) or (len(md.shape) == 3 and batch_size > 1)
            if len(md.shape) == 2:
                md = md.unsqueeze(0).expand(batch_size, -1, -1)

        if do_classifier_free_guidance:
            md = torch.cat([torch.zeros_like(md), md])

        md = md.to(device=device, dtype=dtype)
        return md

    def _default_height_width(self, height, width, image):
        while isinstance(image, list):
            image = image[0]

        if height is None:
            if isinstance(image, PIL.Image.Image):
                height = image.height
            elif isinstance(image, torch.Tensor):
                height = image.shape[2]
            height = (height // 8) * 8

        if width is None:
            if isinstance(image, PIL.Image.Image):
                width = image.width
            elif isinstance(image, torch.Tensor):
                width = image.shape[3]
            width = (width // 8) * 8

        return height, width

    # override DiffusionPipeline
    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        safe_serialization: bool = False,
        variant: Optional[str] = None,
    ):
        # For single or multi controlnet, rely on default save logic.
        super().save_pretrained(save_directory, safe_serialization=safe_serialization, variant=variant)

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        image: Union[torch.FloatTensor, PIL.Image.Image, List[torch.FloatTensor], List[PIL.Image.Image]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        guess_mode: bool = False,
        metadata: Optional[List[float]] = None,
        cond_metadata: Optional[List[float]] = None,
        is_temporal: bool = False,
        conditioning_downsample: bool = True,
    ):
        # 0. Default height and width to unet
        height, width = self._default_height_width(height, width, image)
        cond_height, cond_width = height, width
        if not conditioning_downsample:
            cond_height, cond_width = height // 8, width // 8

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            image,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            controlnet_conditioning_scale,
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0

        from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel

        if isinstance(self.controlnet, MultiControlNetModel) and isinstance(controlnet_conditioning_scale, float):
            controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(self.controlnet.nets)

        # 3. Encode input prompt
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        # 4. Prepare image
        is_compiled = hasattr(F, "scaled_dot_product_attention") and isinstance(
            self.controlnet, torch._dynamo.eval_frame.OptimizedModule
        )
        is_multi_cond = isinstance(image, list)

        if (
            hasattr(self.controlnet, "controlnet_cond_embedding")
            or is_compiled
            and hasattr(self.controlnet._orig_mod, "controlnet_cond_embedding")
        ):
            image = self.prepare_image(
                image=image,
                width=cond_width,
                height=cond_height,
                batch_size=batch_size * num_images_per_prompt,
                num_images_per_prompt=num_images_per_prompt,
                device=device,
                dtype=self.controlnet.dtype,
                do_classifier_free_guidance=do_classifier_free_guidance,
                guess_mode=guess_mode,
            )

        # 5. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 6. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 7. Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # CUSTOM metadata handling (auto-zero filled)
        input_metadata = self.prepare_metadata(batch_size, metadata, 2, do_classifier_free_guidance, device, prompt_embeds.dtype)
        ndims_cond = 3 if is_multi_cond else 2
        cond_metadata = self.prepare_metadata(
            batch_size, cond_metadata, ndims_cond, do_classifier_free_guidance, device, prompt_embeds.dtype
        )
        if input_metadata is not None:
            assert len(input_metadata.shape) == 2 and input_metadata.shape[-1] == getattr(self.unet.config, "num_metadata", input_metadata.shape[-1])
        if cond_metadata is not None:
            assert len(cond_metadata.shape) == ndims_cond and cond_metadata.shape[1] == getattr(self.unet.config, "num_metadata", cond_metadata.shape[1])
            if is_multi_cond and not is_temporal and not isinstance(self.controlnet, MultiControlNetModel):
                assert cond_metadata.shape[2] == self.controlnet.controlnet_cond_embedding.conv_in.in_channels / 3

        if input_metadata is not None:
            assert input_metadata.shape[0] == prompt_embeds.shape[0]

        if is_temporal:
            num_cond = cond_metadata.shape[-1] if cond_metadata is not None else image.shape[1] // self.controlnet.config.conditioning_in_channels
            image = einops.rearrange(image, 'b (t c) h w -> b c t h w', t=num_cond)
        elif isinstance(self.controlnet, MultiControlNetModel) and cond_metadata is not None:
            num_cond = cond_metadata.shape[-1] if cond_metadata is not None else image.shape[1] // self.controlnet.config.conditioning_in_channels
            image = einops.rearrange(image, 'b (t c) h w -> t b c h w', t=num_cond)
            image = [im for im in image]
            cond_metadata = einops.rearrange(cond_metadata, 'b m t -> t b m')
            cond_metadata = [cond_md for cond_md in cond_metadata]

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                if guess_mode and do_classifier_free_guidance:
                    controlnet_latent_model_input = latents
                    controlnet_prompt_embeds = prompt_embeds.chunk(2)[1]
                else:
                    controlnet_latent_model_input = latent_model_input
                    controlnet_prompt_embeds = prompt_embeds

                down_block_res_samples, mid_block_res_sample = self.controlnet(
                    controlnet_latent_model_input,
                    t,
                    encoder_hidden_states=controlnet_prompt_embeds,
                    controlnet_cond=image,
                    metadata=input_metadata,
                    cond_metadata=cond_metadata,
                    conditioning_scale=controlnet_conditioning_scale,
                    guess_mode=guess_mode,
                    return_dict=False,
                )

                if guess_mode and do_classifier_free_guidance:
                    down_block_res_samples = [torch.cat([torch.zeros_like(d), d]) for d in down_block_res_samples]
                    mid_block_res_sample = torch.cat([torch.zeros_like(mid_block_res_sample), mid_block_res_sample])

                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    metadata=input_metadata,
                    cross_attention_kwargs=cross_attention_kwargs,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                    return_dict=False,
                )[0]

                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.unet.to("cpu")
            self.controlnet.to("cpu")
            torch.cuda.empty_cache()

        if output_type == "latent":
            image = latents
            has_nsfw_concept = None
        elif output_type == "pil":
            image = self.decode_latents(latents)
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
            image = self.numpy_to_pil(image)
        else:
            image = self.decode_latents(latents)
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)

        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)


__all__ = ["DiffusionSatControlNetPipeline"]
