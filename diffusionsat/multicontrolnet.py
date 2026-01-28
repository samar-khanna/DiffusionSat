###########################################################################
# References:
# https://github.com/huggingface/diffusers/
###########################################################################
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from diffusers.models.controlnets.multicontrolnet import (
    MultiControlNetModel as HfMultiControlNetModel,
)
from diffusers.models.unets.unet_2d_blocks import CrossAttnDownBlock2D, DownBlock2D
from diffusers.utils import logging

from .controlnet import ControlNetOutput


logger = logging.get_logger(__name__)


class MultiControlNetModel(HfMultiControlNetModel):
    _supports_gradient_checkpointing = True

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (CrossAttnDownBlock2D, DownBlock2D)):
            module.gradient_checkpointing = value

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        controlnet_cond: List[torch.tensor],
        conditioning_scale: Optional[List[float]] = None,
        metadata: Optional[torch.Tensor] = None,
        cond_metadata: Optional[List[torch.Tensor]] = None,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guess_mode: bool = False,
        return_dict: bool = True,
    ) -> Union[ControlNetOutput, Tuple]:
        if conditioning_scale is None:
            conditioning_scale = [1.0] * len(controlnet_cond)
        if cond_metadata is None:
            cond_metadata = [None] * len(controlnet_cond)

        for i, (image, cond_md, scale, controlnet) in enumerate(
            zip(controlnet_cond, cond_metadata, conditioning_scale, self.nets)
        ):
            down_samples, mid_sample = controlnet(
                sample=sample,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=image,
                conditioning_scale=scale,
                metadata=metadata,
                cond_metadata=cond_md,
                class_labels=class_labels,
                timestep_cond=timestep_cond,
                attention_mask=attention_mask,
                added_cond_kwargs=added_cond_kwargs,
                cross_attention_kwargs=cross_attention_kwargs,
                guess_mode=guess_mode,
                return_dict=return_dict,
            )

            if i == 0:
                down_block_res_samples, mid_block_res_sample = down_samples, mid_sample
            else:
                down_block_res_samples = [
                    samples_prev + samples_curr
                    for samples_prev, samples_curr in zip(down_block_res_samples, down_samples)
                ]
                mid_block_res_sample += mid_sample

        return down_block_res_samples, mid_block_res_sample