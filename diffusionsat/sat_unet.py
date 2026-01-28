"""Satellite UNet wrapper with metadata support on top of diffusers."""
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from diffusers.models.unets.unet_2d_condition import (
    UNet2DConditionModel,
    UNet2DConditionOutput,
)
from diffusers.utils import logging

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class SatUNet(UNet2DConditionModel):
    """Thin wrapper around `diffusers.UNet2DConditionModel` with metadata embeddings."""

    _supports_gradient_checkpointing = True

    def __init__(self, *args, use_metadata: bool = True, num_metadata: int = 7, **kwargs):
        super().__init__(*args, **kwargs)

        # Track custom config entries for save/load parity with the base model.
        self.register_to_config(use_metadata=use_metadata, num_metadata=num_metadata)

        self.use_metadata = use_metadata
        self.num_metadata = num_metadata

        if use_metadata:
            # Re-use the same dimensions as the base time embedding.
            timestep_input_dim = self.time_embedding.linear_1.in_features
            time_embed_dim = self.time_embedding.linear_2.out_features
            self.metadata_embedding = nn.ModuleList(
                [self._build_metadata_embedding(timestep_input_dim, time_embed_dim) for _ in range(num_metadata)]
            )
        else:
            self.metadata_embedding = None

    @staticmethod
    def _build_metadata_embedding(timestep_input_dim: int, time_embed_dim: int) -> nn.Module:
        from diffusers.models.embeddings import TimestepEmbedding

        return TimestepEmbedding(timestep_input_dim, time_embed_dim)

    def _encode_metadata(
        self, metadata: Optional[torch.Tensor], dtype: torch.dtype
    ) -> Optional[torch.Tensor]:
        if self.metadata_embedding is None:
            return None

        if metadata is None:
            raise ValueError("metadata must be provided when use_metadata=True")

        if metadata.dim() != 2 or metadata.shape[1] != self.num_metadata:
            raise ValueError(f"Invalid metadata shape {metadata.shape}, expected (batch, {self.num_metadata})")

        md_bsz = metadata.shape[0]
        # Reuse the same projection used for timestep encoding to stay aligned with base embeddings.
        projected = self.time_proj(metadata.view(-1)).view(md_bsz, self.num_metadata, -1).to(dtype=dtype)

        md_emb = projected.new_zeros((md_bsz, projected.shape[-1]))
        for idx, md_embed in enumerate(self.metadata_embedding):
            md_emb = md_emb + md_embed(projected[:, idx, :])

        return md_emb

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        down_intrablock_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        metadata: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[UNet2DConditionOutput, Tuple]:
        # Largely mirrors `UNet2DConditionModel.forward` with a metadata injection on the timestep embedding.

        default_overall_up_factor = 2**self.num_upsamplers
        forward_upsample_size = False
        upsample_size = None

        for dim in sample.shape[-2:]:
            if dim % default_overall_up_factor != 0:
                forward_upsample_size = True
                break

        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        if encoder_attention_mask is not None:
            encoder_attention_mask = (1 - encoder_attention_mask.to(sample.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        t_emb = self.get_time_embed(sample=sample, timestep=timestep)
        emb = self.time_embedding(t_emb, timestep_cond)

        class_emb = self.get_class_embed(sample=sample, class_labels=class_labels)
        if class_emb is not None:
            if self.config.class_embeddings_concat:
                emb = torch.cat([emb, class_emb], dim=-1)
            else:
                emb = emb + class_emb

        aug_emb = self.get_aug_embed(
            emb=emb, encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs or {}
        )
        if self.config.addition_embed_type == "image_hint" and aug_emb is not None:
            aug_emb, hint = aug_emb
            sample = torch.cat([sample, hint], dim=1)

        emb = emb + aug_emb if aug_emb is not None else emb

        md_emb = self._encode_metadata(metadata=metadata, dtype=sample.dtype)
        if md_emb is not None:
            emb = emb + md_emb

        if self.time_embed_act is not None:
            emb = self.time_embed_act(emb)

        encoder_hidden_states = self.process_encoder_hidden_states(
            encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs or {}
        )

        sample = self.conv_in(sample)

        if cross_attention_kwargs is not None and cross_attention_kwargs.get("gligen", None) is not None:
            cross_attention_kwargs = cross_attention_kwargs.copy()
            gligen_args = cross_attention_kwargs.pop("gligen")
            cross_attention_kwargs["gligen"] = {"objs": self.position_net(**gligen_args)}

        if cross_attention_kwargs is not None:
            cross_attention_kwargs = cross_attention_kwargs.copy()
            lora_scale = cross_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        from diffusers.utils import USE_PEFT_BACKEND, scale_lora_layers, unscale_lora_layers, deprecate

        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)

        is_controlnet = mid_block_additional_residual is not None and down_block_additional_residuals is not None
        is_adapter = down_intrablock_additional_residuals is not None
        if not is_adapter and mid_block_additional_residual is None and down_block_additional_residuals is not None:
            deprecate(
                "T2I should not use down_block_additional_residuals",
                "1.3.0",
                "Passing intrablock residual connections with `down_block_additional_residuals` is deprecated "
                "and will be removed in diffusers 1.3.0.  `down_block_additional_residuals` should only be used "
                "for ControlNet. Please use `down_intrablock_additional_residuals` instead.",
                standard_warn=False,
            )
            down_intrablock_additional_residuals = down_block_additional_residuals
            is_adapter = True

        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                additional_residuals: Dict[str, torch.Tensor] = {}
                if is_adapter and len(down_intrablock_additional_residuals) > 0:
                    additional_residuals["additional_residuals"] = down_intrablock_additional_residuals.pop(0)

                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_attention_mask=encoder_attention_mask,
                    **additional_residuals,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
                if is_adapter and len(down_intrablock_additional_residuals) > 0:
                    sample += down_intrablock_additional_residuals.pop(0)

            down_block_res_samples += res_samples

        if is_controlnet:
            new_down_block_res_samples = ()

            for down_block_res_sample, down_block_additional_residual in zip(
                down_block_res_samples, down_block_additional_residuals
            ):
                down_block_res_sample = down_block_res_sample + down_block_additional_residual
                new_down_block_res_samples = new_down_block_res_samples + (down_block_res_sample,)

            down_block_res_samples = new_down_block_res_samples

        if self.mid_block is not None:
            if hasattr(self.mid_block, "has_cross_attention") and self.mid_block.has_cross_attention:
                sample = self.mid_block(
                    sample,
                    emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_attention_mask=encoder_attention_mask,
                )
            else:
                sample = self.mid_block(sample, emb)

            if (
                is_adapter
                and len(down_intrablock_additional_residuals) > 0
                and sample.shape == down_intrablock_additional_residuals[0].shape
            ):
                sample += down_intrablock_additional_residuals.pop(0)

        if is_controlnet:
            sample = sample + mid_block_additional_residual

        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    upsample_size=upsample_size,
                )

        if self.conv_norm_out:
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (sample,)

        return UNet2DConditionOutput(sample=sample)
