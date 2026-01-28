"""ControlNet wrapper that reuses diffusers implementation and adds metadata."""
from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

from diffusers.models.controlnets.controlnet import (
    ControlNetConditioningEmbedding as HFConditioningEmbedding,
    ControlNetModel as HFControlNetModel,
    ControlNetOutput,
    zero_module,
)
from diffusers.utils import logging

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class ControlNetConditioningEmbedding(HFConditioningEmbedding):
    """Adapter to allow variable downsample stride via `scale` while reusing upstream layers."""

    def __init__(
        self,
        conditioning_embedding_channels: int,
        conditioning_channels: int = 3,
        block_out_channels: Tuple[int, ...] = (16, 32, 96, 256),
        scale: int = 1,
    ):
        # Initialize base, then optionally override blocks to respect custom stride.
        super().__init__(
            conditioning_embedding_channels=conditioning_embedding_channels,
            conditioning_channels=conditioning_channels,
            block_out_channels=block_out_channels,
        )
        if scale != 1:
            blocks = nn.ModuleList([])
            current_scale = scale
            for i in range(len(block_out_channels) - 1):
                channel_in = block_out_channels[i]
                channel_out = block_out_channels[i + 1]
                blocks.append(nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=1))
                stride = 2 if current_scale < 8 else 1
                blocks.append(nn.Conv2d(channel_in, channel_out, kernel_size=3, padding=1, stride=stride))
                if current_scale != 8:
                    current_scale = int(current_scale * 2)
            self.blocks = blocks


class ControlNetModel(HFControlNetModel):
    """Thin wrapper around `diffusers.ControlNetModel` with metadata embeddings."""

    def __init__(
        self,
        *args,
        conditioning_in_channels: int = 3,
        conditioning_scale: int = 1,
        use_metadata: bool = True,
        num_metadata: int = 7,
        **kwargs,
    ):
        # Map alias to upstream argument.
        kwargs.setdefault("conditioning_channels", conditioning_in_channels)

        super().__init__(*args, **kwargs)

        # Track custom config entries for save/load parity.
        self.register_to_config(
            use_metadata=use_metadata, num_metadata=num_metadata, conditioning_scale=conditioning_scale
        )

        self.use_metadata = use_metadata
        self.num_metadata = num_metadata

        if use_metadata:
            timestep_input_dim = self.time_embedding.linear_1.in_features
            time_embed_dim = self.time_embedding.linear_2.out_features
            self.metadata_embedding = nn.ModuleList(
                [
                    self._build_metadata_embedding(timestep_input_dim, time_embed_dim)
                    for _ in range(num_metadata)
                ]
            )
        else:
            self.metadata_embedding = None

        # Optionally replace conditioning embedding to honor `conditioning_scale` stride tweaks.
        if conditioning_scale != 1:
            self.controlnet_cond_embedding = ControlNetConditioningEmbedding(
                conditioning_embedding_channels=self.controlnet_cond_embedding.conv_out.out_channels,
                conditioning_channels=conditioning_in_channels,
                block_out_channels=tuple(
                    layer.out_channels for layer in self.controlnet_cond_embedding.blocks[1::2]
                ),
                scale=conditioning_scale,
            )

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
        controlnet_cond: torch.Tensor,
        conditioning_scale: float = 1.0,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guess_mode: bool = False,
        metadata: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[ControlNetOutput, Tuple[Tuple[torch.Tensor, ...], torch.Tensor]]:
        # Start from upstream logic, inserting metadata into the timestep embeddings.

        channel_order = self.config.controlnet_conditioning_channel_order
        if channel_order == "bgr":
            controlnet_cond = torch.flip(controlnet_cond, dims=[1])
        elif channel_order != "rgb":
            raise ValueError(f"unknown `controlnet_conditioning_channel_order`: {channel_order}")

        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        timesteps = timestep
        if not torch.is_tensor(timesteps):
            is_mps = sample.device.type == "mps"
            is_npu = sample.device.type == "npu"
            if isinstance(timestep, float):
                dtype = torch.float32 if (is_mps or is_npu) else torch.float64
            else:
                dtype = torch.int32 if (is_mps or is_npu) else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps).to(dtype=sample.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)

        class_emb = self.get_class_embed(sample=sample, class_labels=class_labels)
        if class_emb is not None:
            if self.config.class_embed_type == "timestep":
                class_emb = class_emb.to(dtype=sample.dtype)
            emb = emb + class_emb

        aug_emb = self.get_aug_embed(
            emb=emb, encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs or {}
        )
        if aug_emb is not None:
            emb = emb + aug_emb

        md_emb = self._encode_metadata(metadata=metadata, dtype=sample.dtype)
        if md_emb is not None:
            emb = emb + md_emb

        sample = self.conv_in(sample)
        controlnet_cond = self.controlnet_cond_embedding(controlnet_cond)
        sample = sample + controlnet_cond

        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
            down_block_res_samples += res_samples

        if self.mid_block is not None:
            if hasattr(self.mid_block, "has_cross_attention") and self.mid_block.has_cross_attention:
                sample = self.mid_block(
                    sample,
                    emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
            else:
                sample = self.mid_block(sample, emb)

        controlnet_down_block_res_samples = ()
        for down_block_res_sample, controlnet_block in zip(down_block_res_samples, self.controlnet_down_blocks):
            down_block_res_sample = controlnet_block(down_block_res_sample)
            controlnet_down_block_res_samples = controlnet_down_block_res_samples + (down_block_res_sample,)
        down_block_res_samples = controlnet_down_block_res_samples

        mid_block_res_sample = self.controlnet_mid_block(sample)

        if guess_mode and not self.config.global_pool_conditions:
            scales = torch.logspace(-1, 0, len(down_block_res_samples) + 1, device=sample.device) * conditioning_scale
            down_block_res_samples = [sample * scale for sample, scale in zip(down_block_res_samples, scales)]
            mid_block_res_sample = mid_block_res_sample * scales[-1]
        else:
            down_block_res_samples = [sample * conditioning_scale for sample in down_block_res_samples]
            mid_block_res_sample = mid_block_res_sample * conditioning_scale

        if self.config.global_pool_conditions:
            down_block_res_samples = [
                torch.mean(sample, dim=(2, 3), keepdim=True) for sample in down_block_res_samples
            ]
            mid_block_res_sample = torch.mean(mid_block_res_sample, dim=(2, 3), keepdim=True)

        if not return_dict:
            return (down_block_res_samples, mid_block_res_sample)

        return ControlNetOutput(
            down_block_res_samples=down_block_res_samples, mid_block_res_sample=mid_block_res_sample
        )
