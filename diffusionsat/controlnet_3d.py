###########################################################################
# References:
# https://github.com/huggingface/diffusers/
###########################################################################
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import einops
import torch
from torch import nn
from torch.nn import functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput, logging
from diffusers.models.attention_processor import AttentionProcessor, AttnProcessor
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.unet_2d_blocks import (
    CrossAttnDownBlock2D,
    DownBlock2D,
    UNetMidBlock2DCrossAttn,
    get_down_block,
)
from diffusers.models.transformer_temporal import TransformerTemporalModel

from .sat_unet import SatUNet


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class MixingParam(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input.clamp(min=0., max=1.) # the value in iterative = 2

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()


@dataclass
class ControlNetOutput(BaseOutput):
    down_block_res_samples: Tuple[torch.Tensor]
    mid_block_res_sample: torch.Tensor


class ControlNetConditioningEmbedding(nn.Module):
    """
    Quoting from https://arxiv.org/abs/2302.05543: "Stable Diffusion uses a pre-processing method similar to VQ-GAN
    [11] to convert the entire dataset of 512 × 512 images into smaller 64 × 64 “latent images” for stabilized
    training. This requires ControlNets to convert image-based conditions to 64 × 64 feature space to match the
    convolution size. We use a tiny network E(·) of four convolution layers with 4 × 4 kernels and 2 × 2 strides
    (activated by ReLU, channels are 16, 32, 64, 128, initialized with Gaussian weights, trained jointly with the full
    model) to encode image-space conditions ... into feature maps ..."
    """

    def __init__(
        self,
        conditioning_embedding_channels: int,
        conditioning_channels: int = 3,
        block_out_channels: Tuple[int] = (16, 32, 96, 256),
    ):
        super().__init__()

        self.conv_in = nn.Conv3d(conditioning_channels, block_out_channels[0], kernel_size=3, padding=1)

        self.blocks = nn.ModuleList([])

        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.blocks.append(nn.Conv3d(channel_in, channel_in, kernel_size=3, padding=1))
            self.blocks.append(nn.Conv3d(channel_in, channel_out, kernel_size=(1, 3, 3), padding=(0, 1, 1), stride=(1, 2, 2)))

        self.conv_out = zero_module(
            nn.Conv3d(block_out_channels[-1], conditioning_embedding_channels, kernel_size=3, padding=1)
        )

    def forward(self, conditioning):
        embedding = self.conv_in(conditioning)
        embedding = F.silu(embedding)

        for block in self.blocks:
            embedding = block(embedding)
            embedding = F.silu(embedding)

        embedding = self.conv_out(embedding)

        return embedding


class ControlNetModel3D(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        in_channels: int = 4,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        down_block_types: Tuple[str] = (
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        only_cross_attention: Union[bool, Tuple[bool]] = False,
        block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
        layers_per_block: int = 2,
        downsample_padding: int = 1,
        mid_block_scale_factor: float = 1,
        act_fn: str = "silu",
        norm_num_groups: Optional[int] = 32,
        norm_eps: float = 1e-5,
        cross_attention_dim: int = 1280,
        attention_head_dim: Union[int, Tuple[int]] = 8,
        use_linear_projection: bool = False,
        class_embed_type: Optional[str] = None,
        num_class_embeds: Optional[int] = None,
        upcast_attention: bool = False,
        resnet_time_scale_shift: str = "default",
        projection_class_embeddings_input_dim: Optional[int] = None,
        controlnet_conditioning_channel_order: str = "rgb",
        conditioning_embedding_out_channels: Optional[Tuple[int]] = (16, 32, 96, 256),
        global_pool_conditions: bool = False,
        conditioning_in_channels=3,
        use_temporal_transformer=False,
        use_metadata=True,
        num_metadata=7,
    ):
        super().__init__()

        # Check inputs
        if len(block_out_channels) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `block_out_channels` as `down_block_types`. `block_out_channels`: {block_out_channels}. `down_block_types`: {down_block_types}."
            )

        if not isinstance(only_cross_attention, bool) and len(only_cross_attention) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `only_cross_attention` as `down_block_types`. `only_cross_attention`: {only_cross_attention}. `down_block_types`: {down_block_types}."
            )

        if not isinstance(attention_head_dim, int) and len(attention_head_dim) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `attention_head_dim` as `down_block_types`. `attention_head_dim`: {attention_head_dim}. `down_block_types`: {down_block_types}."
            )

        # input
        conv_in_kernel = 3
        conv_in_padding = (conv_in_kernel - 1) // 2
        self.conv_in = nn.Conv2d(
            in_channels, block_out_channels[0], kernel_size=conv_in_kernel, padding=conv_in_padding
        )

        # time
        time_embed_dim = block_out_channels[0] * 4

        self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
        timestep_input_dim = block_out_channels[0]

        self.time_embedding = TimestepEmbedding(
            timestep_input_dim,
            time_embed_dim,
            act_fn=act_fn,
        )

        ## Metadata embedding for each metadata item
        if use_metadata:
            self.metadata_embedding = nn.ModuleList([
                TimestepEmbedding(timestep_input_dim, time_embed_dim) for _ in range(num_metadata)
            ])
            # self.metadata_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)
            self.num_metadata = num_metadata
        else:
            self.metadata_embedding = None

        # class embedding
        if class_embed_type is None and num_class_embeds is not None:
            self.class_embedding = nn.Embedding(num_class_embeds, time_embed_dim)
        elif class_embed_type == "timestep":
            self.class_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)
        elif class_embed_type == "identity":
            self.class_embedding = nn.Identity(time_embed_dim, time_embed_dim)
        elif class_embed_type == "projection":
            if projection_class_embeddings_input_dim is None:
                raise ValueError(
                    "`class_embed_type`: 'projection' requires `projection_class_embeddings_input_dim` be set"
                )
            # The projection `class_embed_type` is the same as the timestep `class_embed_type` except
            # 1. the `class_labels` inputs are not first converted to sinusoidal embeddings
            # 2. it projects from an arbitrary input dimension.
            #
            # Note that `TimestepEmbedding` is quite general, being mainly linear layers and activations.
            # When used for embedding actual timesteps, the timesteps are first converted to sinusoidal embeddings.
            # As a result, `TimestepEmbedding` can be passed arbitrary vectors.
            self.class_embedding = TimestepEmbedding(projection_class_embeddings_input_dim, time_embed_dim)
        else:
            self.class_embedding = None

        # control net conditioning embedding
        self.controlnet_cond_embedding = ControlNetConditioningEmbedding(
            conditioning_channels=conditioning_in_channels,
            conditioning_embedding_channels=block_out_channels[0],
            block_out_channels=conditioning_embedding_out_channels,
        )

        self.down_blocks = nn.ModuleList([])
        self.controlnet_down_blocks = nn.ModuleList([])
        self.temporal_blocks = None
        if use_temporal_transformer:
            self.temporal_blocks = nn.ModuleList([])
            self.mixing_attns = []

        if isinstance(only_cross_attention, bool):
            only_cross_attention = [only_cross_attention] * len(down_block_types)

        if isinstance(attention_head_dim, int):
            attention_head_dim = (attention_head_dim,) * len(down_block_types)

        # down
        output_channel = block_out_channels[0]

        controlnet_block = nn.Conv3d(output_channel, output_channel, kernel_size=1)
        controlnet_block = zero_module(controlnet_block)
        self.controlnet_down_blocks.append(controlnet_block)

        if self.temporal_blocks is not None:
            self.temporal_blocks.append(
                TransformerTemporalModel(
                    num_attention_heads=8, attention_head_dim=64,
                    in_channels=output_channel, num_layers=1
                )
            )
            self.mixing_attns.append(nn.Parameter(torch.tensor(0.)))

        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=attention_head_dim[i],
                downsample_padding=downsample_padding,
                use_linear_projection=use_linear_projection,
                only_cross_attention=only_cross_attention[i],
                upcast_attention=upcast_attention,
                resnet_time_scale_shift=resnet_time_scale_shift,
            )
            self.down_blocks.append(down_block)

            for _ in range(layers_per_block):
                controlnet_block = nn.Conv3d(output_channel, output_channel, kernel_size=1)
                controlnet_block = zero_module(controlnet_block)
                self.controlnet_down_blocks.append(controlnet_block)
                if self.temporal_blocks is not None:
                    self.temporal_blocks.append(nn.Identity())
                    self.mixing_attns.append(nn.Parameter(torch.tensor(0.), requires_grad=False))
                    # controlnet_transformer = TransformerTemporalModel(
                    #     output_channel // attention_head_dim[i],
                    #     attention_head_dim[i],
                    #     in_channels=output_channel,
                    #     num_layers=1,
                    #     cross_attention_dim=cross_attention_dim,
                    #     norm_num_groups=norm_num_groups,
                    # )
                    # self.temporal_blocks.append(controlnet_transformer)
                    # self.mixing_attns.append(nn.Parameter(torch.tensor(0.)))

            if not is_final_block:
                controlnet_block = nn.Conv3d(output_channel, output_channel, kernel_size=1)
                controlnet_block = zero_module(controlnet_block)
                self.controlnet_down_blocks.append(controlnet_block)
                if self.temporal_blocks is not None:
                    # self.temporal_blocks.append(nn.Identity())
                    # self.mixing_attns.append(nn.Parameter(torch.tensor(0, dtype=torch.bool), requires_grad=False))
                    controlnet_transformer = TransformerTemporalModel(
                        output_channel // attention_head_dim[i],
                        attention_head_dim[i],
                        in_channels=output_channel,
                        num_layers=1,
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=norm_num_groups,
                    )
                    self.temporal_blocks.append(controlnet_transformer)
                    self.mixing_attns.append(nn.Parameter(torch.tensor(0.)))

            # # Only 1 transformer per block level
            # if self.temporal_blocks is not None:
            #     self.temporal_blocks.pop(-1)
            #     controlnet_transformer = TransformerTemporalModel(
            #             output_channel // attention_head_dim[i],
            #             attention_head_dim[i],
            #             in_channels=output_channel,
            #             num_layers=1,
            #             cross_attention_dim=cross_attention_dim,
            #             norm_num_groups=norm_num_groups,
            #         )
            #     # controlnet_transformer = zero_module(controlnet_transformer)
            #     self.temporal_blocks.append(controlnet_transformer)
            #     self.mixing_attns.pop(-1)
            #     self.mixing_attns.append(nn.Parameter(torch.tensor(0.)))

        if self.temporal_blocks is not None:
            self.mixing_attns = nn.ParameterList(self.mixing_attns)
            assert len(self.temporal_blocks) == len(self.controlnet_down_blocks), \
                f"{len(self.temporal_blocks)} temporal blocks != {len(self.controlnet_down_blocks)} down blocks"
            assert len(self.mixing_attns) == len(self.temporal_blocks), \
                f"{len(self.mixing_attns)} mixing params != {len(self.temporal_blocks)} down blocks"

        # mid
        mid_block_channel = block_out_channels[-1]

        controlnet_block = nn.Conv3d(mid_block_channel, mid_block_channel, kernel_size=1)
        controlnet_block = zero_module(controlnet_block)
        self.controlnet_mid_block = controlnet_block

        self.mid_block = UNetMidBlock2DCrossAttn(
            in_channels=mid_block_channel,
            temb_channels=time_embed_dim,
            resnet_eps=norm_eps,
            resnet_act_fn=act_fn,
            output_scale_factor=mid_block_scale_factor,
            resnet_time_scale_shift=resnet_time_scale_shift,
            cross_attention_dim=cross_attention_dim,
            attn_num_head_channels=attention_head_dim[-1],
            resnet_groups=norm_num_groups,
            use_linear_projection=use_linear_projection,
            upcast_attention=upcast_attention,
        )

    @classmethod
    def from_unet(
        cls,
        unet: SatUNet,
        conditioning_in_channels: int = 3,
        use_temporal_transformer: bool = False,
        controlnet_conditioning_channel_order: str = "rgb",
        conditioning_embedding_out_channels: Optional[Tuple[int]] = (16, 32, 96, 256),
        load_weights_from_unet: bool = True,
    ):
        r"""
        Instantiate Controlnet class from UNet2DConditionModel.

        Parameters:
            unet (`UNet2DConditionModel`):
                UNet model which weights are copied to the ControlNet. Note that all configuration options are also
                copied where applicable.
        """
        controlnet = cls(
            in_channels=unet.config.in_channels,
            flip_sin_to_cos=unet.config.flip_sin_to_cos,
            freq_shift=unet.config.freq_shift,
            down_block_types=unet.config.down_block_types,
            only_cross_attention=unet.config.only_cross_attention,
            block_out_channels=unet.config.block_out_channels,
            layers_per_block=unet.config.layers_per_block,
            downsample_padding=unet.config.downsample_padding,
            mid_block_scale_factor=unet.config.mid_block_scale_factor,
            act_fn=unet.config.act_fn,
            norm_num_groups=unet.config.norm_num_groups,
            norm_eps=unet.config.norm_eps,
            cross_attention_dim=unet.config.cross_attention_dim,
            attention_head_dim=unet.config.attention_head_dim,
            use_linear_projection=unet.config.use_linear_projection,
            class_embed_type=unet.config.class_embed_type,
            num_class_embeds=unet.config.num_class_embeds,
            upcast_attention=unet.config.upcast_attention,
            resnet_time_scale_shift=unet.config.resnet_time_scale_shift,
            # projection_class_embeddings_input_dim=unet.config.projection_class_embeddings_input_dim,
            controlnet_conditioning_channel_order=controlnet_conditioning_channel_order,
            conditioning_embedding_out_channels=conditioning_embedding_out_channels,
            conditioning_in_channels=conditioning_in_channels,
            use_metadata=unet.metadata_embedding is not None,
            num_metadata=getattr(unet, 'num_metadata', 0),
            use_temporal_transformer=use_temporal_transformer,
        )

        if load_weights_from_unet:
            controlnet.conv_in.load_state_dict(unet.conv_in.state_dict())
            controlnet.time_proj.load_state_dict(unet.time_proj.state_dict())
            controlnet.time_embedding.load_state_dict(unet.time_embedding.state_dict())
            if controlnet.metadata_embedding is not None:
                controlnet.metadata_embedding.load_state_dict(unet.metadata_embedding.state_dict())

            if controlnet.class_embedding:
                controlnet.class_embedding.load_state_dict(unet.class_embedding.state_dict())

            controlnet.down_blocks.load_state_dict(unet.down_blocks.state_dict())
            controlnet.mid_block.load_state_dict(unet.mid_block.state_dict())

        return controlnet

    @property
    # Copied from diffusers.models.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "set_processor"):
                processors[f"{name}.processor"] = module.processor

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        Parameters:
            `processor (`dict` of `AttentionProcessor` or `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                of **all** `Attention` layers.
            In case `processor` is a dict, the key needs to define the path to the corresponding cross attention processor. This is strongly recommended when setting trainable attention processors.:

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    # Copied from diffusers.models.unet_2d_condition.UNet2DConditionModel.set_default_attn_processor
    def set_default_attn_processor(self):
        """
        Disables custom attention processors and sets the default attention implementation.
        """
        self.set_attn_processor(AttnProcessor())

    # Copied from diffusers.models.unet_2d_condition.UNet2DConditionModel.set_attention_slice
    def set_attention_slice(self, slice_size):
        r"""
        Enable sliced attention computation.

        When this option is enabled, the attention module will split the input tensor in slices, to compute attention
        in several steps. This is useful to save some memory in exchange for a small speed decrease.

        Args:
            slice_size (`str` or `int` or `list(int)`, *optional*, defaults to `"auto"`):
                When `"auto"`, halves the input to the attention heads, so attention will be computed in two steps. If
                `"max"`, maximum amount of memory will be saved by running only one slice at a time. If a number is
                provided, uses as many slices as `attention_head_dim // slice_size`. In this case, `attention_head_dim`
                must be a multiple of `slice_size`.
        """
        sliceable_head_dims = []

        def fn_recursive_retrieve_sliceable_dims(module: torch.nn.Module):
            if hasattr(module, "set_attention_slice"):
                sliceable_head_dims.append(module.sliceable_head_dim)

            for child in module.children():
                fn_recursive_retrieve_sliceable_dims(child)

        # retrieve number of attention layers
        for module in self.children():
            fn_recursive_retrieve_sliceable_dims(module)

        num_sliceable_layers = len(sliceable_head_dims)

        if slice_size == "auto":
            # half the attention head size is usually a good trade-off between
            # speed and memory
            slice_size = [dim // 2 for dim in sliceable_head_dims]
        elif slice_size == "max":
            # make smallest slice possible
            slice_size = num_sliceable_layers * [1]

        slice_size = num_sliceable_layers * [slice_size] if not isinstance(slice_size, list) else slice_size

        if len(slice_size) != len(sliceable_head_dims):
            raise ValueError(
                f"You have provided {len(slice_size)}, but {self.config} has {len(sliceable_head_dims)} different"
                f" attention layers. Make sure to match `len(slice_size)` to be {len(sliceable_head_dims)}."
            )

        for i in range(len(slice_size)):
            size = slice_size[i]
            dim = sliceable_head_dims[i]
            if size is not None and size > dim:
                raise ValueError(f"size {size} has to be smaller or equal to {dim}.")

        # Recursively walk through all the children.
        # Any children which exposes the set_attention_slice method
        # gets the message
        def fn_recursive_set_attention_slice(module: torch.nn.Module, slice_size: List[int]):
            if hasattr(module, "set_attention_slice"):
                module.set_attention_slice(slice_size.pop())

            for child in module.children():
                fn_recursive_set_attention_slice(child, slice_size)

        reversed_slice_size = list(reversed(slice_size))
        for module in self.children():
            fn_recursive_set_attention_slice(module, reversed_slice_size)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (CrossAttnDownBlock2D, DownBlock2D)):
            module.gradient_checkpointing = value

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        controlnet_cond: torch.FloatTensor,
        conditioning_scale: float = 1.0,
        metadata: Optional[torch.Tensor] = None,
        cond_metadata: Optional[torch.Tensor] = None,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guess_mode: bool = False,
        return_dict: bool = True,
        mixing_autograd_fn=MixingParam.apply
    ) -> Union[ControlNetOutput, Tuple]:
        # check channel order
        channel_order = self.config.controlnet_conditioning_channel_order

        # Number of cond images
        num_frames = controlnet_cond.shape[2]

        if channel_order == "rgb":
            # in rgb order by default
            ...
        elif channel_order == "bgr":
            controlnet_cond = torch.flip(controlnet_cond, dims=[1])
        else:
            raise ValueError(f"unknown `controlnet_conditioning_channel_order`: {channel_order}")

        # prepare attention_mask
        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])
        timesteps = timesteps.repeat_interleave(num_frames)

        encoder_hidden_states = encoder_hidden_states.repeat_interleave(num_frames, dim=0)

        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=sample.dtype)

        emb = self.time_embedding(t_emb, timestep_cond)

        # CUSTOM: metadata
        if self.metadata_embedding is not None:
            assert metadata is not None
            assert len(metadata.shape) == 2 and metadata.shape[1] == self.num_metadata, \
                f"Invalid metadata shape: {metadata.shape}. Need batch x num_metadata"

            md_bsz = metadata.shape[0]
            metadata = self.time_proj(metadata.view(-1)).view(md_bsz, self.num_metadata, -1)  # (N, num_md, D)
            metadata = metadata.to(dtype=self.dtype)
            for i, md_embed in enumerate(self.metadata_embedding):
                md_emb = md_embed(metadata[:, i, :]).repeat_interleave(num_frames, dim=0)  # (N*T, D)
                emb = emb + md_emb  # (N*T, D)

            if cond_metadata is not None:
                assert cond_metadata.shape[1] == self.num_metadata, \
                    f"Invalid cond metadata shape: {cond_metadata.shape}. Need batch x num_metadata x (optional) num_cond"
                if len(cond_metadata.shape) == 3:
                    md_bsz = cond_metadata.shape[0]
                    num_cond = cond_metadata.shape[2]
                    cond_metadata = self.time_proj(cond_metadata.view(-1)).view(md_bsz, self.num_metadata, num_cond, -1)  # (N, num_md,T, D)
                    cond_metadata = cond_metadata.to(dtype=self.dtype)
                    for i, md_embed in enumerate(self.metadata_embedding):
                        md_emb = md_embed(cond_metadata[:, i, :, :]).view(md_bsz * num_frames, -1)  # (N*T, D)
                        emb = emb + md_emb
                else:
                    assert len(cond_metadata.shape) == 2
                    md_bsz = cond_metadata.shape[0]
                    cond_metadata = self.time_proj(cond_metadata.view(-1)).view(md_bsz, self.num_metadata, -1)  # (N, num_md, D)
                    cond_metadata = cond_metadata.to(dtype=self.dtype)
                    for i, md_embed in enumerate(self.metadata_embedding):
                        md_emb = md_embed(cond_metadata[:, i, :]).repeat_interleave(num_frames, dim=0)  # (N*T, D)
                        emb = emb + md_emb  # (N*T, D)

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")

            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)

            class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)
            emb = emb + class_emb

        # 2. pre-process
        sample = self.conv_in(sample)

        controlnet_cond = self.controlnet_cond_embedding(controlnet_cond)
        controlnet_cond = einops.rearrange(controlnet_cond, 'b c t h w -> (b t) c h w')

        sample = sample.repeat_interleave(num_frames, dim=0) + controlnet_cond  # (b*t, c, h, w)

        # 3. down
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

        # 4. mid
        if self.mid_block is not None:
            sample = self.mid_block(
                sample,
                emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
            )

        # 5. Control net blocks

        controlnet_down_block_res_samples = ()

        for i, (down_block_res_sample, controlnet_block) in enumerate(zip(down_block_res_samples, self.controlnet_down_blocks)):
            down_block_res_sample = einops.rearrange(down_block_res_sample, '(b t) c h w -> b c t h w', t=num_frames)
            down_block_res_sample = controlnet_block(down_block_res_sample)
            if self.temporal_blocks is not None:
                temporal_block = self.temporal_blocks[i]
                if isinstance(temporal_block, nn.Identity):
                    down_block_res_sample = temporal_block(down_block_res_sample)
                else:
                    t_alpha = mixing_autograd_fn(self.mixing_attns[i]).to(dtype=down_block_res_sample.dtype)
                    down_block_temporal_sample = einops.rearrange(down_block_res_sample, 'b c t h w -> (b t) c h w')
                    down_block_temporal_sample = temporal_block(
                        down_block_temporal_sample, num_frames=num_frames, cross_attention_kwargs=cross_attention_kwargs
                    ).sample
                    down_block_temporal_sample = einops.rearrange(down_block_temporal_sample, '(b t) c h w -> b c t h w', t=num_frames)
                    down_block_res_sample = t_alpha * down_block_temporal_sample + (1 - t_alpha) * down_block_res_sample
            controlnet_down_block_res_samples = controlnet_down_block_res_samples + (down_block_res_sample.sum(dim=2),)  # (b, c, h, w)

        down_block_res_samples = controlnet_down_block_res_samples

        sample = einops.rearrange(sample, '(b t) c h w -> b c t h w', t=num_frames)
        mid_block_res_sample = self.controlnet_mid_block(sample)
        mid_block_res_sample = mid_block_res_sample.sum(dim=2)  # (b, c, h, w)

        # 6. scaling
        if guess_mode:
            scales = torch.logspace(-1, 0, len(down_block_res_samples) + 1, device=sample.device)  # 0.1 to 1.0

            scales = scales * conditioning_scale
            down_block_res_samples = [sample * scale for sample, scale in zip(down_block_res_samples, scales)]
            mid_block_res_sample = mid_block_res_sample * scales[-1]  # last one
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


def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module
