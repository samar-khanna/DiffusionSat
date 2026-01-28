# DiffusionSat Custom Pipelines

Custom community pipelines for loading DiffusionSat checkpoints directly with `diffusers.DiffusionPipeline.from_pretrained()`.

> See [Diffusers Community Pipeline Documentation](https://huggingface.co/docs/diffusers/using-diffusers/custom_pipeline_overview)

## Available Pipelines

This directory contains two custom pipelines:

1. **`pipeline_diffusionsat.py`**: Standard text-to-image pipeline with DiffusionSat metadata support.
2. **`pipeline_diffusionsat_controlnet.py`**: ControlNet pipeline with DiffusionSat metadata and conditional metadata support.

## Setup

The checkpoint folder (`ckpt/diffusionsat/`) should contain the standard diffusers components (unet, vae, scheduler, etc.). You can reference these pipeline files directly from this directory or copy them to your checkpoint folder.

## Usage

### 1. Text-to-Image Pipeline

Use `pipeline_diffusionsat.py` for standard generation.

```python
import torch
from diffusers import DiffusionPipeline

# Load pipeline
pipe = DiffusionPipeline.from_pretrained(
    "path/to/ckpt/diffusionsat",
    custom_pipeline="./custom_pipelines/pipeline_diffusionsat.py",  # Path to this file
    torch_dtype=torch.float16,
    trust_remote_code=True,
)
pipe = pipe.to("cuda")

# Optional: Metadata (normalized lat, lon, timestamp, GSD, etc.)
# metadata = [0.5, -0.3, 0.7, 0.2, 0.1, 0.0, 0.5] 

# Generate
image = pipe(
    "satellite image of farmland",
    metadata=None,  # Optional
    num_inference_steps=30,
).images[0]
```

### 2. ControlNet Pipeline

Use `pipeline_diffusionsat_controlnet.py` for ControlNet generation.

```python
import torch
from diffusers import DiffusionPipeline, ControlNetModel
from diffusers.utils import load_image

# 1. Load ControlNet
controlnet = ControlNetModel.from_pretrained(
    "path/to/ckpt/diffusionsat/controlnet",
    torch_dtype=torch.float16
)

# 2. Load Pipeline with ControlNet
pipe = DiffusionPipeline.from_pretrained(
    "path/to/ckpt/diffusionsat",
    controlnet=controlnet,
    custom_pipeline="./custom_pipelines/pipeline_diffusionsat_controlnet.py", # Path to this file
    torch_dtype=torch.float16,
    trust_remote_code=True,
)
pipe = pipe.to("cuda")

# 3. Prepare Control Image
control_image = load_image("path/to/conditioning_image.png")

# 4. Generate
# metadata: Target image metadata (optional)
# cond_metadata: Conditioning image metadata (optional)

image = pipe(
    "satellite image of farmland",
    image=control_image,
    metadata=None,       
    cond_metadata=None,
    num_inference_steps=30,
).images[0]
```
