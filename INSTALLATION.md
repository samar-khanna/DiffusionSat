# Environment Installation Instructions

We use conda to create our environments. You will have to do the following:
```bash
cd DiffusionSat 
conda create -n diffusionsat python=3.10

# if you want cuda 11.8, run this replace the index url with https://download.pytorch.org/whl/cu118
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121
pip install -e ".[torch]"  # install editable diffusers
pip install -r remaining_requirements.txt
```