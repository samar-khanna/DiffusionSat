export WANDB_API_KEY="API KEY"
export WANDB_PROJ="finetune_controlnet"
export WANDB_ENTITY="YOUR USER OR TEAM NAME"

export HF_HOME="./hf_cache"
export MODEL_NAME="stabilityai/stable-diffusion-2-1"

export UNET_PATH="./diffusionsat_checkpoints/finetune_sd21_sn-satlas-fmow_snr5_md7norm_bs64/checkpoint-100000/"
export OUT_DIR="./outputs/controlnet3d-mixattn_sd21_md7norm_texas"

accelerate launch --config_file="$1" --mixed_precision="fp16" --main_process_port=47860  --gpu_ids="$CUDA_VISIBLE_DEVICES"  train_controlnet.py \
  --wandb="${WANDB_PROJ}" \
  --pretrained_model_name_or_path=$MODEL_NAME --unet_path=$UNET_PATH \
  --dataset="texas" --texas_task="random" --shardlist="./datasets/texas_shardlist_disk.txt" \
  --num_metadata=7 --num_cond=3 --temporal --temporal_attn \
  --resolution=256 \
  --train_batch_size=32 \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --max_train_steps=50000 \
  --checkpointing_steps=10000 \
  --checkpoint_preempt_steps=500 \
  --learning_rate=8e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="${OUT_DIR}" \
  --resume_from_checkpoint="latest"
