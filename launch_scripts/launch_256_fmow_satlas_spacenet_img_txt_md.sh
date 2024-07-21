export WANDB_API_KEY="API KEY"
export WANDB_PROJ="finetune_single_image"
export WANDB_ENTITY="YOUR USERNAME"

export HF_HOME="./hf_cache"
export MODEL_NAME="stabilityai/stable-diffusion-2-1"

export OUT_DIR="./outputs/finetune_single_image_256_md7_snr5_bs16"

accelerate launch --config_file="$1" --mixed_precision="fp16" --main_process_port=42960  --gpu_ids="$CUDA_VISIBLE_DEVICES" train_text_to_image.py \
  --wandb="${WANDB_PROJ}" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --use_ema \
  --num_metadata=7 \
  --snr_gamma=5.0 \
  --resolution=256 --center_crop  \
  --train_batch_size=16 \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --max_train_steps=150000 \
  --checkpointing_steps=10000 \
  --checkpoint_preempt_steps=500 \
  --learning_rate=4e-06 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="${OUT_DIR}" \
  --resume_from_checkpoint="latest"
