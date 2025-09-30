#!/bin/bash

# Training script for Flux LoRA with Accelerate

export MODEL_NAME="black-forest-labs/FLUX.1-Fill-dev"
export DATASET_NAME="./3d_icon"
export OUTPUT_DIR="./stage2_ckpt"

accelerate launch diffusers/examples/advanced_diffusion_training/train_dreambooth_lora_flux_fill_concat_d_stage2_d2i.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --instance_prompt="A beautiful scene" \
  --output_dir=$OUTPUT_DIR \
  --caption_column=prompt \
  --mixed_precision=bf16 \
  --resolution=1024 \
  --train_batch_size=6 \
  --repeats=1 \
  --report_to=wandb \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --learning_rate=1e-4 \
  --text_encoder_lr=1.0 \
  --optimizer=AdamW \
  --lr_scheduler=constant \
  --lr_warmup_steps=0 \
  --rank=64 \
  --checkpointing_steps=8400 \
  --validation_epochs=1 \
  --num_train_epochs=10 \
  --seed=0\
  --validation_prompt="A beautiful scene"\


