#!/bin/bash

echo "Starting the training script..."

# 参数配置
max_num_refs=6
ref_size=512
height=512
width=512
bsz=1

lr="8e-5"
optimizer="adamw"

rank=512
gradient_accumulation_steps=1
total_bsz=$((bsz * gradient_accumulation_steps * num_processes))

datetime=$(date +"%Y_%m_%d_%H_%M")

version="num_refs${max_num_refs}_lora_rank_${rank}_${optimizer}_lr_${lr}_bsz_${total_bsz}_ref_${ref_size}_tgt_${height}_${datetime}"


output_dir="${version}"
mkdir -p "${output_dir}"

# 启动训练
accelerate launch --config_file "./accelerate_config.yaml" \
    --num_machines=${num_machines} \
    --num_processes=${num_processes} \
    --machine_rank=${ARNOLD_ID} \
    --main_process_ip=${METIS_WORKER_0_HOST} \
    --main_process_port=${port} \
    train.py \
    --output_dir ${output_dir} \
    --mixed_precision "bf16" \
    --pretrained_model_name_or_path "black-forest-labs/FLUX.1-dev" \
    --max_num_refs ${max_num_refs} \
    --ref_size ${ref_size} \
    --height ${height} \
    --width ${width} \
    --rank ${rank} \
    --weighting_scheme "logit_normal" \
    --num_train_epochs 10 \
    --train_batch_size ${bsz} \
    --learning_rate ${lr} \
    --optimizer ${optimizer} \
    --shuffle \
    --dataloader_num_workers 8 \
    --guidance_scale 1 \
    --validation_steps 1000 \
    --checkpointing_steps 1000 \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --lr_scheduler "constant" \
    --lr_warmup_steps 0 \
    --seed 42 \
    --adam_beta1 0.9 \
    --adam_beta2 0.999 \
    --adam_weight_decay 1e-2 \
    --adam_epsilon 1e-8 \
    --max_grad_norm 1.0 \
    --report_to "tensorboard" \
    --ema \
    --gradient_checkpointing \
    --get_attn_maps