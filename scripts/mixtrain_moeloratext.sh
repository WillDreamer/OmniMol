#!/bin/bash

PROMPT_VERSION="llama3"
MODEL_VERSION="llama"

GRAPH_TOWER="moleculestm"
INIT_CHECKPOINT_GNN="assets/moleculestm.pth"

CHECKPOINT_FOLDER_PREFIX="_checkpoints"
# TASK="forward:1/retrosynthesis:1/reagent:1/homolumo:1/molcap:1/solvent:1/catalyst:1/yield:1/experiment:0.5"
TASK="forward:1/retrosynthesis:1/reagent:1/homolumo:1/molcap:1/solvent:1/catalyst:1/yield:1"
PROJECTOR="naive_linear"
REMARK="1B-deepseek-moe-5EP-qurater-sharedEP-clip-alpha-embed-Tok2-8tasks-puretext"

export WANDB_ENTITY="Omni-Mol"
export WANDB_PROJECT="${WANDB_ENTITY}_${PROMPT_VERSION}"
export WANDB_API_KEY="ba70fcbc92808cc7a1750dd80ac3908295e6854f"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# --deepspeed scripts/zero_configs/zero2.json \
deepspeed --master_port 29505 train.py \
    --training_recipe puretext \
    --use_alpha True \
    --task_config $TASK \
    --model_name_or_path meta-llama/Llama-3.2-1B-Instruct  \
    --base_model meta-llama/Llama-3.2-1B-Instruct  \
    --language_backbone_name $MODEL_VERSION \
    --version $PROMPT_VERSION \
    --data_path data/train \
    --remove_unused_columns False \
    --data_type $TASK \
    --graph_tower $GRAPH_TOWER \
    --mm_projector_type $PROJECTOR \
    --graph_init_checkpoint $INIT_CHECKPOINT_GNN \
    --pretrain_mm_mlp_adapter assets/mm_projector.bin \
    --bf16 True \
    --output_dir $CHECKPOINT_FOLDER_PREFIX/$MODEL_VERSION-$REMARK \
    --num_train_epochs 15 \
    --per_device_train_batch_size 12 \
    --per_device_eval_batch_size 12 \
    --gradient_accumulation_steps 1 \
    --stop_epoch 15 \
    --eval_strategy "no" \
    --eval_steps 500 \
    --split_eval False \
    --val_ratio 0.1 \
    --eval_on_start False \
    --save_strategy "epoch" \
    --save_total_limit 5 \
    --learning_rate 8e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.0075 \
    --lr_scheduler_type "cosine" \
    --logging_steps 100 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --report_to all \
    --logging_dir tf-logs/$TASK-llava-$GRAPH_TOWER-$MODEL_VERSION-$PROJECTOR-$REMARK \
    --moe_class deepseek \
    --moe_mode second_quarter\
    --ep_size 1 \
    --num_experts 5 \
    --use_residual True \
    --router_aux_loss_coef 0.01 \
    --is_training True \
    --top_k_experts 2 \
    --use_task_loss False \
    --ddp_find_unused_parameters False \
    --norm_topk_prob False

