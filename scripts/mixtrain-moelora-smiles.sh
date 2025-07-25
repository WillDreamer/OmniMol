#!/bin/bash

cd /wanghaixin/OmniMol
# PROMPT_VERSION="tinyllama"
# MODEL_VERSION="llama"

PROMPT_VERSION="llama3"
MODEL_VERSION="llama"

GRAPH_TOWER="moleculestm"
if [ "$GRAPH_TOWER" == "graphmvp" ]; then
    INIT_CHECKPOINT_GNN="checkpoints/graphMVP.pth"
elif [ "$GRAPH_TOWER" == "moleculestm" ]; then
    INIT_CHECKPOINT_GNN="/wanghaixin/OmniMol/checkpoints/moleculestm.pth"
elif [ "$GRAPH_TOWER" == "himol" ]; then
    INIT_CHECKPOINT_GNN="/root/autodl-tmp/MoleculeMoE/MolMoE/checkpoints/himol_encoder.pth"
else
    echo "Not supported graph tower"
fi


CHECKPOINT_FOLDER_PREFIX="_checkpoints/moe"
# TASK="forward:1/retrosynthesis:1/reagent:1/homolumo:1/molcap:1/solvent:1/catalyst:1/yield:1/experiment:0.5"
# TASK="forward:1/retrosynthesis:1/reagent:1/homolumo:1/molcap:1/solvent:1/catalyst:1/yield:1/experiment:0.5/scf:0.25/complexity:1/tpsa:1/weight:1/dqa:1/logp:1"
# TASK="forward:1/retrosynthesis:1/reagent:1/homolumo:1/molcap:1/solvent:1/catalyst:1/yield:1/experiment:0.5/scf:0.25/complexity:1/tpsa:1/weight:1/dqa:1/logp:1"
PROJECTOR="naive_linear"
REMARK="1B-deepseek-moe-5expert-second-quarter-sharedEP-clip-alpha-embed-Tok2-smiles-4GPUs"


export WANDB_ENTITY="Omni-Mol"
export WANDB_PROJECT="${WANDB_ENTITY}_${PROMPT_VERSION}"
export WANDB_API_KEY="ba70fcbc92808cc7a1750dd80ac3908295e6854f"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
/root/anaconda3/bin/deepspeed --master_port 29505 /wanghaixin/OmniMol/train.py \
    --deepspeed scripts/zero_configs/zero2.json \
    --training_recipe moe+lora \
    --use_alpha True \
    --task_config $TASK \
    --model_name_or_path /wanghaixin/OmniMol/checkpoints/Llama-3.2-1B-Instruct \
    --base_model /wanghaixin/OmniMol/checkpoints/Llama-3.2-1B-Instruct \
    --language_backbone_name $MODEL_VERSION \
    --version $PROMPT_VERSION \
    --data_path /wanghaixin/OmniMol/Molecule-oriented_Instructions/train \
    --data_type $TASK \
    --graph_tower $GRAPH_TOWER \
    --mm_projector_type $PROJECTOR \
    --graph_init_checkpoint $INIT_CHECKPOINT_GNN \
    --pretrain_mm_mlp_adapter /wanghaixin/OmniMol/checkpoints/mm_projector.bin \
    --bf16 True \
    --output_dir $CHECKPOINT_FOLDER_PREFIX/$MODEL_VERSION-$REMARK \
    --num_train_epochs 15 \
    --per_device_train_batch_size 36 \
    --per_device_eval_batch_size 36 \
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
    --report_to tensorboard \
    --logging_dir /wanghaixin/OmniMol/tf-logs/$TASK-llava-$GRAPH_TOWER-$MODEL_VERSION-$PROJECTOR-$REMARK \
    --moe_class deepseek \
    --moe_mode second_quarter \
    --ep_size 1 \
    --num_experts 5 \
    --use_residual True \
    --router_aux_loss_coef 0.01 \
    --is_training True \
    --top_k_experts 2 \
    --if_smiles True
