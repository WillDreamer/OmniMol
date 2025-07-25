#!/bin/bash

#############################################
# Part 1: Training
#############################################

PROMPT_VERSION="llama3"
MODEL_VERSION="llama"

GRAPH_TOWER="moleculestm"
INIT_CHECKPOINT_GNN="assets/moleculestm.pth"

CHECKPOINT_FOLDER_PREFIX="_checkpoints"
TASK="forward:1/retrosynthesis:1/reagent:1/homolumo:1/molcap:1/solvent:1/catalyst:1/yield:1/experiment:1/tpsa:1/weight:1/dqa:1/logp:1/iupac:1/textguidedmolgen:1/molediting:1"
PROJECTOR="naive_linear"
REMARK="1B-deepseek-moe-5EP-qurater-sharedEP-clip-alpha-embed-Tok2-16tasks"

export WANDB_ENTITY="Omni-Mol"
export WANDB_PROJECT="${WANDB_ENTITY}_${PROMPT_VERSION}"
export WANDB_API_KEY="ba70fcbc92808cc7a1750dd80ac3908295e6854f"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HUGGINGFACE_HUB_TOKEN="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"

echo "========== Start Training =========="
deepspeed --master_port 29505 train.py \
    --deepspeed scripts/zero_configs/zero2.json \
    --training_recipe loramoe \
    --use_alpha True \
    --task_config $TASK \
    --model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
    --base_model meta-llama/Llama-3.2-1B-Instruct \
    --language_backbone_name $MODEL_VERSION \
    --version $PROMPT_VERSION \
    --data_path data/train \
    --data_type $TASK \
    --graph_tower $GRAPH_TOWER \
    --mm_projector_type $PROJECTOR \
    --graph_init_checkpoint $INIT_CHECKPOINT_GNN \
    --pretrain_mm_mlp_adapter assets/mm_projector.bin \
    --bf16 True \
    --output_dir $CHECKPOINT_FOLDER_PREFIX/$MODEL_VERSION-$REMARK \
    --num_train_epochs 15 \
    --per_device_train_batch_size 18 \
    --per_device_eval_batch_size 18 \
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

echo "========== Training Finished =========="


#############################################
# Part 2: Evaluation (Test)
#############################################

echo "========== Start Evaluation =========="

TYPE="loramoe"
PROMPT="llama3"
BACKBONE="meta-llama/Llama-3.2-1B-Instruct"
GRAPH_TOWER="moleculestm"
GRAPH_PATH="assets/moleculestm.pth"
BATCH_SIZE=1
DTYPE="bfloat16"
DEVICE="cuda"
MAX_NEW_TOKENS=512
NUM_BEAMS=1
TOP_P=1.0
TEMPERATURE=0.2
REPETITION_PENALTY=1.0
ADD_SELFIES=True
IS_TRAINING=False

EPOCH_LIST=(11 12 13 14 15)
CHECKPOINT_LIST=(154902 168984 183066 197148 211230)

TASK_LIST=(
     "forward"
     "reagent"
     "retrosynthesis"
     "homolumo"
     "molcap"
     "solvent"
     "catalyst"
     "yield_BH"
     "yield_SM"
     "logp"
     "weight"
     "tpsa"
     "experiment"
     "iupac2selfies"
     "molediting"
     "textguidedmolgen"
)

for i in "${!EPOCH_LIST[@]}"; do
    EPOCH=${EPOCH_LIST[$i]}
    CKPT=${CHECKPOINT_LIST[$i]}

    BASE_REMARK="llama-1B-deepseek-moe-5EP-qurater-sharedEP-clip-alpha-embed-Tok2-16tasks-Epoch${EPOCH}"
    MODEL_BASE_PATH="_checkpoints/llama-1B-deepseek-moe-5EP-qurater-sharedEP-clip-alpha-embed-Tok2-16tasks/checkpoint-${CKPT}"

    for TASK in "${TASK_LIST[@]}"; do
        case "$TASK" in
            "forward") DATA_PATH="data/evaluate/forward_reaction_prediction.json" ;;
            "reagent") DATA_PATH="data/evaluate/reagent_prediction.json" ;;
            "retrosynthesis") DATA_PATH="data/evaluate/retrosynthesis.json" ;;
            "homolumo") DATA_PATH="data/evaluate/property_prediction.json" ;;
            "molcap") DATA_PATH="data/evaluate/molcap_test.json" ;;
            "solvent") DATA_PATH="data/evaluate/solvent_pred.json" ;;
            "catalyst") DATA_PATH="data/evaluate/catalyst_pred.json" ;;
            "yield_BH") DATA_PATH="data/evaluate/yields_regression_BH.json" ;;
            "yield_SM") DATA_PATH="data/evaluate/yields_regression_SM.json" ;;
            "iupac2selfies") DATA_PATH="data/evaluate/iupac2selfies.json" ;;
            "molediting") DATA_PATH="data/evaluate/molecule_editing.json" ;;
            "textguidedmolgen") DATA_PATH="data/evaluate/text_guided_mol_generation.json" ;;
            "logp"|"weight"|"tpsa")
                DATA_PATH="data/evaluate/3d_moit_no_homolumo_filtered_test.json" ;;
            "experiment") DATA_PATH="data/evaluate/exp_procedure_pred.json" ;;
            *) echo "Warning: Unknown TASK: $TASK"; continue ;;
        esac

        REMARK="${BASE_REMARK}-${TASK}"
        if [[ "$TASK" == "yield_BH" || "$TASK" == "yield_SM" ]]; then
            MODEL_REMARK="${BASE_REMARK}-yields_regression"
        else
            MODEL_REMARK=$REMARK
        fi

        PPP_PATH="eval_result/save_all_tasks/${BASE_REMARK}/${TASK}-${TYPE}-${PROMPT}-answer.json"
        METRIC_PATH="eval_result/save_all_tasks_metric/${BASE_REMARK}/${TASK}-${TYPE}-${PROMPT}-metric.json"

        echo "--------------------------------------"
        echo "Epoch:       $EPOCH"
        echo "Checkpoint:  $CKPT"
        echo "Task:        $TASK"
        echo "Model path:  $MODEL_BASE_PATH"
        echo "Data path:   $DATA_PATH"
        echo "--------------------------------------"

        accelerate launch --main_process_port 29519 eval_engine.py \
            --model_type "$TYPE" \
            --task "$TASK" \
            --model_path "$MODEL_BASE_PATH" \
            --metric_path "$METRIC_PATH" \
            --language_backbone "$BACKBONE" \
            --prompt_version "$PROMPT" \
            --graph_tower "$GRAPH_TOWER" \
            --graph_path "$GRAPH_PATH" \
            --num_beams "$NUM_BEAMS" \
            --top_p "$TOP_P" \
            --temperature "$TEMPERATURE" \
            --data_path "$DATA_PATH" \
            --output_path "$PPP_PATH" \
            --batch_size "$BATCH_SIZE" \
            --dtype "$DTYPE" \
            --use_flash_atten True \
            --device "$DEVICE" \
            --add_selfies "$ADD_SELFIES" \
            --is_training "$IS_TRAINING" \
            --max_new_tokens "$MAX_NEW_TOKENS" \
            --repetition_penalty "$REPETITION_PENALTY"

        echo "Finish Task: $TASK"
        echo
    done
done

echo "========== All Evaluation Finished =========="
