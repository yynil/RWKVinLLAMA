#!/bin/bash

TRAINING_LAYER=1
NUM_DEVICES=8
MAX_SEQ_LENGTH=2048
MICRO_BSZ=2
WARMUP=1000
LR_INIT=2e-4
LR_FINAL=2e-5
DROPOUT=0.1
BASE_DIR_TRAIN="/data/rwkv/data/ultrachat_llama3_1_pseudo_ds"
OUTPUT_DIR=“/data/rwkv/tmp/distill-en-zh_llama3_1_pseudo_ds_all_kl_div_stepwise”
ACCUMULATE_GRAD_BATCHES=4
MAX_EPOCHS=2
LOG_EVERY_N_STEPS=500
while getopts ":T:l:n:m:M:b:w:v:i:f:d:t:o:c:k:h:A:" opt; do
case $opt in
    T) TRAINING_LAYER="$OPTARG"
    ;;
    n) NUM_DEVICES="$OPTARG"
    ;;
    M) MAX_SEQ_LENGTH="$OPTARG"
    ;;
    b) MICRO_BSZ="$OPTARG"
    ;;
    w) WARMUP="$OPTARG"
    ;;
    i) LR_INIT="$OPTARG"
    ;;
    f) LR_FINAL="$OPTARG"
    ;;
    d) DROPOUT="$OPTARG"
    ;;
    l) LOG_EVERY_N_STEPS="$OPTARG"
    ;;
    t) STRATEGY="$OPTARG"
    ;;
    o) OUTPUT_DIR="$OPTARG"
    ;;
    k) CKPT_FILE="--ckpt_file $OPTARG"
    ;;
    h) MAX_EPOCHS="$OPTARG"
    ;;
    A) ACCUMULATE_GRAD_BATCHES="$OPTARG"
    ;;
    
    \?) echo "无效的选项 -$OPTARG" >&2
    exit 1
    ;;
  esac
done

echo "参数设置："
echo "NUM_DEVICES: $NUM_DEVICES"
echo "MIN_SEQ_LENGTH: $MIN_SEQ_LENGTH"
echo "MAX_SEQ_LENGTH: $MAX_SEQ_LENGTH"
echo "MICRO_BSZ: $MICRO_BSZ"
echo "WARMUP: $WARMUP"
echo "LR_INIT: $LR_INIT"
echo "LR_FINAL: $LR_FINAL"
echo "DROPOUT: $DROPOUT"
echo "LOG_EVERY_N_STEPS: $LOG_EVERY_N_STEPS"
echo "STRATEGY: $STRATEGY"
CONFIG_FILE="configs/step_wise/test_hybrid_${TRAINING_LAYER}_layer_llamamlp.yaml"
OUTPUT_DIR="/data/rwkv/tmp/distill-en-zh_llama3_1_pseudo_ds_all_kl_div_stepwise_layer_${TRAINING_LAYER}"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "MAX_EPOCHS: $MAX_EPOCHS"
echo "ACCUMULATE_GRAD_BATCHES: $ACCUMULATE_GRAD_BATCHES"
echo "CKPT_FILE: $CKPT_FILE"


WKV=fla CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,0,7 python train_scripts/train_hybrid.py \
    --num_devices $NUM_DEVICES \
    --grad_cp 1 \
    --max_seq_length $MAX_SEQ_LENGTH \
    --output_dir $OUTPUT_DIR \
    --config_file $CONFIG_FILE \
    --lr_init $LR_INIT \
    --micro_bsz $MICRO_BSZ \
    --preprocessed_data $BASE_DIR_TRAIN \
    --dropout $DROPOUT \
    --strategy deepspeed_stage_3_offload \
    --log_every_n_steps $LOG_EVERY_N_STEPS \
    --lr_final $LR_FINAL \
    --accumulate_grad_batches $ACCUMULATE_GRAD_BATCHES \
    --warmup_steps $WARMUP \
    --max_epochs $MAX_EPOCHS \
    $CKPT_FILE \
    --wandb hybrid_trainer_llama3_1_pseudo_ds_${MAX_SEQ_LENGTH}