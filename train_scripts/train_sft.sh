#!/bin/bash

# 默认值
MAX=256
MIN=0
BS=32
WARMUP=1700
VAL_CHECK_INTERVAL=5000
NUM_DEVICES=8
LR_INIT=1e-4
LR_FINAL=5e-4
DROPOUT=0.01
LOG_EVERY_N_STEPS=10000
OUTPUT_ALL_HIDDENS=“”

# 新增的默认路径前缀
TRAIN_PREFIX="/data/rwkv/data/Infinity-Instruct/ds/7M/length_"
OUTPUT_PREFIX="/data/rwkv/tmp/distill-en-zh-sft_"
CONFIG_FILE="configs/test_hybrid_full_sft.yaml"
CKPT_FILE="/data/rwkv/tmp/distill-en-zh-stage-2_513_768/ckpt_513_768.pth"

# 解析命名参数
while getopts ":M:m:b:w:v:n:i:f:d:l:t:a:o:c:k:h:" opt; do
  case $opt in
    m) MIN="$OPTARG"
    ;;
    M) MAX="$OPTARG"
    ;;
    b) BS="$OPTARG"
    ;;
    w) WARMUP="$OPTARG"
    ;;
    v) VAL_CHECK_INTERVAL="$OPTARG"
    ;;
    n) NUM_DEVICES="$OPTARG"
    ;;
    i) LR_INIT="$OPTARG"
    ;;
    f) LR_FINAL="$OPTARG"
    ;;
    d) DROPOUT="$OPTARG"
    ;;
    l) LOG_EVERY_N_STEPS="$OPTARG"
    ;;
    t) TRAIN_PREFIX="$OPTARG"
    ;;
    a) VAL_PREFIX="$OPTARG"
    ;;
    o) OUTPUT_PREFIX="$OPTARG"
    ;;
    c) CONFIG_FILE="$OPTARG"
    ;;
    k) CKPT_FILE="$OPTARG"
    ;;
    h) OUTPUT_ALL_HIDDENS="--output_all_hiddens"
    ;;
    \?) echo "无效的选项 -$OPTARG" >&2
    exit 1
    ;;
  esac
done

# 构建完整的路径
BASE_DIR_TRAIN="${TRAIN_PREFIX}${MIN}_${MAX}"
OUTPUT_DIR="${OUTPUT_PREFIX}${MIN}_${MAX}"

echo "参数设置："
echo "MAX=$MAX, BS=$BS, WARMUP=$WARMUP, VAL_CHECK_INTERVAL=$VAL_CHECK_INTERVAL MIN=$MIN"
echo "NUM_DEVICES=$NUM_DEVICES, LR_INIT=$LR_INIT, LR_FINAL=$LR_FINAL, DROPOUT=$DROPOUT"
echo "LOG_EVERY_N_STEPS=$LOG_EVERY_N_STEPS"
echo "BASE_DIR_TRAIN=$BASE_DIR_TRAIN"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "CONFIG_FILE=$CONFIG_FILE"
echo "CKPT_FILE=$CKPT_FILE"
echo "OUTPUT_ALL_HIDDENS=$OUTPUT_ALL_HIDDENS"

echo "开始训练..."
WKV=fla CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,0,7 python train_scripts/train_hybrid.py \
    --num_devices $NUM_DEVICES \
    --grad_cp 1 \
    --max_seq_length $MAX \
    --output_dir $OUTPUT_DIR \
    --config_file $CONFIG_FILE \
    --lr_init $LR_INIT \
    --micro_bsz $BS \
    --preprocessed_data $BASE_DIR_TRAIN \
    --dropout $DROPOUT \
    --strategy deepspeed_stage_3_offload \
    --log_every_n_steps $LOG_EVERY_N_STEPS \
    --lr_final $LR_FINAL \
    --warmup_steps $WARMUP \
    --ckpt_file $CKPT_FILE \
    --val_check_interval $VAL_CHECK_INTERVAL \
    --wandb hybrid_trainer_sft_${MAX}