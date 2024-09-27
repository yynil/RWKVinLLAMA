#!/bin/bash

# 默认值
MAX_SEQ_LENGTH=2048
BS=6
WARMUP=150
NUM_DEVICES=8
LR_INIT=1e-4
LR_FINAL=5e-4
DROPOUT=0.01
LOG_EVERY_N_STEPS=1000
OUTPUT_ALL_HIDDENS=“”
MODEL_ID="/data/rwkv/models/meta-llama/Meta-Llama-3.1-8B-Instruct/"

# 新增的默认路径前缀
BASE_DIR_TRAIN="/data/rwkv/data/ultrachat_llama3_1_pseudo_ds/"
OUTPUT_DIR="/data/rwkv/tmp/distill-en-zh_llama3_1_pseudo_ds"
CONFIG_FILE="configs/test_hybrid_full_logits_llamamlp.yaml"
CKPT_FILE=""

# 解析命名参数
while getopts ":m:M:b:w:v:n:i:f:d:l:t:a:o:c:k:h:A:" opt; do
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
    t) BASE_DIR_TRAIN="$OPTARG"
    ;;
    o) OUTPUT_DIR="$OPTARG"
    ;;
    c) CONFIG_FILE="$OPTARG"
    ;;
    k) CKPT_FILE="--ckpt_file $OPTARG"
    ;;
    h) OUTPUT_ALL_HIDDENS="--output_all_hiddens"
    ;;
    A) MAX_LENGTH="$OPTARG"
    ;;
    \?) echo "无效的选项 -$OPTARG" >&2
    exit 1
    ;;
  esac
done


echo "参数设置："
echo "MAX_SEQ_LENGTH=$MAX_SEQ_LENGTH, BS=$BS, WARMUP=$WARMUP, VAL_CHECK_INTERVAL=$VAL_CHECK_INTERVAL"
echo "NUM_DEVICES=$NUM_DEVICES, LR_INIT=$LR_INIT, LR_FINAL=$LR_FINAL, DROPOUT=$DROPOUT"
echo "LOG_EVERY_N_STEPS=$LOG_EVERY_N_STEPS"
echo "BASE_DIR_TRAIN=$BASE_DIR_TRAIN"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "CONFIG_FILE=$CONFIG_FILE"
echo "CKPT_FILE=$CKPT_FILE"
echo "OUTPUT_ALL_HIDDENS=$OUTPUT_ALL_HIDDENS"
# echo "启动教师服务器"
# python server/teacher_server_nccl_gather.py --model_id $MODEL_ID --batch $BS --length $MAX_SEQ_LENGTH --size 4 --device_id 0 --nccl_id_file nccl.txt_0 &
# python server/teacher_server_nccl_gather.py --model_id $MODEL_ID --batch $BS --length $MAX_SEQ_LENGTH --size 4 --device_id 7 --nccl_id_file nccl.txt_1 &

echo "启动学生服务器"
WKV=fla CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,0,7 python train_scripts/train_hybrid.py \
    --num_devices $NUM_DEVICES \
    --grad_cp 1 \
    --max_seq_length $MAX_SEQ_LENGTH \
    --output_dir $OUTPUT_DIR \
    --config_file $CONFIG_FILE \
    --lr_init $LR_INIT \
    --micro_bsz $BS \
    --preprocessed_data $BASE_DIR_TRAIN \
    --dropout $DROPOUT \
    --strategy deepspeed_stage_3_offload \
    --log_every_n_steps $LOG_EVERY_N_STEPS \
    --lr_final $LR_FINAL \
    --accumulate_grad_batches 4 \
    --warmup_steps $WARMUP \
    $CKPT_FILE \
    --wandb hybrid_trainer_llama3_1_pseudo_ds_${MAX_SEQ_LENGTH}