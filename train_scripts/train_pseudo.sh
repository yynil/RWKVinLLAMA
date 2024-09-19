#!/bin/bash

# 默认值
BS=5
WARMUP=1000
VAL_CHECK_INTERVAL=5000
NUM_DEVICES=6
LR_INIT=1e-4
LR_FINAL=5e-4
DROPOUT=0.01
LOG_EVERY_N_STEPS=200
OUTPUT_ALL_HIDDENS=“”
MAX_LENGTH=2048

# 新增的默认路径前缀
INPUT_IDS_FILE="/data/rwkv/data/pseudo_labels/input_ids.pt"
LABELS_FILE="/data/rwkv/data/pseudo_labels/labels.pt"
OUTPUT_PREFIX="/data/rwkv/tmp/distill-en-zh-pseudo_labels"
CONFIG_FILE="/home/rwkv/github/RWKVinLLAMA/configs/test_hybrid_full_logits_llamamlp.yaml"
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
    t) INPUT_IDS_FILE="$OPTARG"
    ;;
    a) LABELS_FILE="$OPTARG"
    ;;
    o) OUTPUT_PREFIX="$OPTARG"
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



OUTPUT_DIR="${OUTPUT_PREFIX}"

echo "参数设置："
echo "MIN=$MIN, MAX=$MAX, BS=$BS, WARMUP=$WARMUP, VAL_CHECK_INTERVAL=$VAL_CHECK_INTERVAL"
echo "NUM_DEVICES=$NUM_DEVICES, LR_INIT=$LR_INIT, LR_FINAL=$LR_FINAL, DROPOUT=$DROPOUT"
echo "LOG_EVERY_N_STEPS=$LOG_EVERY_N_STEPS"
echo "BASE_DIR_TRAIN=$BASE_DIR_TRAIN"
echo "BASE_DIR_VAL=$BASE_DIR_VAL"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "CONFIG_FILE=$CONFIG_FILE"
echo "CKPT_FILE=$CKPT_FILE"
echo "OUTPUT_ALL_HIDDENS=$OUTPUT_ALL_HIDDENS"
# echo "启动教师服务器"
python server/teacher_server_nccl_gather.py --model_id /data/rwkv/models/meta-llama/Meta-Llama-3.1-8B-Instruct/ --batch $BS --length $MAX_LENGTH --size 4 --device_id 0 --nccl_id_file nccl.txt_0 &
python server/teacher_server_nccl_gather.py --model_id /data/rwkv/models/meta-llama/Meta-Llama-3.1-8B-Instruct/ --batch $BS --length $MAX_LENGTH --size 4 --device_id 7 --nccl_id_file nccl.txt_1 &

echo "启动学生服务器"
WKV=fla CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,0,7 python train_scripts/train_hybrid.py \
    --num_devices $NUM_DEVICES \
    --grad_cp 1 \
    --max_seq_length $MAX_LENGTH \
    --output_dir $OUTPUT_DIR \
    --config_file $CONFIG_FILE \
    --lr_init $LR_INIT \
    --micro_bsz $BS \
    --input_ids_file $INPUT_IDS_FILE \
    --labels_file $LABELS_FILE \
    --dropout $DROPOUT \
    --strategy deepspeed_stage_3_offload \
    --log_every_n_steps $LOG_EVERY_N_STEPS \
    --lr_final $LR_FINAL \
    --warmup_steps $WARMUP \
    $CKPT_FILE \
    --val_check_interval $VAL_CHECK_INTERVAL \
    --wandb hybrid_pseudo_trainer_ULTRACHAT_ULTRAFEEDBACK \
    --max_epochs 2