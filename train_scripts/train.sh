#!/bin/bash

# 默认值
MIN=1
MAX=256
BS=32
WARMUP=600
VAL_CHECK_INTERVAL=5000
NUM_DEVICES=6
LR_INIT=1e-4
LR_FINAL=5e-4
DROPOUT=0.01
LOG_EVERY_N_STEPS=20000

# 解析命名参数
while getopts ":m:M:b:w:v:n:i:f:d:l:" opt; do
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
    \?) echo "无效的选项 -$OPTARG" >&2
    exit 1
    ;;
  esac
done

BASE_DIR_TRAIN=/home/rwkv/preprocessed_${MIN}_${MAX}
BASE_DIR_VAL=/home/rwkv/preprocessed_val_${MIN}_${MAX}
OUPUT_DIR=/data/rwkv/tmp/distill-en-zh-stage-2_${MIN}_${MAX}
CONFIG_FILE=configs/test_hybrid_full_logits_stage_2.yaml
CKPT_FILE=/data/rwkv/tmp/distill-c4-en-zh/pytorch_model.1400m.bin

echo "参数设置："
echo "MIN=$MIN, MAX=$MAX, BS=$BS, WARMUP=$WARMUP, VAL_CHECK_INTERVAL=$VAL_CHECK_INTERVAL"
echo "NUM_DEVICES=$NUM_DEVICES, LR_INIT=$LR_INIT, LR_FINAL=$LR_FINAL, DROPOUT=$DROPOUT"
echo "LOG_EVERY_N_STEPS=$LOG_EVERY_N_STEPS"
echo "BASE_DIR_TRAIN=$BASE_DIR_TRAIN"
echo "BASE_DIR_VAL=$BASE_DIR_VAL"
echo "OUPUT_DIR=$OUPUT_DIR"
echo "CONFIG_FILE=$CONFIG_FILE"
echo "CKPT_FILE=$CKPT_FILE"

# echo "启动教师服务器"
python server/teacher_server_nccl_gather.py --model_id /data/rwkv/models/meta-llama/Meta-Llama-3.1-8B-Instruct/ --batch $BS --length $MAX --size 4 --device_id 0 --nccl_id_file nccl.txt_0 &
python server/teacher_server_nccl_gather.py --model_id /data/rwkv/models/meta-llama/Meta-Llama-3.1-8B-Instruct/ --batch $BS --length $MAX --size 4 --device_id 7 --nccl_id_file nccl.txt_1 &

echo "启动学生服务器"
WKV=fla CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,0,7 python train_scripts/train_hybrid.py \
    --num_devices $NUM_DEVICES \
    --grad_cp 1 \
    --max_seq_length $MAX \
    --output_dir $OUPUT_DIR \
    --config_file $CONFIG_FILE \
    --lr_init $LR_INIT \
    --micro_bsz $BS \
    --preprocessed_data $BASE_DIR_TRAIN $BASE_DIR_VAL \
    --dropout $DROPOUT \
    --strategy deepspeed_stage_3_offload \
    --log_every_n_steps $LOG_EVERY_N_STEPS \
    --lr_final $LR_FINAL \
    --warmup_steps $WARMUP \
    --ckpt_file $CKPT_FILE \
    --val_check_interval $VAL_CHECK_INTERVAL \
    --wandb hybrid_trainer_${MIN}_${MAX}