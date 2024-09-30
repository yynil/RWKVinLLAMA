#!/bin/bash
NUM_DEVICES=8
MODEL_ID="/data/rwkv/models/meta-llama/Meta-Llama-3.1-8B-Instruct/"
BASE_PORT=8001
MAX_MODEL_LEN=16384
while getopts ":n:m:p:l:" opt; do
    case $opt in
        n) NUM_DEVICES="$OPTARG"
        ;;
        m) MODEL_ID="$OPTARG"
        ;;
        p) BASE_PORT="$OPTARG"
        ;;
        l) MAX_MODEL_LEN="$OPTARG"
        ;;  
    esac
done

for i in $(seq 0 $((NUM_DEVICES-1))); do
    port=$((BASE_PORT + i))
    CUDA_VISIBLE_DEVICES=$i vllm serve ${MODEL_ID} --max-model-len ${MAX_MODEL_LEN} --port ${port} --enable-chunked-prefill --pipeline-parallel-size 1 --enable_prefix_caching &
done