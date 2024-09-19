#!/bin/bash

data_prefix="$1"
data_chunks="$2"
model_id="$3"
output_prefix="$4"
start_of_device_id="$5"

for ((i=0; i<data_chunks; i++)); do
    device_id=$((start_of_device_id + i))
    python data/create_pseudo_labels.py --model_id "$model_id" --data_dir "${data_prefix}${i}/" --output_dir "${output_prefix}${i}/" --device "cuda:${device_id}" &
done

wait