#!/bin/bash

model="s1.1-3B"
dataset="math500"

## fullthinking:
if [ ! -f "outputs_exp/${model}_${dataset}_fullthinking.jsonl" ];then
    for ((i=0; i<8; i++)); do
        CUDA_VISIBLE_DEVICES=$i nohup \
        python main.py --partial $i --model $model --cal_set $dataset --task fullthinking\
        > log_out/log_${i}_${model}_fullthinking.out 2>&1 &
    done

    wait
fi

## remove duplicate tokens:
python remove_dup.py --dataset $dataset --model $model

## get min tokens:
for ((i=0; i<8; i++)); do
    CUDA_VISIBLE_DEVICES=$i nohup \
    python main.py --partial $i --model $model --cal_set $dataset --task min_token\
    > log_out/log_${i}_${model}_min_token.out 2>&1 &
done

wait