#!/bin/bash

model="s1.1-14B"
dataset="gsm8k"

# ## fullthinking:
# if [ ! -f "outputs_exp/${model}_${dataset}_fullthinking.jsonl" ];then
#     for ((i=0; i<8; i++)); do
#         CUDA_VISIBLE_DEVICES=$i nohup \
#         python main.py --partial $i --model $model --cal_set $dataset --task fullthinking\
#         > log_out/log_${i}_${model}_fullthinking.out 2>&1 &
#     done

#     wait
# fi

# ## remove duplicate tokens:
# python remove_dup.py --dataset $dataset --model $model

## get fulltrace:
for ((i=0; i<2; i++)); do
    CUDA_VISIBLE_DEVICES=$((i+4)),$((i+6)) nohup \
    python main.py --partial $i --model $model --dataset $dataset --task fulltrace\
    > log_out/log_${i}_${model}_fulltrace.out 2>&1 &
done

wait

# model="s1.1-7B"
dataset="math500"

for ((i=0; i<2; i++)); do
    CUDA_VISIBLE_DEVICES=$((i+4)),$((i+6)) nohup \
    python main.py --partial $i --model $model --dataset $dataset --task fulltrace\
    > log_out/log_${i}_${model}_fulltrace.out 2>&1 &
done

wait

# model="s1.1-7B"
# dataset="gsm8k"

# for ((i=0; i<8; i++)); do
#     CUDA_VISIBLE_DEVICES=$i nohup \
#     python main.py --partial $i --model $model --dataset $dataset --task fulltrace\
#     > log_out/log_${i}_${model}_fulltrace.out 2>&1 &
# done

# wait

## get right token range:
for ((i=0; i<5; i++)); do
    nohup \
    python fix_tokens_range.py --partial $i --model $model --dataset $dataset \
    > log_out/log_${i}_${model}_fix_min_token.out 2>&1 &
done

wait

# model="s1.1-7B"
dataset="gsm8k"
for ((i=0; i<5; i++)); do
    nohup \
    python fix_tokens_range.py --partial $i --model $model --dataset $dataset \
    > log_out/log_${i}_${model}_fix_min_token.out 2>&1 &
done

wait

# ## get nudup answers:
# for ((i=0; i<8; i++)); do
#     CUDA_VISIBLE_DEVICES=$i nohup \
#     python main.py --partial $i --model $model --cal_set $dataset --task nodup_ans\
#     > log_out/log_${i}_${model}_nudup_ans.out 2>&1 &
# done

# wait