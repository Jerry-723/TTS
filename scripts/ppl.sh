#!/bin/bash

# model="s1.1-3B"

# CUDA_VISIBLE_DEVICES=0,1 nohup python get_ppl.py --model $model --dataset math500 > log_out/ppl_1.out&

# CUDA_VISIBLE_DEVICES=2,3 nohup python get_ppl.py --model $model --dataset gsm8k > log_out/ppl_2.out&

# model="s1.1-7B"

# CUDA_VISIBLE_DEVICES=4,5 nohup python get_ppl.py --model $model --dataset math500 > log_out/ppl_3.out&

# CUDA_VISIBLE_DEVICES=6,7 nohup python get_ppl.py --model $model --dataset gsm8k > log_out/ppl_4.out&

# wait

model="s1.1-14B"

CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python get_ppl.py --model $model --dataset math500 > log_out/ppl_5.out&

CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python get_ppl.py --model $model --dataset gsm8k > log_out/ppl_6.out&

wait