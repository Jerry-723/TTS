#!/bin/bash

for ((i=0; i<8; i++)); do
    CUDA_VISIBLE_DEVICES=$i nohup \
    python main.py --partial $i --model s1.1-7B \
    > log_out/log_"$i"_7B.out 2>&1 &
done

wait

for ((i=0; i<8; i++)); do
    CUDA_VISIBLE_DEVICES=$i nohup \
    python main.py --partial $i --model s1.1-14B \
    > log_out/log_"$i"_14B.out 2>&1 &
done

wait