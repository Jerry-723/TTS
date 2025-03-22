#!/bin/bash

for ((i=0; i<8; i++)); do
    CUDA_VISIBLE_DEVICES=$i nohup \
    python ../main.py --partial $i \
    > log_out/log_$i.out 2>&1 &
done

wait