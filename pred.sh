#!/bin/bash
ALPHA=$1

for ((i=0; i<8; i++)); do
    CUDA_VISIBLE_DEVICES=$i nohup \
    python main.py --partial $i --alpha $ALPHA\
    > log_out/log_"$ALPHA"_$i.out 2>&1 &
done

wait