#!/bin/bash

NUM_TRAINERS=3

OMP_NUM_THREADS=1 python3 -m torch.distributed.launch --standalone --nnodes=1 --nproc-per-node=$NUM_TRAINERS tests/distributed_test.py
# OMP_NUM_THREADS=1 torchrun --standalone --nnodes=1 --nproc-per-node=$NUM_TRAINERS tests/distributed_test.py