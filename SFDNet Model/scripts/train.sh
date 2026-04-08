#!/bin/bash
# SFDNet 训练脚本

echo "============================================"
echo "  SFDNet Training - LOL-v1 Dataset"
echo "============================================"

CUDA_VISIBLE_DEVICES=0 python train.py \
    --config configs/train_lol.yaml \
    --gpu 0

echo "Training completed!"