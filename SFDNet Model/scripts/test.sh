#!/bin/bash
# SFDNet 测试脚本

echo "============================================"
echo "  SFDNet Testing - LOL-v1 Dataset"
echo "============================================"

CUDA_VISIBLE_DEVICES=0 python test.py \
    --config configs/train_lol.yaml \
    --checkpoint pretrained/sfdnet_lol_best.pth \
    --input_dir ./datasets/LOLv1/test/low/ \
    --output_dir ./results/lol_v1/ \
    --gpu 0

echo "Testing completed!"