"""
SFDNet Test Script
使用方法:
    python test.py \
        --config configs/train_lol.yaml \
        --checkpoint pretrained/sfdnet_lol_best.pth \
        --input_dir ./datasets/LOLv1/test/low/ \
        --output_dir ./results/lol_v1/ \
        --gpu 0
"""

import os
import torch
from tqdm import tqdm
from glob import glob

from options.test_options import parse_test_options
from models import SFDNet
from utils.metrics import MetricCalculator
from utils.img_utils import load_image, save_image, tensor_to_numpy


def main():
    config = parse_test_options()
    os.environ['CUDA_VISIBLE_DEVICES'] = config['gpu']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ========== 模型加载 ==========
    model = SFDNet(
        in_channels=config['model']['in_channels'],
        out_channels=config['model']['out_channels'],
        base_channels=config['model']['base_channels'],
    ).to(device)

    checkpoint = torch.load(
        config['checkpoint'], map_location=device
    )
    model.load_state_dict(checkpoint['model'])
    model.eval()

    print(f"Model loaded from: {config['checkpoint']}")
    print(f"Best PSNR: {checkpoint.get('best_psnr', 'N/A')}")

    # ========== 准备输入 ==========
    input_dir = config['input_dir']
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    image_paths = sorted(
        glob(os.path.join(input_dir, '*.png')) +
        glob(os.path.join(input_dir, '*.jpg'))
    )

    print(f"Found {len(image_paths)} images in {input_dir}")

    # ========== 指标计算器 ==========
    metric_calc = MetricCalculator(device=device)
    psnr_list, ssim_list, lpips_list = [], [], []

    # ========== 推理 ==========
    with torch.no_grad():
        for img_path in tqdm(image_paths, desc='Testing'):
            fname = os.path.basename(img_path)

            # 加载图像
            low = load_image(img_path).to(device)

            # 推理
            pred = model(low)
            pred = pred.clamp(0, 1)

            # 保存结果
            save_image(pred, os.path.join(output_dir, fname))

            # 如果有 GT, 计算指标
            gt_path = img_path.replace('low', 'high').replace(
                'input', 'GT'
            )
            if os.path.exists(gt_path):
                gt = load_image(gt_path).to(device)

                pred_np = tensor_to_numpy(pred)
                gt_np = tensor_to_numpy(gt)

                p = metric_calc.calculate_psnr(pred_np, gt_np)
                s = metric_calc.calculate_ssim(pred_np, gt_np)
                l = metric_calc.calculate_lpips(pred, gt)

                psnr_list.append(p)
                ssim_list.append(s)
                lpips_list.append(l)

                print(
                    f"  {fname}: PSNR={p:.2f}, "
                    f"SSIM={s:.4f}, LPIPS={l:.4f}"
                )

    # ========== 打印平均结果 ==========
    if psnr_list:
        print("\n" + "=" * 50)
        print(f"Average PSNR:  {sum(psnr_list)/len(psnr_list):.2f}")
        print(f"Average SSIM:  {sum(ssim_list)/len(ssim_list):.4f}")
        print(f"Average LPIPS: {sum(lpips_list)/len(lpips_list):.4f}")
        print("=" * 50)

    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()