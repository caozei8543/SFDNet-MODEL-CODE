"""
评测指标: PSNR, SSIM, LPIPS
"""

import torch
import numpy as np
from skimage.metrics import (
    peak_signal_noise_ratio as psnr,
    structural_similarity as ssim,
)
import lpips


class MetricCalculator:
    """统一的指标计算器"""

    def __init__(self, device='cuda'):
        self.device = device
        self.lpips_fn = lpips.LPIPS(net='alex').to(device)
        self.lpips_fn.eval()

    def calculate_psnr(self, pred, target):
        """
        Args:
            pred: numpy array [H, W, 3], range [0, 1]
            target: numpy array [H, W, 3], range [0, 1]
        """
        return psnr(target, pred, data_range=1.0)

    def calculate_ssim(self, pred, target):
        """
        Args:
            pred: numpy array [H, W, 3], range [0, 1]
            target: numpy array [H, W, 3], range [0, 1]
        """
        return ssim(
            target, pred, multichannel=True,
            channel_axis=2, data_range=1.0
        )

    @torch.no_grad()
    def calculate_lpips(self, pred, target):
        """
        Args:
            pred: torch tensor [1, 3, H, W], range [0, 1]
            target: torch tensor [1, 3, H, W], range [0, 1]
        """
        pred = pred.to(self.device) * 2 - 1
        target = target.to(self.device) * 2 - 1
        return self.lpips_fn(pred, target).item()

    def compute_all(self, pred_np, target_np, pred_tensor, target_tensor):
        """一次性计算所有指标"""
        p = self.calculate_psnr(pred_np, target_np)
        s = self.calculate_ssim(pred_np, target_np)
        l = self.calculate_lpips(pred_tensor, target_tensor)
        return {'PSNR': p, 'SSIM': s, 'LPIPS': l}