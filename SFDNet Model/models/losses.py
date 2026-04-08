"""
Multi-scale Loss Functions
论文 Section 3.5 中描述的联合损失函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1 的平滑变体, 比 MSE 更鲁棒)"""

    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, pred, target):
        diff = pred - target
        loss = torch.mean(torch.sqrt(diff * diff + self.eps))
        return loss


class EdgeLoss(nn.Module):
    """边缘感知损失: 用 Laplacian 算子提取边缘"""

    def __init__(self):
        super(EdgeLoss, self).__init__()
        # Laplacian 核
        kernel = torch.tensor(
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
            dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0)
        self.kernel = kernel.repeat(3, 1, 1, 1)
        self.char_loss = CharbonnierLoss()

    def forward(self, pred, target):
        self.kernel = self.kernel.to(pred.device)
        pred_edge = F.conv2d(pred, self.kernel, padding=1, groups=3)
        target_edge = F.conv2d(target, self.kernel, padding=1, groups=3)
        return self.char_loss(pred_edge, target_edge)


class FrequencyLoss(nn.Module):
    """频域损失: 约束输出和GT在频域的一致性"""

    def __init__(self):
        super(FrequencyLoss, self).__init__()

    def forward(self, pred, target):
        pred_fft = torch.fft.rfft2(pred, norm='ortho')
        target_fft = torch.fft.rfft2(target, norm='ortho')

        # 幅值损失
        amp_loss = F.l1_loss(
            torch.abs(pred_fft), torch.abs(target_fft)
        )
        # 相位损失
        pha_loss = F.l1_loss(
            torch.angle(pred_fft), torch.angle(target_fft)
        )

        return amp_loss + pha_loss


class PerceptualLoss(nn.Module):
    """基于 LPIPS 的感知损失"""

    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.lpips_fn = lpips.LPIPS(net='vgg', verbose=False)
        # 冻结 VGG 参数
        for param in self.lpips_fn.parameters():
            param.requires_grad = False

    def forward(self, pred, target):
        # LPIPS 要求输入范围 [-1, 1]
        pred_scaled = pred * 2 - 1
        target_scaled = target * 2 - 1
        return self.lpips_fn(pred_scaled, target_scaled).mean()


class TotalLoss(nn.Module):
    """
    总损失函数:
    L_total = λ1 * L_char + λ2 * L_edge + λ3 * L_freq + λ4 * L_percep
    """

    def __init__(self, lambda_char=1.0, lambda_edge=0.5,
                 lambda_freq=0.5, lambda_percep=0.1):
        super(TotalLoss, self).__init__()
        self.char_loss = CharbonnierLoss()
        self.edge_loss = EdgeLoss()
        self.freq_loss = FrequencyLoss()
        self.percep_loss = PerceptualLoss()

        self.lambda_char = lambda_char
        self.lambda_edge = lambda_edge
        self.lambda_freq = lambda_freq
        self.lambda_percep = lambda_percep

    def forward(self, pred, target):
        l_char = self.char_loss(pred, target)
        l_edge = self.edge_loss(pred, target)
        l_freq = self.freq_loss(pred, target)
        l_percep = self.percep_loss(pred, target)

        total = (self.lambda_char * l_char +
                 self.lambda_edge * l_edge +
                 self.lambda_freq * l_freq +
                 self.lambda_percep * l_percep)

        loss_dict = {
            'total': total,
            'charbonnier': l_char,
            'edge': l_edge,
            'frequency': l_freq,
            'perceptual': l_percep,
        }

        return total, loss_dict