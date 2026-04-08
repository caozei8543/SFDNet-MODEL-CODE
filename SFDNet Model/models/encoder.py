"""
Spatial Domain Encoder
U-Net 风格的多尺度编码器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """基础卷积块: Conv -> InstanceNorm -> LeakyReLU"""

    def __init__(self, in_ch, out_ch):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    """
    四级下采样编码器
    输出四个尺度的特征图用于跳跃连接
    """

    def __init__(self, in_channels=3, base_channels=32):
        super(Encoder, self).__init__()
        ch = base_channels

        self.enc1 = ConvBlock(in_channels, ch)         # H x W
        self.enc2 = ConvBlock(ch, ch * 2)               # H/2 x W/2
        self.enc3 = ConvBlock(ch * 2, ch * 4)           # H/4 x W/4
        self.enc4 = ConvBlock(ch * 4, ch * 8)           # H/8 x W/8

        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W]
        Returns:
            features: list of [e1, e2, e3, e4]
        """
        e1 = self.enc1(x)              # [B, 32, H, W]
        e2 = self.enc2(self.pool(e1))  # [B, 64, H/2, W/2]
        e3 = self.enc3(self.pool(e2))  # [B, 128, H/4, W/4]
        e4 = self.enc4(self.pool(e3))  # [B, 256, H/8, W/8]

        return [e1, e2, e3, e4]