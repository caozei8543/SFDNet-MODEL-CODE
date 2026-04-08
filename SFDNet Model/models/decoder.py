"""
Spatial Domain Decoder
U-Net 风格的多尺度解码器 (带跳跃连接)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UpConvBlock(nn.Module):
    """上采样 + 卷积块"""

    def __init__(self, in_ch, out_ch):
        super(UpConvBlock, self).__init__()
        self.up = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x, skip):
        """
        Args:
            x: 来自更深层的特征
            skip: 编码器的跳跃连接特征
        """
        x = self.up(x)

        # 处理尺寸不匹配
        if x.shape != skip.shape:
            x = F.interpolate(
                x, size=skip.shape[2:],
                mode='bilinear', align_corners=True
            )

        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class Decoder(nn.Module):
    """
    四级上采样解码器 (带跳跃连接)
    """

    def __init__(self, out_channels=3, base_channels=32):
        super(Decoder, self).__init__()
        ch = base_channels

        self.dec3 = UpConvBlock(ch * 8 + ch * 4, ch * 4)   # H/4
        self.dec2 = UpConvBlock(ch * 4 + ch * 2, ch * 2)   # H/2
        self.dec1 = UpConvBlock(ch * 2 + ch, ch)            # H

        self.final = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch, out_channels, 1, 1, 0),
            nn.Sigmoid(),  # 输出归一化到 [0, 1]
        )

    def forward(self, features):
        """
        Args:
            features: [e1, e2, e3, bottleneck]
        Returns:
            output: [B, 3, H, W]
        """
        e1, e2, e3, bottleneck = features

        d3 = self.dec3(bottleneck, e3)  # [B, 128, H/4, W/4]
        d2 = self.dec2(d3, e2)          # [B, 64,  H/2, W/2]
        d1 = self.dec1(d2, e1)          # [B, 32,  H,   W]

        output = self.final(d1)         # [B, 3,   H,   W]
        return output