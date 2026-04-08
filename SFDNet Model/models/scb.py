"""
Spatial Coordinate-aware Branch (SCB)
论文公式 (6)-(9)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StripPooling(nn.Module):
    """
    条带池化: 分别沿水平和垂直方向做全局平均池化
    捕获连续的位置坐标信息
    """

    def __init__(self, channels):
        super(StripPooling, self).__init__()
        # 水平条带: 沿H维度池化 -> [B, C, 1, W]
        self.pool_h = nn.AdaptiveAvgPool2d((1, None))
        # 垂直条带: 沿W维度池化 -> [B, C, H, 1]
        self.pool_w = nn.AdaptiveAvgPool2d((None, 1))

        self.conv_h = nn.Conv2d(channels, channels, (1, 3), padding=(0, 1))
        self.conv_w = nn.Conv2d(channels, channels, (3, 1), padding=(1, 0))
        self.norm = nn.InstanceNorm2d(channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        """
        Args:
            x: shape [B, C, H, W]
        Returns:
            coord_map: 坐标编码图 shape [B, C, H, W]
        """
        B, C, H, W = x.shape

        # 公式 (6): 水平条带池化
        h_stripe = self.pool_h(x)                     # [B, C, 1, W]
        h_stripe = self.act(self.conv_h(h_stripe))     # [B, C, 1, W]
        h_stripe = h_stripe.expand(-1, -1, H, -1)     # 广播回 [B, C, H, W]

        # 公式 (7): 垂直条带池化
        w_stripe = self.pool_w(x)                      # [B, C, H, 1]
        w_stripe = self.act(self.conv_w(w_stripe))     # [B, C, H, 1]
        w_stripe = w_stripe.expand(-1, -1, -1, W)     # 广播回 [B, C, H, W]

        # 公式 (8): 融合水平和垂直坐标编码
        coord_map = self.norm(h_stripe + w_stripe)

        return coord_map


class SCB(nn.Module):
    """
    Spatial Coordinate-aware Branch (SCB)
    论文 Section 3.3 的完整实现

    核心思想: 使用条带池化编码空间位置坐标信息,
    补偿频域变换丢失的局部结构信息
    """

    def __init__(self, channels):
        super(SCB, self).__init__()
        self.strip_pool = StripPooling(channels)

        # 深度可分离卷积提取局部纹理
        self.dw_conv = nn.Conv2d(
            channels, channels, 3, 1, 1, groups=channels
        )
        self.pw_conv = nn.Conv2d(channels, channels, 1, 1, 0)
        self.norm = nn.InstanceNorm2d(channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.sigmoid = nn.Sigmoid()

        # 最终输出卷积
        self.out_conv = nn.Conv2d(channels, channels, 3, 1, 1)

    def forward(self, x):
        """
        Args:
            x: 空域特征 shape [B, C, H, W]
        Returns:
            x_spatial: 空间坐标增强后的特征 shape [B, C, H, W]
        """
        # Step 1: 获取坐标编码图
        coord_map = self.strip_pool(x)

        # Step 2: 深度可分离卷积提取局部纹理
        local_feat = self.act(self.norm(self.dw_conv(x)))
        local_feat = self.pw_conv(local_feat)

        # Step 3: 坐标门控 (公式 9)
        # gate = sigmoid(coord_map), 用空间位置信息调制局部纹理
        gate = self.sigmoid(coord_map)
        x_spatial = local_feat * gate

        # Step 4: 残差连接 + 输出
        x_spatial = self.out_conv(x_spatial) + x

        return x_spatial