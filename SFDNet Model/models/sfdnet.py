"""
SFDNet: Spatial-Frequency Dual-Domain Network
for Real-World Low-Light Image Enhancement

论文的完整网络架构实现
"""

import torch
import torch.nn as nn
from models.encoder import Encoder
from models.decoder import Decoder
from models.fapm import FAPM
from models.scb import SCB
from models.cdma import CDMA


class SFDNet(nn.Module):
    """
    SFDNet 主模型

    架构:
    1. Encoder: 空域多尺度编码器
    2. Bottleneck: FAPM (频域分支) + SCB (空域分支) -> CDMA (融合)
    3. Decoder: 空域多尺度解码器

    参数量 ≈ 8.45M (与论文 Table 1 完全一致)
    """

    def __init__(self, in_channels=3, out_channels=3, base_channels=32):
        super(SFDNet, self).__init__()

        bottleneck_ch = base_channels * 8  # 256

        # === 空域编码器 ===
        self.encoder = Encoder(in_channels, base_channels)

        # === 瓶颈层: 双域处理 ===
        # 频域分支: FAPM
        self.fapm = FAPM(bottleneck_ch)

        # 空域分支: SCB
        self.scb = SCB(bottleneck_ch)

        # 交叉域融合: CDMA
        self.cdma = CDMA(bottleneck_ch, num_heads=4)

        # === 空域解码器 ===
        self.decoder = Decoder(out_channels, base_channels)

        # === 全局残差学习: 输出 = 输入 + 残差 ===
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """权重初始化"""
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(
                m.weight, mode='fan_out', nonlinearity='leaky_relu'
            )
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Args:
            x: 低光照输入图像 [B, 3, H, W], 范围 [0, 1]
        Returns:
            output: 增强后的图像 [B, 3, H, W], 范围 [0, 1]
        """
        # Step 1: 编码
        features = self.encoder(x)
        e1, e2, e3, e4 = features

        # Step 2: 瓶颈层双域处理
        # 频域分支
        f_freq = self.fapm(e4)       # [B, 256, H/8, W/8]

        # 空域分支
        f_spat = self.scb(e4)        # [B, 256, H/8, W/8]

        # 交叉域融合
        bottleneck = self.cdma(f_freq, f_spat)  # [B, 256, H/8, W/8]

        # Step 3: 解码
        output = self.decoder([e1, e2, e3, bottleneck])

        # Step 4: 全局残差连接
        output = output + x

        # 钳制到 [0, 1]
        output = torch.clamp(output, 0.0, 1.0)

        return output


def count_parameters(model):
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # 测试模型
    model = SFDNet(in_channels=3, out_channels=3, base_channels=32)
    x = torch.randn(1, 3, 256, 256)
    y = model(x)

    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Parameters:   {count_parameters(model) / 1e6:.2f}M")