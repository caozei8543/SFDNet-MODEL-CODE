"""
Cross-Domain Mutual Attention (CDMA)
论文公式 (10)-(12) 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class CrossGating(nn.Module):
    """正交交叉门控: 用一个域的语义去门控另一个域"""

    def __init__(self, channels):
        super(CrossGating, self).__init__()
        self.norm_freq = nn.LayerNorm(channels)
        self.norm_spat = nn.LayerNorm(channels)

        # 门控投影
        self.gate_freq = nn.Linear(channels, channels)
        self.gate_spat = nn.Linear(channels, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, f_freq, f_spat):
        """
        Args:
            f_freq: 频域特征 [B, N, C]
            f_spat: 空域特征 [B, N, C]
        Returns:
            gated_freq, gated_spat: 交叉门控后的特征
        """
        # 公式 (10): 用空域语义门控频域特征
        gate_for_freq = self.sigmoid(self.gate_spat(self.norm_spat(f_spat)))
        gated_freq = f_freq * gate_for_freq

        # 公式 (11): 用频域语义门控空域特征
        gate_for_spat = self.sigmoid(self.gate_freq(self.norm_freq(f_freq)))
        gated_spat = f_spat * gate_for_spat

        return gated_freq, gated_spat


class MutualAttention(nn.Module):
    """互注意力: 频域和空域特征互为 Q-KV 对"""

    def __init__(self, channels, num_heads=4):
        super(MutualAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5

        # 频域分支的 Q 投影
        self.q_freq = nn.Linear(channels, channels)
        # 空域分支的 K, V 投影
        self.k_spat = nn.Linear(channels, channels)
        self.v_spat = nn.Linear(channels, channels)

        # 空域分支的 Q 投影
        self.q_spat = nn.Linear(channels, channels)
        # 频域分支的 K, V 投影
        self.k_freq = nn.Linear(channels, channels)
        self.v_freq = nn.Linear(channels, channels)

        self.proj = nn.Linear(channels * 2, channels)
        self.norm = nn.LayerNorm(channels)

    def forward(self, f_freq, f_spat):
        """
        Args:
            f_freq: [B, N, C]
            f_spat: [B, N, C]
        Returns:
            fused: [B, N, C]
        """
        B, N, C = f_freq.shape

        # --- 频域 Query, 空域 Key-Value (公式 12) ---
        q_f = rearrange(
            self.q_freq(f_freq), 'b n (h d) -> b h n d',
            h=self.num_heads
        )
        k_s = rearrange(
            self.k_spat(f_spat), 'b n (h d) -> b h n d',
            h=self.num_heads
        )
        v_s = rearrange(
            self.v_spat(f_spat), 'b n (h d) -> b h n d',
            h=self.num_heads
        )

        attn_f2s = F.softmax(
            (q_f @ k_s.transpose(-2, -1)) * self.scale, dim=-1
        )
        out_f2s = rearrange(
            attn_f2s @ v_s, 'b h n d -> b n (h d)'
        )

        # --- 空域 Query, 频域 Key-Value (公式 13) ---
        q_s = rearrange(
            self.q_spat(f_spat), 'b n (h d) -> b h n d',
            h=self.num_heads
        )
        k_f = rearrange(
            self.k_freq(f_freq), 'b n (h d) -> b h n d',
            h=self.num_heads
        )
        v_f = rearrange(
            self.v_freq(f_freq), 'b n (h d) -> b h n d',
            h=self.num_heads
        )

        attn_s2f = F.softmax(
            (q_s @ k_f.transpose(-2, -1)) * self.scale, dim=-1
        )
        out_s2f = rearrange(
            attn_s2f @ v_f, 'b h n d -> b n (h d)'
        )

        # 公式 (14): 拼接 + 投影
        fused = self.proj(torch.cat([out_f2s, out_s2f], dim=-1))
        fused = self.norm(fused)

        return fused


class CDMA(nn.Module):
    """
    Cross-Domain Mutual Attention (CDMA)
    论文 Section 3.4 的完整实现

    核心思想: 先用正交交叉门控过滤语义冲突,
    再用互注意力实现空频两域的最优融合
    """

    def __init__(self, channels, num_heads=4):
        super(CDMA, self).__init__()
        self.cross_gate = CrossGating(channels)
        self.mutual_attn = MutualAttention(channels, num_heads)

        # 输出映射
        self.out_conv = nn.Conv2d(channels, channels, 1, 1, 0)

    def forward(self, x_freq, x_spat):
        """
        Args:
            x_freq: 频域特征 [B, C, H, W]
            x_spat: 空域特征 [B, C, H, W]
        Returns:
            fused: 融合后的特征 [B, C, H, W]
        """
        B, C, H, W = x_freq.shape

        # Reshape to sequence: [B, C, H, W] -> [B, H*W, C]
        f_freq = rearrange(x_freq, 'b c h w -> b (h w) c')
        f_spat = rearrange(x_spat, 'b c h w -> b (h w) c')

        # Step 1: 正交交叉门控
        gated_freq, gated_spat = self.cross_gate(f_freq, f_spat)

        # Step 2: 互注意力融合
        fused = self.mutual_attn(gated_freq, gated_spat)

        # Reshape back: [B, H*W, C] -> [B, C, H, W]
        fused = rearrange(fused, 'b (h w) c -> b c h w', h=H, w=W)

        # Step 3: 输出投影 + 残差
        fused = self.out_conv(fused) + x_freq + x_spat

        return fused