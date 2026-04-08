"""
Frequency Amplitude-Phase Modulator (FAPM)
论文公式 (1)-(5)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AmplitudeModulator(nn.Module):
    """幅值调制子网络 M_A: 处理全局亮度"""

    def __init__(self, channels):
        super(AmplitudeModulator, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.norm = nn.InstanceNorm2d(channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, amplitude):
        """
        Args:
            amplitude: 幅值谱 |F(x)|, shape [B, C, H, W]
        Returns:
            modulated_amplitude: 调制后的幅值谱
        """
        residual = amplitude
        out = self.act(self.norm(self.conv1(amplitude)))
        out = self.sigmoid(self.conv2(out))
        # 公式 (3): A'(u,v) = A(u,v) * sigmoid(M_A(A))
        return residual * out


class PhaseModulator(nn.Module):
    """相位调制子网络 M_P: 滤除噪声的混沌相位"""

    def __init__(self, channels):
        super(PhaseModulator, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.norm = nn.InstanceNorm2d(channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.tanh = nn.Tanh()

    def forward(self, phase):
        """
        Args:
            phase: 相位谱 angle(F(x)), shape [B, C, H, W]
        Returns:
            modulated_phase: 调制后的相位谱
        """
        residual = phase
        out = self.act(self.norm(self.conv1(phase)))
        out = self.tanh(self.conv2(out))
        # 公式 (4): P'(u,v) = P(u,v) + tanh(M_P(P))
        return residual + out


class FAPM(nn.Module):
    """
    Frequency Amplitude-Phase Modulator (FAPM)
    论文 Section 3.2 的完整实现

    核心思想: 对输入特征做2D FFT, 分别对幅值和相位进行独立调制,
    然后通过IFFT重建空域特征
    """

    def __init__(self, channels):
        super(FAPM, self).__init__()
        self.amp_mod = AmplitudeModulator(channels)
        self.pha_mod = PhaseModulator(channels)
        # 频域到空域的过渡卷积
        self.fusion_conv = nn.Conv2d(channels, channels, 1, 1, 0)

    def forward(self, x):
        """
        Args:
            x: 空域特征 shape [B, C, H, W]
        Returns:
            x_freq: 经频域调制后的空域特征 shape [B, C, H, W]
        """
        B, C, H, W = x.shape

        # Step 1: 2D FFT (公式 1)
        # torch.fft.rfft2 输出复数张量, shape [B, C, H, W//2+1]
        freq = torch.fft.rfft2(x, norm='ortho')

        # Step 2: 分离幅值和相位 (公式 2)
        amplitude = torch.abs(freq)       # |F(u,v)|
        phase = torch.angle(freq)         # angle(F(u,v))

        # Step 3: 独立调制
        amp_mod = self.amp_mod(amplitude)  # 公式 (3)
        pha_mod = self.pha_mod(phase)      # 公式 (4)

        # Step 4: 重建复数频谱并 IFFT (公式 5)
        # F'(u,v) = A'(u,v) * exp(j * P'(u,v))
        real = amp_mod * torch.cos(pha_mod)
        imag = amp_mod * torch.sin(pha_mod)
        freq_mod = torch.complex(real, imag)

        # Step 5: 逆FFT回到空域
        x_freq = torch.fft.irfft2(freq_mod, s=(H, W), norm='ortho')

        # Step 6: 1x1 卷积平滑过渡
        x_freq = self.fusion_conv(x_freq)

        return x_freq