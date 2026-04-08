"""
图像处理工具函数
"""

import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms


def load_image(path, size=None):
    """加载图像并转为 tensor"""
    img = Image.open(path).convert('RGB')
    if size is not None:
        img = img.resize((size, size), Image.BICUBIC)
    transform = transforms.ToTensor()
    return transform(img).unsqueeze(0)


def save_image(tensor, path):
    """将 tensor 保存为图像"""
    tensor = tensor.squeeze(0).clamp(0, 1)
    img = transforms.ToPILImage()(tensor.cpu())
    img.save(path)


def tensor_to_numpy(tensor):
    """tensor [1, 3, H, W] -> numpy [H, W, 3]"""
    return tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()


def numpy_to_tensor(array):
    """numpy [H, W, 3] -> tensor [1, 3, H, W]"""
    return torch.from_numpy(
        array.transpose(2, 0, 1)
    ).unsqueeze(0).float()