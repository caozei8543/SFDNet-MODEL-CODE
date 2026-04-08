"""
LOL-v1 Dataset Loader
"""

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import random


class LOLv1Dataset(Dataset):
    """
    LOL-v1 数据集
    结构:
        data_root/
            low/   -> 低光照图像
            high/  -> 正常光照图像 (GT)
    """

    def __init__(self, data_root, split='train', patch_size=256):
        super(LOLv1Dataset, self).__init__()
        self.data_root = data_root
        self.split = split
        self.patch_size = patch_size

        self.low_dir = os.path.join(data_root, split, 'low')
        self.high_dir = os.path.join(data_root, split, 'high')

        self.filenames = sorted(os.listdir(self.low_dir))

        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]

        low_img = Image.open(
            os.path.join(self.low_dir, fname)
        ).convert('RGB')
        high_img = Image.open(
            os.path.join(self.high_dir, fname)
        ).convert('RGB')

        low_img = self.transform(low_img)
        high_img = self.transform(high_img)

        # 训练时随机裁剪
        if self.split == 'train':
            low_img, high_img = self._random_crop(low_img, high_img)
            low_img, high_img = self._random_augment(low_img, high_img)

        return {
            'low': low_img,
            'high': high_img,
            'filename': fname,
        }

    def _random_crop(self, low, high):
        """随机裁剪 patch"""
        _, h, w = low.shape
        ps = self.patch_size

        if h < ps or w < ps:
            low = transforms.functional.resize(low, [ps, ps])
            high = transforms.functional.resize(high, [ps, ps])
            return low, high

        top = random.randint(0, h - ps)
        left = random.randint(0, w - ps)

        low = low[:, top:top + ps, left:left + ps]
        high = high[:, top:top + ps, left:left + ps]
        return low, high

    def _random_augment(self, low, high):
        """随机数据增强 (翻转 + 旋转)"""
        # 随机水平翻转
        if random.random() > 0.5:
            low = torch.flip(low, [-1])
            high = torch.flip(high, [-1])
        # 随机垂直翻转
        if random.random() > 0.5:
            low = torch.flip(low, [-2])
            high = torch.flip(high, [-2])
        return low, high