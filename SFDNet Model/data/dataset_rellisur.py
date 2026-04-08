"""
RELLISUR Dataset Loader
"""

import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class RELLISURDataset(Dataset):
    """
    RELLISUR 数据集
    结构:
        data_root/
            input/  -> 低光照图像
            GT/     -> 正常光照图像
    """

    def __init__(self, data_root, split='test', size=512):
        super(RELLISURDataset, self).__init__()
        self.input_dir = os.path.join(data_root, split, 'input')
        self.gt_dir = os.path.join(data_root, split, 'GT')

        self.filenames = sorted(os.listdir(self.input_dir))
        self.size = size

        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]

        input_img = Image.open(
            os.path.join(self.input_dir, fname)
        ).convert('RGB')
        gt_img = Image.open(
            os.path.join(self.gt_dir, fname)
        ).convert('RGB')

        input_img = self.transform(input_img)
        gt_img = self.transform(gt_img)

        return {
            'low': input_img,
            'high': gt_img,
            'filename': fname,
        }