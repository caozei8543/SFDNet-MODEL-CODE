
训练日志管理


import os
import logging
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


class Logger
    统一日志管理器

    def __init__(self, log_dir='.logs', exp_name='sfdnet')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.exp_name = f{exp_name}_{timestamp}
        self.log_dir = os.path.join(log_dir, self.exp_name)
        os.makedirs(self.log_dir, exist_ok=True)

        # TensorBoard
        self.writer = SummaryWriter(self.log_dir)

        # Python Logger
        self.logger = logging.getLogger(self.exp_name)
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(
            os.path.join(self.log_dir, 'train.log')
        )
        handler.setFormatter(
            logging.Formatter('%(asctime)s - %(message)s')
        )
        self.logger.addHandler(handler)
        self.logger.addHandler(logging.StreamHandler())

    def log(self, message)
        self.logger.info(message)

    def log_scalar(self, tag, value, step)
        self.writer.add_scalar(tag, value, step)

    def log_images(self, tag, images, step)
        self.writer.add_images(tag, images, step)

    def close(self)
        self.writer.close()