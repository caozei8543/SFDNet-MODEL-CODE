"""
SFDNet Training Script
使用方法:
    python train.py --config configs/train_lol.yaml --gpu 0
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from options.train_options import parse_train_options
from models import SFDNet, TotalLoss
from data import LOLv1Dataset, RELLISURDataset, LSRWDataset
from utils.metrics import MetricCalculator
from utils.img_utils import tensor_to_numpy
from utils.logger import Logger


def get_dataset(config, split='train'):
    """根据配置文件选择数据集"""
    name = config['dataset']['name']
    root = config['dataset'][f'{split}_root']
    ps = config['dataset'].get('patch_size', 256)

    if name == 'LOLv1':
        return LOLv1Dataset(root, split=split, patch_size=ps)
    elif name == 'RELLISUR':
        return RELLISURDataset(root, split=split)
    elif name == 'LSRW':
        return LSRWDataset(root, split=split)
    else:
        raise ValueError(f"Unknown dataset: {name}")


def train_one_epoch(model, loader, criterion, optimizer, device, logger,
                    epoch):
    """训练一个 epoch"""
    model.train()
    total_loss = 0.0

    pbar = tqdm(loader, desc=f'Epoch {epoch}')
    for i, batch in enumerate(pbar):
        low = batch['low'].to(device)
        high = batch['high'].to(device)

        # 前向传播
        pred = model(low)

        # 计算损失
        loss, loss_dict = criterion(pred, high)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪 (防止频域梯度爆炸)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()

        # 更新进度条
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'char': f"{loss_dict['charbonnier'].item():.4f}",
            'freq': f"{loss_dict['frequency'].item():.4f}",
        })

        # TensorBoard 记录
        step = epoch * len(loader) + i
        for key, value in loss_dict.items():
            logger.log_scalar(f'train/{key}', value.item(), step)

    avg_loss = total_loss / len(loader)
    return avg_loss


@torch.no_grad()
def validate(model, loader, metric_calc, device):
    """验证"""
    model.eval()
    psnr_list, ssim_list, lpips_list = [], [], []

    for batch in tqdm(loader, desc='Validating'):
        low = batch['low'].to(device)
        high = batch['high'].to(device)

        pred = model(low)

        # 计算指标
        pred_np = tensor_to_numpy(pred.clamp(0, 1))
        high_np = tensor_to_numpy(high)

        psnr_val = metric_calc.calculate_psnr(pred_np, high_np)
        ssim_val = metric_calc.calculate_ssim(pred_np, high_np)
        lpips_val = metric_calc.calculate_lpips(pred, high)

        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)
        lpips_list.append(lpips_val)

    avg_psnr = sum(psnr_list) / len(psnr_list)
    avg_ssim = sum(ssim_list) / len(ssim_list)
    avg_lpips = sum(lpips_list) / len(lpips_list)

    return avg_psnr, avg_ssim, avg_lpips


def main():
    # ========== 解析配置 ==========
    config = parse_train_options()
    os.environ['CUDA_VISIBLE_DEVICES'] = config['gpu']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ========== 日志 ==========
    logger = Logger(
        log_dir=config['save']['log_dir'],
        exp_name=config['dataset']['name']
    )
    logger.log(f"Configuration: {config}")

    # ========== 数据集 ==========
    train_dataset = get_dataset(config, split='train')
    test_dataset = get_dataset(config, split='test')

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['dataset']['num_workers'],
        pin_memory=True,
        drop_last=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
    )

    # ========== 模型 ==========
    model = SFDNet(
        in_channels=config['model']['in_channels'],
        out_channels=config['model']['out_channels'],
        base_channels=config['model']['base_channels'],
    ).to(device)

    # 打印参数量
    total_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    logger.log(f"Total parameters: {total_params / 1e6:.2f}M")

    # ========== 损失函数 ==========
    criterion = TotalLoss(
        lambda_char=config['loss']['lambda_char'],
        lambda_edge=config['loss']['lambda_edge'],
        lambda_freq=config['loss']['lambda_freq'],
        lambda_percep=config['loss']['lambda_percep'],
    ).to(device)

    # ========== 优化器 & 调度器 ==========
    optimizer = AdamW(
        model.parameters(),
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay'],
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config['training']['epochs'],
        eta_min=config['training']['min_lr'],
    )

    # ========== 恢复训练 ==========
    start_epoch = 0
    best_psnr = 0.0

    if config['resume'] is not None:
        checkpoint = torch.load(config['resume'], map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        best_psnr = checkpoint['best_psnr']
        logger.log(f"Resumed from epoch {start_epoch}")

    # ========== 指标计算器 ==========
    metric_calc = MetricCalculator(device=device)

    # ========== 训练循环 ==========
    save_dir = config['save']['save_dir']
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(start_epoch, config['training']['epochs']):
        # 训练
        avg_loss = train_one_epoch(
            model, train_loader, criterion, optimizer,
            device, logger, epoch
        )
        scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        logger.log(
            f"Epoch [{epoch}/{config['training']['epochs']}] "
            f"Loss: {avg_loss:.4f} | LR: {current_lr:.6f}"
        )
        logger.log_scalar('train/epoch_loss', avg_loss, epoch)
        logger.log_scalar('train/lr', current_lr, epoch)

        # 验证
        if (epoch + 1) % config['save']['save_freq'] == 0:
            avg_psnr, avg_ssim, avg_lpips = validate(
                model, test_loader, metric_calc, device
            )

            logger.log(
                f"  Val -> PSNR: {avg_psnr:.2f} | "
                f"SSIM: {avg_ssim:.4f} | LPIPS: {avg_lpips:.4f}"
            )
            logger.log_scalar('val/psnr', avg_psnr, epoch)
            logger.log_scalar('val/ssim', avg_ssim, epoch)
            logger.log_scalar('val/lpips', avg_lpips, epoch)

            # 保存最优模型
            if avg_psnr > best_psnr:
                best_psnr = avg_psnr
                torch.save({
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'best_psnr': best_psnr,
                }, os.path.join(save_dir, 'best_model.pth'))
                logger.log(
                    f"  *** New Best PSNR: {best_psnr:.2f} ***"
                )

            # 保存当前 checkpoint
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_psnr': best_psnr,
            }, os.path.join(save_dir, f'epoch_{epoch}.pth'))

    logger.log(f"Training finished. Best PSNR: {best_psnr:.2f}")
    logger.close()


if __name__ == '__main__':
    main()