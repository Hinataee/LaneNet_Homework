"""
LaneNet训练脚本
"""

import os
import sys
import argparse
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import LaneNet, LaneNetLoss
from data import TuSimpleDataset, get_train_transforms, get_val_transforms
from config.config import DATASET_CONFIG, MODEL_CONFIG, TRAIN_CONFIG, LOSS_CONFIG


class Trainer:
    """训练器"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # 创建保存目录
        os.makedirs(TRAIN_CONFIG['save_dir'], exist_ok=True)
        os.makedirs(TRAIN_CONFIG['log_dir'], exist_ok=True)
        
        # TensorBoard
        self.writer = SummaryWriter(TRAIN_CONFIG['log_dir'])
        
        # 构建模型
        self.model = LaneNet(embedding_dim=MODEL_CONFIG['embedding_dim'], use_hnet=False).to(self.device)
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # 损失函数
        self.criterion = LaneNetLoss(
            delta_v=LOSS_CONFIG['discriminative_delta_v'],
            delta_d=LOSS_CONFIG['discriminative_delta_d'],
            binary_weight=LOSS_CONFIG['binary_weight'],
            instance_weight=LOSS_CONFIG['instance_weight']
        ).to(self.device)
        
        # 优化器
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=TRAIN_CONFIG['learning_rate'],
            weight_decay=TRAIN_CONFIG['weight_decay']
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=TRAIN_CONFIG['num_epochs']
        )
        
        # 加载数据集
        print("Loading datasets...")
        self.train_dataset = TuSimpleDataset(
            root_dir=DATASET_CONFIG['train_dir'],
            json_file=DATASET_CONFIG['train_json'],
            transform=get_train_transforms(),
            target_size=MODEL_CONFIG['input_size']
        )
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=TRAIN_CONFIG['batch_size'],
            shuffle=True,
            num_workers=TRAIN_CONFIG['num_workers'],
            pin_memory=True
        )
        
        print(f"Train samples: {len(self.train_dataset)}")
        
        # 恢复训练
        self.start_epoch = 0
        if TRAIN_CONFIG['resume']:
            self.load_checkpoint(TRAIN_CONFIG['resume'])
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        
        epoch_losses = {
            'total': 0.0,
            'binary': 0.0,
            'instance': 0.0,
            'var': 0.0,
            'dist': 0.0,
            'reg': 0.0,
            'hnet': 0.0  

        }
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{TRAIN_CONFIG['num_epochs']}")
        
        for batch_idx, batch in enumerate(pbar):
            # 数据移到设备
            images = batch['image'].to(self.device)
            binary_labels = batch['binary_label'].to(self.device)
            instance_labels = batch['instance_label'].to(self.device)
            
            # 前向传播
            outputs = self.model(images)
            
            # 计算损失
            losses = self.criterion(outputs, binary_labels, instance_labels)
            
            # 反向传播
            self.optimizer.zero_grad()
            losses['total_loss'].backward()
            self.optimizer.step()
            
            # 统计损失
            epoch_losses['total'] += losses['total_loss'].item()
            epoch_losses['binary'] += losses['binary_loss'].item()
            epoch_losses['instance'] += losses['instance_loss'].item()
            epoch_losses['var'] += losses['var_loss'].item()
            epoch_losses['dist'] += losses['dist_loss'].item()
            epoch_losses['reg'] += losses['reg_loss'].item()
            epoch_losses['hnet'] += losses['hnet_loss'].item() 
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f"{losses['total_loss'].item():.4f}",
                'binary': f"{losses['binary_loss'].item():.4f}",
                'instance': f"{losses['instance_loss'].item():.4f}",
                'hnet': f"{losses['hnet_loss'].item():.4f}"})
            
            # TensorBoard记录
            if batch_idx % TRAIN_CONFIG['print_freq'] == 0:
                step = epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('train/total_loss', losses['total_loss'].item(), step)
                self.writer.add_scalar('train/binary_loss', losses['binary_loss'].item(), step)
                self.writer.add_scalar('train/instance_loss', losses['instance_loss'].item(), step)
                self.writer.add_scalar('train/var_loss', losses['var_loss'].item(), step)
                self.writer.add_scalar('train/dist_loss', losses['dist_loss'].item(), step)
                self.writer.add_scalar('train/reg_loss', losses['reg_loss'].item(), step)
                self.writer.add_scalar('train/hnet_loss', losses['hnet_loss'].item(), step) 
 
        # 平均损失
        num_batches = len(self.train_loader)
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    def train(self):
        """训练主循环"""
        print("\n" + "=" * 50)
        print("Start Training")
        print("=" * 50 + "\n")
        
        best_loss = float('inf')
        
        for epoch in range(self.start_epoch, TRAIN_CONFIG['num_epochs']):
            start_time = time.time()
            
            # 训练
            train_losses = self.train_epoch(epoch)
            
            # 更新学习率
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 记录学习率
            self.writer.add_scalar('train/learning_rate', current_lr, epoch)
            
            # 打印信息
            epoch_time = time.time() - start_time
            print(f"\nEpoch {epoch}/{TRAIN_CONFIG['num_epochs']} - "
                  f"Time: {epoch_time:.2f}s - "
                  f"LR: {current_lr:.6f}")
            print(f"Train Loss: {train_losses['total']:.4f} "
                  f"(Binary: {train_losses['binary']:.4f}, "
                  f"Instance: {train_losses['instance']:.4f})"
                  f"HNet: {train_losses['hnet']:.4f})")
            
            # 保存checkpoint
            if (epoch + 1) % TRAIN_CONFIG['save_freq'] == 0:
                self.save_checkpoint(epoch, train_losses['total'])
            
            # 保存最佳模型
            if train_losses['total'] < best_loss:
                best_loss = train_losses['total']
                self.save_checkpoint(epoch, train_losses['total'], is_best=True)
                print(f"✓ Best model saved! Loss: {best_loss:.4f}")
        
        print("\n" + "=" * 50)
        print("Training Complete!")
        print("=" * 50)
        
        self.writer.close()
    
    def save_checkpoint(self, epoch, loss, is_best=False):
        """保存checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
        }
        
        if is_best:
            save_path = os.path.join(TRAIN_CONFIG['save_dir'], 'best_model.pth')
        else:
            save_path = os.path.join(TRAIN_CONFIG['save_dir'], f'checkpoint_epoch_{epoch}.pth')
        
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved to {save_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """加载checkpoint"""
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        
        print(f"Resumed from epoch {self.start_epoch}")


def main():
    parser = argparse.ArgumentParser(description='Train LaneNet')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    args = parser.parse_args()
    
    # 检查数据集
    if not os.path.exists(DATASET_CONFIG['train_json']):
        print("Error: Training dataset not found!")
        print(f"Expected at: {DATASET_CONFIG['train_json']}")
        print("\nPlease run: python data/download_dataset.py")
        return
    
    # 创建训练器
    trainer = Trainer(args)
    
    # 开始训练
    trainer.train()


if __name__ == '__main__':
    main()
