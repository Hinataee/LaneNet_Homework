"""
LaneNet评测脚本
"""

import os
import sys
import argparse
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import LaneNet
from data import TuSimpleDataset, get_val_transforms
from config.config import DATASET_CONFIG, MODEL_CONFIG, EVAL_CONFIG
from utils import embedding_post_process, fit_lane_lines, visualize_lanes


class Evaluator:
    """评测器"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # 创建可视化目录
        if EVAL_CONFIG['save_visualizations']:
            os.makedirs(EVAL_CONFIG['visualization_dir'], exist_ok=True)
        
        # 构建模型
        self.model = LaneNet(embedding_dim=MODEL_CONFIG['embedding_dim'], use_hnet=True).to(self.device)
        
        # 加载checkpoint
        checkpoint_path = args.checkpoint or EVAL_CONFIG['checkpoint']
        if checkpoint_path is None:
            raise ValueError("Please provide checkpoint path!")
        
        self.load_checkpoint(checkpoint_path)
        
        # 加载测试数据集
        print("Loading test dataset...")
        self.test_dataset = TuSimpleDataset(
            root_dir=DATASET_CONFIG['test_dir'],
            json_file=DATASET_CONFIG['test_json'],
            transform=get_val_transforms(),
            target_size=MODEL_CONFIG['input_size']
        )
        
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=EVAL_CONFIG['batch_size'],
            shuffle=False,
            num_workers=EVAL_CONFIG['num_workers'],
            pin_memory=True
        )
        
        print(f"Test samples: {len(self.test_dataset)}")
    
    def load_checkpoint(self, checkpoint_path):
        """加载checkpoint"""
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print("Checkpoint loaded successfully!")
    
    @torch.no_grad()
    def evaluate(self):
        """评测"""
        self.model.eval()
        
        print("\n" + "=" * 50)
        print("Start Evaluation")
        print("=" * 50 + "\n")
        
        total_accuracy = 0.0
        total_fp = 0
        total_fn = 0
        num_samples = 0
        
        pbar = tqdm(self.test_loader, desc="Evaluating")
        
        for batch_idx, batch in enumerate(pbar):
            # 数据移到设备
            images = batch['image'].to(self.device)
            binary_labels = batch['binary_label'].to(self.device)
            
            # 前向传播
            outputs = self.model(images)
            
            # 二值分割预测
            binary_seg_pred = torch.argmax(outputs['binary_seg'], dim=1)  # [B, H, W]
            
            # 计算准确率
            correct = (binary_seg_pred == binary_labels).float()
            accuracy = correct.mean().item()
            total_accuracy += accuracy
            
            # 计算FP和FN
            fp = ((binary_seg_pred == 1) & (binary_labels == 0)).sum().item()
            fn = ((binary_seg_pred == 0) & (binary_labels == 1)).sum().item()
            total_fp += fp
            total_fn += fn
            
            num_samples += images.size(0)
            
            # 更新进度条
            pbar.set_postfix({
                'acc': f"{accuracy:.4f}"
            })
            
            # 可视化
            if EVAL_CONFIG['save_visualizations'] and batch_idx < 10:
                self.visualize_batch(batch, outputs, batch_idx)
        
        # 计算平均指标
        avg_accuracy = total_accuracy / len(self.test_loader)
        precision = 1.0 - (total_fp / (total_fp + (num_samples * MODEL_CONFIG['input_size'][0] * MODEL_CONFIG['input_size'][1]) - total_fp - total_fn))
        recall = 1.0 - (total_fn / (total_fn + (num_samples * MODEL_CONFIG['input_size'][0] * MODEL_CONFIG['input_size'][1]) - total_fp - total_fn))
        
        # 打印结果
        print("\n" + "=" * 50)
        print("Evaluation Results")
        print("=" * 50)
        print(f"Average Accuracy: {avg_accuracy:.4f}")
        print(f"False Positives: {total_fp}")
        print(f"False Negatives: {total_fn}")
        print("=" * 50 + "\n")
        
        return avg_accuracy
    
    def visualize_batch(self, batch, outputs, batch_idx):
        """可视化一个batch"""
        images = batch['image'].cpu()
        binary_labels = batch['binary_label'].cpu().numpy()
        instance_labels = batch['instance_label'].cpu().numpy()
        
        binary_seg_pred = torch.argmax(outputs['binary_seg'], dim=1).cpu().numpy()
        instance_seg_pred = outputs['instance_seg'].cpu().permute(0, 2, 3, 1).numpy()

        # 获取 HNet 预测参数
        hnet_params = [None]*images.size(0)
        if 'hnet_params' in outputs:
            hnet_params = outputs['hnet_params'].cpu()
            H = torch.zeros(hnet_params.size(0), 3, 3, device=hnet_params.device)
            # Row 0: [a, b, c]
            H[:, 0, 0] = hnet_params[:, 0] # a
            H[:, 0, 1] = hnet_params[:, 1] # b
            H[:, 0, 2] = hnet_params[:, 2] # c
            # Row 1: [0, d, e]
            H[:, 1, 1] = hnet_params[:, 3] # d
            H[:, 1, 2] = hnet_params[:, 4] # e
            # Row 2: [0, f, 1]
            H[:, 2, 1] = hnet_params[:, 5] # f
            H[:, 2, 2] = 1.0
            hnet_params = H.cpu()
        
        batch_size = images.size(0)
        
        for i in range(min(batch_size, 4)):  # 最多可视化4张
            # 反归一化图像
            image = images[i].permute(1, 2, 0).numpy()
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image = image * std + mean
            image = (image * 255).astype(np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # 后处理
            instance_mask, num_lanes = embedding_post_process(
                instance_seg_pred[i],
                binary_seg_pred[i],
                delta_v=0.5,
                min_cluster_size=50
            )


            
            # 拟合车道线
            lane_lines = fit_lane_lines(instance_mask, num_lanes, hnet_matrix=hnet_params[i])
            
            # 可视化
            vis_image = visualize_lanes(image, lane_lines)
            
            # 生成二值分割可视化（预测与标签）
            binary_pred_vis = (binary_seg_pred[i] * 255).astype(np.uint8)
            binary_gt_vis = (binary_labels[i] * 255).astype(np.uint8)

            # 生成实例分割可视化（简易调色板）
            def colorize_instance(mask: np.ndarray) -> np.ndarray:
                palette = [
                    (0, 0, 0),       # 背景
                    (255, 0, 0),     # 车道1
                    (0, 255, 0),     # 车道2
                    (0, 0, 255),     # 车道3
                    (255, 255, 0),   # 车道4
                    (255, 0, 255),   # 车道5
                    (0, 255, 255),   # 车道6
                ]
                h, w = mask.shape
                color = np.zeros((h, w, 3), dtype=np.uint8)
                for idx, c in enumerate(palette):
                    color[mask == idx] = c
                # 其他大于调色板的实例采用循环颜色
                max_label = mask.max()
                if max_label >= len(palette):
                    extra_colors = palette[1:]  # 跳过背景
                    for label in range(len(palette), max_label + 1):
                        color[mask == label] = extra_colors[(label - 1) % len(extra_colors)]
                return color

            instance_pred_vis = colorize_instance(instance_mask)
            instance_gt_vis = colorize_instance(instance_labels[i])

            # 保存：原图叠加、二值预测/标签、实例预测/标签
            base = f'batch_{batch_idx}_sample_{i}'

            cv2.imwrite(os.path.join(EVAL_CONFIG['visualization_dir'], base + '_overlay.jpg'), vis_image)
            cv2.imwrite(os.path.join(EVAL_CONFIG['visualization_dir'], base + '_binary_pred.png'), binary_pred_vis)
            cv2.imwrite(os.path.join(EVAL_CONFIG['visualization_dir'], base + '_binary_gt.png'), binary_gt_vis)
            cv2.imwrite(os.path.join(EVAL_CONFIG['visualization_dir'], base + '_instance_pred.png'), instance_pred_vis)
            cv2.imwrite(os.path.join(EVAL_CONFIG['visualization_dir'], base + '_instance_gt.png'), instance_gt_vis)


def main():
    parser = argparse.ArgumentParser(description='Evaluate LaneNet')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    args = parser.parse_args()
    
    # 检查数据集
    if not os.path.exists(DATASET_CONFIG['test_json']):
        print("Error: Test dataset not found!")
        print(f"Expected at: {DATASET_CONFIG['test_json']}")
        print("\nPlease run: python data/download_dataset.py")
        return
    
    # 创建评测器
    evaluator = Evaluator(args)
    
    # 开始评测
    evaluator.evaluate()


if __name__ == '__main__':
    main()
