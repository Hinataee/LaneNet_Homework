"""
TuSimple数据集加载器
"""

import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class TuSimpleDataset(Dataset):
    """TuSimple车道线检测数据集"""
    
    def __init__(self, root_dir, json_file, transform=None, target_size=(256, 512)):
        """
        Args:
            root_dir: 数据集根目录
            json_file: 标注文件路径
            transform: 数据增强
            target_size: 目标尺寸 (height, width)
        """
        self.root_dir = root_dir
        self.target_size = target_size
        self.transform = transform
        
        # 加载标注
        self.annotations = []
        if os.path.exists(json_file):
            with open(json_file, 'r') as f:
                for line in f:
                    self.annotations.append(json.loads(line))
        else:
            print(f"Warning: {json_file} not found!")
            
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        anno = self.annotations[idx]
        
        # 加载图像
        img_path = os.path.join(self.root_dir, anno['raw_file'])
        image = Image.open(img_path).convert('RGB')
        
        # 获取原始尺寸
        original_size = image.size  # (width, height)
        
        # 生成标签
        binary_label, instance_label = self._generate_label(anno, original_size)
        
        # 调整大小
        image = image.resize((self.target_size[1], self.target_size[0]), Image.BILINEAR)
        binary_label = cv2.resize(binary_label, (self.target_size[1], self.target_size[0]), 
                                  interpolation=cv2.INTER_NEAREST)
        instance_label = cv2.resize(instance_label, (self.target_size[1], self.target_size[0]), 
                                    interpolation=cv2.INTER_NEAREST)
        
        # 转换为tensor
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
            image = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                        std=[0.229, 0.224, 0.225])(image)
        
        binary_label = torch.from_numpy(binary_label).long()
        instance_label = torch.from_numpy(instance_label).long()
        
        return {
            'image': image,
            'binary_label': binary_label,
            'instance_label': instance_label,
            'img_path': img_path
        }
    
    def _generate_label(self, anno, original_size):
        """生成二值和实例标签"""
        width, height = original_size
        
        # 初始化标签
        binary_label = np.zeros((height, width), dtype=np.uint8)
        instance_label = np.zeros((height, width), dtype=np.uint8)
        
        # 绘制车道线
        lanes = anno['lanes']
        h_samples = anno['h_samples']
        
        lane_id = 1
        for lane in lanes:
            # 过滤有效点
            points = []
            for x, y in zip(lane, h_samples):
                if x >= 0:  # -2表示该点不存在
                    points.append([x, y])
            
            if len(points) < 2:
                continue
                
            points = np.array(points, dtype=np.int32)
            
            # 绘制二值标签
            cv2.polylines(binary_label, [points], False, 1, thickness=5)
            
            # 绘制实例标签
            cv2.polylines(instance_label, [points], False, lane_id, thickness=5)
            
            lane_id += 1
        
        return binary_label, instance_label


def get_train_transforms():
    """训练数据增强"""
    return transforms.Compose([
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_val_transforms():
    """验证数据增强"""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


if __name__ == '__main__':
    # 测试数据集
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config.config import DATASET_CONFIG
    
    dataset = TuSimpleDataset(
        root_dir=DATASET_CONFIG['train_dir'],
        json_file=DATASET_CONFIG['train_json'],
        transform=get_train_transforms()
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Image shape: {sample['image'].shape}")
        print(f"Binary label shape: {sample['binary_label'].shape}")
        print(f"Instance label shape: {sample['instance_label'].shape}")
        print(f"Unique instances: {torch.unique(sample['instance_label'])}")
