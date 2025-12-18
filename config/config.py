"""
LaneNet配置文件
包含训练、数据集、模型等相关配置
"""

import os

# 项目根目录
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 数据集配置
DATASET_CONFIG = {
    'name': 'TuSimple',
    'root_dir': os.path.join(ROOT_DIR, 'data', 'tusimple'),
    'train_dir': os.path.join(ROOT_DIR, 'data', 'tusimple', 'train_set'),
    'test_dir': os.path.join(ROOT_DIR, 'data', 'tusimple', 'test_set'),
    'train_json': os.path.join(ROOT_DIR, 'data', 'tusimple', 'train_set', 'label_data.json'),
    'test_json': os.path.join(ROOT_DIR, 'data', 'tusimple', 'test_set', 'test_label.json'),
    'download_urls': {
        'train': 'https://s3.us-east-2.amazonaws.com/benchmark-frontend/datasets/1/train_set.zip',
        'test': 'https://s3.us-east-2.amazonaws.com/benchmark-frontend/datasets/1/test_set.zip',
    }
}

# 模型配置
MODEL_CONFIG = {
    'backbone': 'ENet',  # 可选: ResNet, ENet
    'embedding_dim': 4,  # 实例分割嵌入维度
    'num_lanes': 5,      # 最多检测的车道线数量
    'input_size': (256, 512),  # (height, width)
    'output_size': (256, 512),
}

# 训练配置
TRAIN_CONFIG = {
    'batch_size': 8,
    'num_epochs': 150,
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    'num_workers': 4,
    'save_dir': os.path.join(ROOT_DIR, 'checkpoints'),
    'log_dir': os.path.join(ROOT_DIR, 'logs'),
    'resume': None,  # 恢复训练的checkpoint路径
    'print_freq': 10,  # 打印频率
    'save_freq': 10,   # 保存频率（epoch）
}

# 损失函数权重
LOSS_CONFIG = {
    'binary_weight': 1.0,      # 二值分割损失权重
    'instance_weight': 1.0,    # 实例分割损失权重
    'discriminative_delta_v': 0.5,  # 方差损失margin
    'discriminative_delta_d': 3.0,  # 距离损失margin
}

# 评测配置
EVAL_CONFIG = {
    'checkpoint': None,  # 评测使用的checkpoint路径
    'batch_size': 8,
    'num_workers': 4,
    'save_visualizations': True,
    'visualization_dir': os.path.join(ROOT_DIR, 'visualizations'),
}

# 后处理配置
POSTPROCESS_CONFIG = {
    'min_cluster_size': 100,  # 最小聚类点数
    'dbscan_eps': 0.5,        # DBSCAN聚类参数
    'dbscan_min_samples': 100,
}
