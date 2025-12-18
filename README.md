# LaneNet 车道线检测项目

基于LaneNet的端到端车道线检测系统，使用PyTorch实现。

## 项目特点

- ✅ 完整的LaneNet模型实现（ENet编码器-解码器架构）
- ✅ 二值分割 + 实例分割双分支设计
- ✅ 判别损失函数用于实例聚类
- ✅ TuSimple数据集自动下载
- ✅ 完整的训练、评测、推理流程
- ✅ 可视化工具
- ✅ 支持图像和视频检测

## 项目结构

```
self_driving_automobile/
├── config/
│   └── config.py           # 配置文件
├── data/
│   ├── __init__.py
│   ├── dataset.py          # 数据集加载器
│   └── download_dataset.py # 数据集下载脚本
├── models/
│   ├── __init__.py
│   ├── lanenet.py          # LaneNet模型
│   └── loss.py             # 损失函数
├── utils/
│   ├── __init__.py
│   └── lane_utils.py       # 车道线后处理工具
├── train.py                # 训练脚本
├── evaluate.py             # 评测脚本
├── inference.py            # 推理脚本
├── requirements.txt        # 依赖包
└── README.md              # 说明文档
```

## 安装依赖

### 1. 创建Python环境（推荐）

```bash
# 使用conda
conda create -n lanenet python=3.8
conda activate lanenet

# 或使用venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
```

### 2. 安装依赖包

```bash
pip install -r requirements.txt
```

主要依赖：
- PyTorch >= 1.10.0
- torchvision >= 0.11.0
- OpenCV
- scikit-learn
- matplotlib
- tqdm
- tensorboard

## 快速开始

### 步骤1：下载数据集

运行以下命令下载TuSimple数据集（约10GB）：

```bash
python data/download_dataset.py
```

数据集将自动下载到 `data/tusimple/` 目录。

**注意**：如果自动下载失败，可以手动下载：
- 训练集: https://s3.us-east-2.amazonaws.com/benchmark-frontend/datasets/1/train_set.zip
- 测试集: https://s3.us-east-2.amazonaws.com/benchmark-frontend/datasets/1/test_set.zip

下载后解压到 `data/tusimple/` 目录。

### 步骤2：训练模型

```bash
python train.py
```

训练参数可在 `config/config.py` 中修改：
- `batch_size`: 批次大小（默认8）
- `num_epochs`: 训练轮数（默认100）
- `learning_rate`: 学习率（默认0.001）

训练过程中会自动保存：
- Checkpoints: `checkpoints/`
- TensorBoard日志: `logs/`

查看训练日志：
```bash
tensorboard --logdir=logs
```

### 步骤3：评测模型

```bash
python evaluate.py --checkpoint checkpoints/best_model.pth
```

评测结果将显示：
- 平均准确率
- False Positives/Negatives
- 可视化结果保存在 `visualizations/` 目录

### 步骤4：推理

#### 图像检测

```bash
python inference.py \
    --checkpoint checkpoints/best_model.pth \
    --input /path/to/image.jpg \
    --output result.jpg \
    --type image
```

#### 视频检测

```bash
python inference.py \
    --checkpoint checkpoints/best_model.pth \
    --input /path/to/video.mp4 \
    --output result.mp4 \
    --type video
```

## 模型架构

### LaneNet

LaneNet采用编码器-解码器架构，包含两个分支：

1. **二值分割分支**：检测车道线区域（前景/背景）
2. **实例分割分支**：区分不同车道线（输出嵌入向量）

```
输入图像 (3, 256, 512)
    ↓
ENet编码器
    ↓
特征图 (128, 32, 64)
    ↓        ↓
二值解码器  实例解码器
    ↓        ↓
(2, 256, 512)  (4, 256, 512)
    ↓        ↓
二值分割    嵌入向量
```

### 损失函数

总损失 = 二值分割损失 + 实例分割损失

1. **二值分割损失**：交叉熵损失
2. **实例分割损失**：判别损失（Discriminative Loss）
   - 方差损失：拉近同一实例内的点
   - 距离损失：推远不同实例的中心
   - 正则化损失：让中心接近原点

## 配置说明

主要配置在 `config/config.py`：

### 数据集配置
```python
DATASET_CONFIG = {
    'name': 'TuSimple',
    'root_dir': 'data/tusimple',
    'input_size': (256, 512),  # (height, width)
}
```

### 模型配置
```python
MODEL_CONFIG = {
    'backbone': 'ENet',
    'embedding_dim': 4,      # 嵌入维度
    'input_size': (256, 512),
}
```

### 训练配置
```python
TRAIN_CONFIG = {
    'batch_size': 8,
    'num_epochs': 100,
    'learning_rate': 0.001,
    'save_freq': 10,  # 每10个epoch保存一次
}
```

### 损失权重
```python
LOSS_CONFIG = {
    'binary_weight': 1.0,
    'instance_weight': 0.5,
    'discriminative_delta_v': 0.5,
    'discriminative_delta_d': 3.0,
}
```

## 参考文献

1. LaneNet论文: [Towards End-to-End Lane Detection: an Instance Segmentation Approach](https://arxiv.org/abs/1802.05591)
2. ENet: [ENet: A Deep Neural Network Architecture for Real-time Semantic Segmentation](https://arxiv.org/abs/1606.02147)
3. Discriminative Loss: [Semantic Instance Segmentation with a Discriminative Loss Function](https://arxiv.org/abs/1708.02551)

## 许可证

MIT License

