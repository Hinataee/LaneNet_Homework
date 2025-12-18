#!/bin/bash
# 一键运行脚本 - 完整流程

echo "=========================================="
echo "LaneNet 车道线检测 - 完整流程"
echo "=========================================="

# 1. 检查Python环境
echo ""
echo "[1/5] 检查Python环境..."
python --version
if [ $? -ne 0 ]; then
    echo "错误：未找到Python！请先安装Python 3.8+"
    exit 1
fi

# 2. 安装依赖
echo ""
echo "[2/5] 安装依赖包..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "错误：依赖包安装失败！"
    exit 1
fi

# 3. 下载数据集
echo ""
echo "[3/5] 下载数据集..."
if [ ! -f "data/tusimple/train_set/label_data.json" ]; then
    python data/download_dataset.py
    if [ $? -ne 0 ]; then
        echo "警告：数据集下载失败，请手动下载！"
    fi
else
    echo "数据集已存在，跳过下载"
fi

# 4. 训练模型
echo ""
echo "[4/5] 开始训练模型..."
echo "训练可能需要数小时，请耐心等待..."
python train.py
if [ $? -ne 0 ]; then
    echo "错误：训练失败！"
    exit 1
fi

# 5. 评测模型
echo ""
echo "[5/5] 评测模型..."
python evaluate.py --checkpoint checkpoints/best_model.pth
if [ $? -ne 0 ]; then
    echo "错误：评测失败！"
    exit 1
fi

echo ""
echo "=========================================="
echo "完成！"
echo "=========================================="
echo "训练好的模型: checkpoints/best_model.pth"
echo "可视化结果: visualizations/"
echo ""
echo "使用推理脚本检测图像："
echo "python inference.py --checkpoint checkpoints/best_model.pth --input image.jpg --output result.jpg --type image"
