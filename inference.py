"""
推理脚本 - 对单张图像或视频进行车道线检测
"""

import os
import sys
import argparse

import torch
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import LaneNet
from config.config import MODEL_CONFIG
from utils import embedding_post_process, fit_lane_lines, visualize_lanes


class LaneDetector:
    """车道线检测器"""
    
    def __init__(self, checkpoint_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # 构建模型
        self.model = LaneNet(embedding_dim=MODEL_CONFIG['embedding_dim']).to(self.device)
        
        # 加载checkpoint
        self.load_checkpoint(checkpoint_path)
        
        # 预处理
        self.transform = transforms.Compose([
            transforms.Resize(MODEL_CONFIG['input_size']),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.model.eval()
    
    def load_checkpoint(self, checkpoint_path):
        """加载checkpoint"""
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print("Checkpoint loaded successfully!")
    
    @torch.no_grad()
    def detect(self, image):
        """
        检测单张图像的车道线
        
        Args:
            image: PIL Image 或 numpy array (BGR)
        
        Returns:
            vis_image: 可视化图像
            lane_lines: 车道线列表
        """
        # 转换为PIL Image
        if isinstance(image, np.ndarray):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        
        original_size = image.size  # (width, height)
        
        # 预处理
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # 前向传播
        outputs = self.model(input_tensor)

        # 获取 HNet 预测参数
        hnet_params = [None]
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
        
        # 二值分割
        binary_seg = torch.argmax(outputs['binary_seg'], dim=1)[0].cpu().numpy()
        
        # 实例分割
        instance_seg = outputs['instance_seg'][0].permute(1, 2, 0).cpu().numpy()
        
        # 后处理
        instance_mask, num_lanes = embedding_post_process(
            instance_seg,
            binary_seg,
            delta_v=0.5,
            min_cluster_size=50
        )
        
        # 拟合车道线
        lane_lines = fit_lane_lines(instance_mask, num_lanes, hnet_matrix=hnet_params[0])
        
        # 调整车道线坐标到原始尺寸
        scale_x = original_size[0] / MODEL_CONFIG['input_size'][1]
        scale_y = original_size[1] / MODEL_CONFIG['input_size'][0]
        
        scaled_lane_lines = []
        for lane in lane_lines:
            scaled_lane = lane.copy()
            scaled_lane[:, 0] *= scale_x
            scaled_lane[:, 1] *= scale_y
            scaled_lane_lines.append(scaled_lane)
        
        # 可视化
        image_np = np.array(image.resize((original_size[0], original_size[1])))
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        vis_image = visualize_lanes(image_np, scaled_lane_lines)
        
        return vis_image, scaled_lane_lines


def detect_image(detector, image_path, output_path):
    """检测单张图像"""
    print(f"Processing {image_path}...")
    
    # 读取图像
    image = cv2.imread(image_path)
    
    # 检测
    vis_image, lane_lines = detector.detect(image)
    
    # 保存
    cv2.imwrite(output_path, vis_image)
    print(f"Saved to {output_path}")
    print(f"Detected {len(lane_lines)} lanes")


def detect_video(detector, video_path, output_path):
    """检测视频"""
    print(f"Processing {video_path}...")
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    
    # 获取视频属性
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Video info: {width}x{height} @ {fps}fps, {total_frames} frames")
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 检测
        vis_frame, _ = detector.detect(frame)
        
        # 写入
        out.write(vis_frame)
        
        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"Processed {frame_idx}/{total_frames} frames")
    
    cap.release()
    out.release()
    
    print(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Lane Detection Inference')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--input', type=str, required=True, help='Input image or video path')
    parser.add_argument('--output', type=str, required=True, help='Output path')
    parser.add_argument('--type', type=str, choices=['image', 'video'], default='image',
                        help='Input type')
    args = parser.parse_args()
    
    # 创建检测器
    detector = LaneDetector(args.checkpoint)
    
    # 检测
    if args.type == 'image':
        detect_image(detector, args.input, args.output)
    else:
        detect_video(detector, args.input, args.output)


if __name__ == '__main__':
    main()
