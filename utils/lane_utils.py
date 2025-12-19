"""
车道线聚类和后处理
"""

import numpy as np
from sklearn.cluster import DBSCAN
import cv2


def embedding_post_process(embedding, binary_seg, delta_v=0.5, min_cluster_size=100):
    """
    对实例分割嵌入向量进行聚类
    
    Args:
        embedding: [H, W, D] 嵌入向量
        binary_seg: [H, W] 二值分割结果
        delta_v: 聚类阈值
        min_cluster_size: 最小聚类点数
    
    Returns:
        instance_mask: [H, W] 实例分割结果
        num_lanes: 检测到的车道线数量
    """
    height, width, embedding_dim = embedding.shape
    
    # 获取车道线像素点
    lane_pixels = binary_seg > 0
    if lane_pixels.sum() == 0:
        return np.zeros((height, width), dtype=np.uint8), 0
    
    # 提取车道线点的嵌入向量
    lane_embedding = embedding[lane_pixels]  # [N, D]
    lane_coords = np.column_stack(np.where(lane_pixels))  # [N, 2]
    
    # DBSCAN聚类
    db = DBSCAN(eps=delta_v, min_samples=min_cluster_size)
    labels = db.fit_predict(lane_embedding)
    
    # 构建实例mask
    instance_mask = np.zeros((height, width), dtype=np.uint8)
    
    # 过滤噪声点（label=-1）
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels >= 0]
    
    for idx, label in enumerate(unique_labels):
        cluster_mask = labels == label
        cluster_coords = lane_coords[cluster_mask]
        
        # 填充实例mask
        for coord in cluster_coords:
            instance_mask[coord[0], coord[1]] = idx + 1
    
    return instance_mask, len(unique_labels)


def fit_lane_lines(instance_mask, num_lanes, hnet_matrix=None):
    """
    拟合车道线多项式
    
    Args:
        instance_mask: [H, W] 实例分割结果
        num_lanes: 车道线数量
        hnet_matrix: [3, 3] 透视变换矩阵 (可选)
       
    Returns:
        lane_lines: 车道线点集列表
    """
    lane_lines = []
    
    # 如果提供了H矩阵，计算逆矩阵用于还原
    H_inv = None
    if hnet_matrix is not None:
        try:
            H_inv = np.linalg.inv(hnet_matrix)
        except np.linalg.LinAlgError:
            print("Warning: HNet matrix is singular, falling back to regular fit.")
            hnet_matrix = None


    for lane_id in range(1, num_lanes + 1):
        # 获取该车道线的点
        lane_mask = instance_mask == lane_id
        coords = np.column_stack(np.where(lane_mask))
        
        if len(coords) < 10:
            continue

        if hnet_matrix is not None:
            # try:
            # 构建齐次坐标 [x, y, 1]
            # 注意：图像坐标 x对应列(coords[:, 1]), y对应行(coords[:, 0])
            if hasattr(hnet_matrix, 'detach'):
                hnet_matrix = hnet_matrix.detach().cpu().numpy()
            xs = coords[:, 1]
            ys = coords[:, 0]
            ones = np.ones_like(xs)
            P = np.vstack((xs, ys, ones)) # [3, N]
            
            # 透视变换 P' = H * P
            P_prime = hnet_matrix @ P
            # 归一化
            denominator = P_prime[2, :]
            denominator[np.abs(denominator) < 1e-6] = 1e-6 # 避免除零
            xs_proj = P_prime[0, :] / denominator
            ys_proj = P_prime[1, :] / denominator
            print("Denominator for original coords:", denominator)
            
            # 在 BEV 空间拟合: x' = f(y')
            # 使用 2阶 或 3阶 多项式
            poly_params = np.polyfit(ys_proj, xs_proj, deg=3)
            poly_func = np.poly1d(poly_params)
            
            # 生成平滑点
            y_min, y_max = np.min(ys_proj), np.max(ys_proj)
            plot_ys = np.linspace(y_min, y_max, num=50)
            plot_xs = poly_func(plot_ys)
            
            # 反变换回原图: P = H_inv * P'
            bev_coords = np.vstack((plot_xs, plot_ys, np.ones_like(plot_xs)))
            orig_coords = H_inv @ bev_coords
            
            # 归一化
            denom_orig = orig_coords[2, :]
            denom_orig[np.abs(denom_orig) < 1e-6] = 1e-6
            final_xs = orig_coords[0, :] / denom_orig
            final_ys = orig_coords[1, :] / denom_orig
            
            # 过滤出图像范围内的点
            h, w = instance_mask.shape
            valid_mask = (final_xs >= 0) & (final_xs < w) & (final_ys >= 0) & (final_ys < h)
            
            if np.sum(valid_mask) > 2:
                lane_points = np.column_stack([final_xs[valid_mask], final_ys[valid_mask]])
                lane_lines.append(lane_points)
            else:
                # 如果反变换后点都跑出去了，回退到原始点
                raise ValueError("All points out of bounds after inverse transform")
                    
    
        else:
            # 按y坐标排序
            coords = coords[coords[:, 0].argsort()]
        
            # 二次多项式拟合
            try:
                z = np.polyfit(coords[:, 0], coords[:, 1], 2)
                p = np.poly1d(z)
                
                # 生成拟合后的点
                y_coords = np.arange(coords[:, 0].min(), coords[:, 0].max(), 1)
                x_coords = p(y_coords)
                
                # 过滤出有效点
                valid_mask = (x_coords >= 0) & (x_coords < instance_mask.shape[1])
                y_coords = y_coords[valid_mask]
                x_coords = x_coords[valid_mask]
                
                lane_points = np.column_stack([x_coords, y_coords])
                lane_lines.append(lane_points)
            except:
                # 拟合失败，使用原始点
                lane_points = np.column_stack([coords[:, 1], coords[:, 0]])
                lane_lines.append(lane_points)
        
    return lane_lines


def visualize_lanes(image, lane_lines, alpha=0.7):
    """
    可视化车道线
    
    Args:
        image: [H, W, 3] RGB图像
        lane_lines: 车道线点集列表
        alpha: 叠加透明度
    
    Returns:
        vis_image: 可视化图像
    """
    vis_image = image.copy()
    overlay = image.copy()
    
    # 不同车道线用不同颜色
    colors = [
        (255, 0, 0),    # 红色
        (0, 255, 0),    # 绿色
        (0, 0, 255),    # 蓝色
        (255, 255, 0),  # 黄色
        (255, 0, 255),  # 品红
        (0, 255, 255),  # 青色
    ]
    
    for idx, lane_points in enumerate(lane_lines):
        if len(lane_points) < 2:
            continue
        
        color = colors[idx % len(colors)]
        lane_points = lane_points.astype(np.int32)
        
        # 绘制车道线
        cv2.polylines(overlay, [lane_points], False, color, thickness=5)
    
    # 叠加
    vis_image = cv2.addWeighted(vis_image, 1 - alpha, overlay, alpha, 0)
    
    return vis_image


if __name__ == '__main__':
    # 测试聚类
    import torch
    
    height, width = 256, 512
    embedding_dim = 4
    
    # 模拟嵌入向量和二值分割
    embedding = torch.randn(1, embedding_dim, height, width)
    binary_seg = torch.randint(0, 2, (1, height, width))
    
    # 转换为numpy
    embedding_np = embedding[0].permute(1, 2, 0).cpu().numpy()
    binary_seg_np = binary_seg[0].cpu().numpy()
    
    # 后处理
    instance_mask, num_lanes = embedding_post_process(embedding_np, binary_seg_np)
    
    print(f"Detected {num_lanes} lanes")
    print(f"Instance mask shape: {instance_mask.shape}")
    print(f"Unique instances: {np.unique(instance_mask)}")
