"""
LaneNet损失函数
包含二值分割损失和实例分割损失（判别损失）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiscriminativeLoss(nn.Module):
    """
    判别损失用于实例分割
    论文: Semantic Instance Segmentation with a Discriminative Loss Function
    """
    def __init__(self, delta_v=0.5, delta_d=3.0, alpha=1.0, beta=1.0, gamma=0.001):
        """
        Args:
            delta_v: 方差损失的margin
            delta_d: 距离损失的margin
            alpha: 方差损失权重
            beta: 距离损失权重
            gamma: 正则化损失权重
        """
        super(DiscriminativeLoss, self).__init__()
        self.delta_v = delta_v
        self.delta_d = delta_d
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
    def forward(self, embedding, instance_mask):
        """
        Args:
            embedding: [B, D, H, W] 嵌入向量
            instance_mask: [B, H, W] 实例标签 (0表示背景，1,2,3...表示不同车道线)
        """
        batch_size, embedding_dim, height, width = embedding.shape
        
        # 重塑为 [B, H*W, D]
        embedding = embedding.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, embedding_dim)
        instance_mask = instance_mask.view(batch_size, -1)
        
        loss_var_list = []
        loss_dist_list = []
        loss_reg_list = []
        
        for b in range(batch_size):
            embedding_b = embedding[b]  # [H*W, D]
            instance_mask_b = instance_mask[b]  # [H*W]
            
            # 获取唯一的实例ID（排除背景0）
            unique_instances = torch.unique(instance_mask_b)
            unique_instances = unique_instances[unique_instances != 0]
            
            num_instances = len(unique_instances)
            
            if num_instances == 0:
                # 如果没有实例，跳过
                continue
                
            # 计算每个实例的中心
            centers = []
            for instance_id in unique_instances:
                mask = (instance_mask_b == instance_id)
                if mask.sum() == 0:
                    continue
                instance_embedding = embedding_b[mask]  # [N, D]
                center = torch.mean(instance_embedding, dim=0)  # [D]
                centers.append(center)
                
            if len(centers) == 0:
                continue
                
            centers = torch.stack(centers)  # [num_instances, D]
            
            # 1. 方差损失（拉近同一实例内的点）
            loss_var = 0.0
            for i, instance_id in enumerate(unique_instances):
                mask = (instance_mask_b == instance_id)
                if mask.sum() == 0:
                    continue
                instance_embedding = embedding_b[mask]  # [N, D]
                center = centers[i]  # [D]
                
                # 计算到中心的距离
                distance = torch.norm(instance_embedding - center, dim=1)  # [N]
                distance = torch.clamp(distance - self.delta_v, min=0.0) ** 2
                loss_var += torch.mean(distance)
                
            loss_var = loss_var / num_instances
            loss_var_list.append(loss_var)
            
            # 2. 距离损失（推远不同实例的中心）
            loss_dist = 0.0
            for i in range(num_instances):
                for j in range(i + 1, num_instances):
                    center_i = centers[i]
                    center_j = centers[j]
                    distance = torch.norm(center_i - center_j)
                    distance = torch.clamp(2 * self.delta_d - distance, min=0.0) ** 2
                    loss_dist += distance
                    
            if num_instances > 1:
                loss_dist = loss_dist / (num_instances * (num_instances - 1) / 2)
            loss_dist_list.append(loss_dist)
            
            # 3. 正则化损失（让中心接近原点）
            loss_reg = torch.mean(torch.norm(centers, dim=1))
            loss_reg_list.append(loss_reg)
            
        # 汇总损失
        if len(loss_var_list) > 0:
            loss_var = torch.mean(torch.stack(loss_var_list))
        else:
            loss_var = torch.tensor(0.0, device=embedding.device)
            
        if len(loss_dist_list) > 0:
            loss_dist = torch.mean(torch.stack(loss_dist_list))
        else:
            loss_dist = torch.tensor(0.0, device=embedding.device)
            
        if len(loss_reg_list) > 0:
            loss_reg = torch.mean(torch.stack(loss_reg_list))
        else:
            loss_reg = torch.tensor(0.0, device=embedding.device)
            
        total_loss = self.alpha * loss_var + self.beta * loss_dist + self.gamma * loss_reg
        
        return total_loss, loss_var, loss_dist, loss_reg
    
class HNetLoss(nn.Module):
    """
    HNet损失函数
    通过最小化变换后车道线点的多项式拟合误差来训练HNet
    """
    def __init__(self, order=3):
        super(HNetLoss, self).__init__()
        self.order = order # 多项式阶数，通常为2或3
        
    def forward(self, hnet_params, instance_label):
        """
        Args:
            hnet_params: [B, 6] 预测的变换矩阵参数
            instance_label: [B, H, W] 实例标签
        """
        batch_size = hnet_params.size(0)
        loss = 0.0
        valid_lanes = 0
        
        # 构建变换矩阵 H: [B, 3, 3]
        # 预测的是前两行，最后一行固定为 [0, 0, 1]
        H = torch.zeros(batch_size, 3, 3, device=hnet_params.device)
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
        
        for b in range(batch_size):
            label = instance_label[b]
            H_mat = H[b]
            H_mat_inv = torch.inverse(H_mat)
            
            unique_ids = torch.unique(label)
            unique_ids = unique_ids[unique_ids != 0] # 排除背景
            
            if len(unique_ids) == 0:
                continue
                
            for lane_id in unique_ids:
                # 获取车道线点坐标 (y, x) -> (row, col)
                ys, xs = torch.where(label == lane_id)
                
                if len(xs) < self.order + 1: # 点太少无法拟合
                    continue
                
                # 构建齐次坐标 [x, y, 1]^T
                # 注意：图像坐标通常 x对应列，y对应行
                ones = torch.ones_like(xs)
                P = torch.stack([xs.float(), ys.float(), ones.float()], dim=0) # [3, N]
                
                # 透视变换 P' = H * P
                P_prime = torch.matmul(H_mat, P) # [3, N]
                
                # 归一化 (x', y')
                # 避免除以0
                denominator = P_prime[2, :]
                denominator = torch.where(torch.abs(denominator) < 1e-5, 
                                        torch.ones_like(denominator) * 1e-5, 
                                        denominator)
                
                x_prime = P_prime[0, :] / denominator
                y_prime = P_prime[1, :] / denominator
                
                # 最小二乘法拟合 x' = f(y')
                # 目标：找到系数 w 使得 ||Yw - x'||^2 最小
                # 多项式: x = a*y^2 + b*y + c (order=2)
                # 设计矩阵 Y = [y^2, y, 1]
                
                Y_list = []
                for i in range(self.order, -1, -1):
                    Y_list.append(y_prime ** i)
                Y = torch.stack(Y_list, dim=1) # [N, order+1]
                
                # 使用 torch.linalg.lstsq 求解
                try:
                    # lstsq 返回 (solution, residuals, rank, singular_values)
                    result = torch.linalg.lstsq(Y, x_prime)
                    w = result.solution
                    
                    # 计算预测值
                    x_pred = torch.matmul(Y, w)
                    
                    # 计算MSE损失
                    lane_loss = torch.mean((x_prime - x_pred) ** 2) * torch.mean(torch.abs(denominator)) 
                    loss += lane_loss
                    valid_lanes += 1
                except Exception:
                    # 矩阵奇异等情况
                    continue
                    
        if valid_lanes > 0:
            return loss / valid_lanes
        else:
            # 返回带梯度的0，避免计算图断裂
            return torch.tensor(0.0, device=hnet_params.device, requires_grad=True)



class LaneNetLoss(nn.Module):
    """LaneNet总损失"""
    def __init__(self, delta_v=0.5, delta_d=3.0, binary_weight=1.0, instance_weight=0.5, hnet_weight=1.0):
        super(LaneNetLoss, self).__init__()
        self.binary_weight = binary_weight
        self.instance_weight = instance_weight
        self.hnet_weight = hnet_weight
        
        # 二值分割损失（交叉熵）
        self.binary_loss = nn.CrossEntropyLoss()
        
        # 实例分割损失（判别损失）
        self.instance_loss = DiscriminativeLoss(delta_v=delta_v, delta_d=delta_d)

        # HNet损失
        self.hnet_loss = HNetLoss()
   
        
    def forward(self, output, binary_label, instance_label):
        """
        Args:
            output: 模型输出字典，包含 'binary_seg' 和 'instance_seg'
            binary_label: [B, H, W] 二值标签
            instance_label: [B, H, W] 实例标签
        """
        losses = {}
        # 二值分割损失
        binary_seg = output['binary_seg']  # [B, 2, H, W]
        loss_binary = self.binary_loss(binary_seg, binary_label.long())
        losses['binary_loss'] = loss_binary
 
        # 实例分割损失
        instance_seg = output['instance_seg']  # [B, D, H, W]
        loss_instance, loss_var, loss_dist, loss_reg = self.instance_loss(instance_seg, instance_label)
        losses['instance_loss'] = loss_instance
        losses['var_loss'] = loss_var
        losses['dist_loss'] = loss_dist
        losses['reg_loss'] = loss_reg
        
        # 3. HNet损失 (如果存在)
        loss_hnet = torch.tensor(0.0, device=binary_seg.device)
        if 'hnet_params' in output:
            loss_hnet = self.hnet_loss(output['hnet_params'], instance_label)
        losses['hnet_loss'] = loss_hnet
        
        # 总损失
        total_loss = (self.binary_weight * loss_binary + 
                      self.instance_weight * loss_instance + 
                      self.hnet_weight * loss_hnet)
        
        losses['total_loss'] = total_loss
        
        return losses
    


if __name__ == '__main__':
    # 测试损失函数
    criterion = LaneNetLoss()
    
    # 模拟数据
    batch_size = 2
    output = {
        'binary_seg': torch.randn(batch_size, 2, 256, 512),
        'instance_seg': torch.randn(batch_size, 4, 256, 512)
    }
    binary_label = torch.randint(0, 2, (batch_size, 256, 512))
    instance_label = torch.randint(0, 5, (batch_size, 256, 512))
    
    losses = criterion(output, binary_label, instance_label)
    
    print("Loss components:")
    for key, value in losses.items():
        print(f"{key}: {value.item():.4f}")
