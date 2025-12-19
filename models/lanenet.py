"""
LaneNet模型主架构
包含编码器、解码器和两个分支（二值分割、实例分割）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InitialBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super(InitialBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels - in_channels, 3, stride=2, padding=1, bias=False)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        main = self.conv(x)
        side = self.pool(x)
        x = torch.cat([main, side], dim=1)
        x = self.bn(x)
        x = self.relu(x)
        return x

    
class BottleneckModule(nn.Module):
    def __init__(self, in_ch, out_ch, module_type, padding = 1, dilated = 0, asymmetric = 5, dropout_prob = 0):
        super(BottleneckModule, self).__init__()
        self.input_channel = in_ch
        self.activate = nn.PReLU()

        self.module_type = module_type
        if self.module_type == 'downsampling':
            self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size = 2, stride = 2),
                nn.BatchNorm2d(out_ch),
                nn.PReLU(),
                nn.Conv2d(out_ch, out_ch, kernel_size = 3, stride=1, padding=padding),
                nn.BatchNorm2d(out_ch),
                nn.PReLU(),
                nn.Conv2d(out_ch, out_ch, kernel_size = 1),
                nn.BatchNorm2d(out_ch),
                nn.PReLU(),
                nn.Dropout2d(p=dropout_prob)
            )
        elif self.module_type == 'upsampling':
            self.maxunpool = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size = 1),
                nn.BatchNorm2d(out_ch),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)    # Use upsample instead of maxunpooling
            )
            
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size = 1),
                nn.BatchNorm2d(out_ch),
                nn.PReLU(),
                nn.ConvTranspose2d(out_ch, out_ch, kernel_size=2, stride=2),
                nn.BatchNorm2d(out_ch),
                nn.PReLU(),
                nn.Conv2d(out_ch, out_ch, kernel_size = 1),
                nn.BatchNorm2d(out_ch),
                nn.PReLU(),
                nn.Dropout2d(p=dropout_prob)
            )
        elif self.module_type == 'regular':
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size = 1),
                nn.BatchNorm2d(out_ch),
                nn.PReLU(),
                nn.Conv2d(out_ch, out_ch, kernel_size = 3, stride=1, padding=padding),
                nn.BatchNorm2d(out_ch),
                nn.PReLU(),
                nn.Conv2d(out_ch, out_ch, kernel_size = 1),
                nn.BatchNorm2d(out_ch),
                nn.PReLU(),
                nn.Dropout2d(p=dropout_prob)
            )
        elif self.module_type == 'asymmetric':
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size = 1),
                nn.BatchNorm2d(out_ch),
                nn.PReLU(),
                nn.Conv2d(out_ch, out_ch, (asymmetric, 1), stride=1, padding=(padding, 0)),
                nn.Conv2d(out_ch, out_ch, (1, asymmetric), stride=1, padding=(0, padding)),
                nn.BatchNorm2d(out_ch),
                nn.PReLU(),
                nn.Conv2d(out_ch, out_ch, kernel_size = 1),
                nn.BatchNorm2d(out_ch),
                nn.PReLU(),
                nn.Dropout2d(p=dropout_prob)
            )
        elif self.module_type == 'dilated':
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size = 1),
                nn.BatchNorm2d(out_ch),
                nn.PReLU(),
                nn.Conv2d(out_ch, out_ch, kernel_size = 3, stride=1, padding=padding, dilation=dilated),
                nn.BatchNorm2d(out_ch),
                nn.PReLU(),
                nn.Conv2d(out_ch, out_ch, kernel_size = 1),
                nn.BatchNorm2d(out_ch),
                nn.PReLU(),
                nn.Dropout2d(p=dropout_prob)
            )
        else:
            raise Exception("Module Type error")

    def forward(self, x):
        if self.module_type == 'downsampling':
            conv_branch = self.conv(x)
            maxp_branch = self.maxpool(x)
            bs, conv_ch, h, w = conv_branch.size()
            maxp_ch = maxp_branch.size()[1]
            padding = torch.zeros(bs, conv_ch - maxp_ch, h, w).to(x.device)

            maxp_branch = torch.cat([maxp_branch, padding], 1)
            output = maxp_branch + conv_branch
        elif self.module_type == 'upsampling':
            conv_branch = self.conv(x)
            maxunp_branch = self.maxunpool(x)
            output = maxunp_branch + conv_branch
        else:
            output = self.conv(x) + x
        
        return self.activate(output)


class ENetEncoder(nn.Module):
    """ENet编码器"""
    def __init__(self):
        super(ENetEncoder, self).__init__()
        
        # Stage 0
        self.initial = InitialBlock(3, 16)
        
        # Stage 1
        self.bottleneck1_0 = BottleneckModule(16, 64,  module_type = 'downsampling')
        self.bottleneck1_1 = BottleneckModule(64, 64, module_type = 'regular')
        self.bottleneck1_2 = BottleneckModule(64, 64, module_type = 'regular')
        self.bottleneck1_3 = BottleneckModule(64, 64, module_type = 'regular')
        
        # Stage 2
        self.bottleneck2_0 = BottleneckModule(64, 128,  module_type = 'downsampling')
        self.bottleneck2_1 = BottleneckModule(128, 128, module_type = 'regular')
        self.bottleneck2_2 = BottleneckModule(128, 128, module_type = 'regular')
        self.bottleneck2_3 = BottleneckModule(128, 128, module_type = 'regular')
        
    def forward(self, x):
        # Stage 0
        x = self.initial(x)
        
        # Stage 1
        x = self.bottleneck1_0(x)
        x = self.bottleneck1_1(x)
        x = self.bottleneck1_2(x)
        x = self.bottleneck1_3(x)
        
        # Stage 2
        x = self.bottleneck2_0(x)
        x = self.bottleneck2_1(x)
        x = self.bottleneck2_2(x)
        x = self.bottleneck2_3(x)
        
        return x


class ENetDecoder(nn.Module):
    """ENet解码器"""
    def __init__(self, num_classes):
        super(ENetDecoder, self).__init__()
        
        # Stage 4
        self.bottleneck4_0 = BottleneckModule(128, 64, module_type = 'upsampling')
        self.bottleneck4_1 = BottleneckModule(64, 64, module_type = 'regular')
        
        # Stage 5
        self.bottleneck5_0 = BottleneckModule(64, 16, module_type = 'upsampling')
        self.bottleneck5_1 = BottleneckModule(16, 16, module_type = 'regular')
        
        # Final upsampling
        self.upsample = nn.ConvTranspose2d(16, num_classes, 2, stride=2, bias=False)
        
    def forward(self, x):
        # Stage 4
        x = self.bottleneck4_0(x)
        x = self.bottleneck4_1(x)
        
        # Stage 5
        x = self.bottleneck5_0(x)
        x = self.bottleneck5_1(x)
        
        # Final upsampling
        x = self.upsample(x)
        
        return x


class LaneNet(nn.Module):
    """
    LaneNet主模型
    包含共享编码器和两个解码分支：
    1. 二值分割分支：检测车道线区域
    2. 实例分割分支：区分不同车道线
    """
    def __init__(self, embedding_dim=4, use_hnet=True):
        super(LaneNet, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.use_hnet = use_hnet
        
        # 共享编码器
        self.encoder = ENetEncoder()
        
        # 二值分割分支
        self.binary_decoder = ENetDecoder(num_classes=2)
        
        # 实例分割分支
        self.instance_decoder = ENetDecoder(num_classes=embedding_dim)

        # HNet分支
        if self.use_hnet:
            self.hnet = HNet()
        
    def forward(self, x):
        # 共享编码
        encoded = self.encoder(x)
        
        # 二值分割
        binary_output = self.binary_decoder(encoded)
        
        # 实例分割（输出嵌入向量）
        instance_output = self.instance_decoder(encoded)

        output = {
            'binary_seg': binary_output,      # [B, 2, H, W]
            'instance_seg': instance_output   # [B, embedding_dim, H, W]
        }
        
        # HNet输出
        if self.use_hnet:
            hnet_output = self.hnet(x)
            output['hnet_params'] = hnet_output
            
        return output


def test_model():
    """测试模型"""
    model = LaneNet(embedding_dim=4)
    x = torch.randn(2, 3, 256, 512)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Binary seg shape: {output['binary_seg'].shape}")
    print(f"Instance seg shape: {output['instance_seg'].shape}")
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

class HNet(nn.Module):
    """
    HNet: Homography Network
    用于预测透视变换矩阵的参数，帮助更好地拟合车道线
    """
    def __init__(self):
        super(HNet, self).__init__()
        
        # 简单的CNN特征提取
        self.features = nn.Sequential(
            # Conv1
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.PReLU(), nn.MaxPool2d(2),
            # Conv2
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.PReLU(), nn.MaxPool2d(2),
            # Conv3
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.PReLU(), nn.MaxPool2d(2),
            # Conv4
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.PReLU(), nn.MaxPool2d(2)
        )
        
        # 全连接层预测参数
        # 假设输入图像大小为 256x512，经过4次下采样(2^4=16)后变为 16x32
        # 64 channels * 16 height * 32 width
        self.fc = nn.Sequential(
            nn.Linear(64 * 16 * 32, 1024),
            nn.BatchNorm1d(1024),
            nn.PReLU(),
            nn.Linear(1024, 6)  # 输出6个参数用于构建变换矩阵 (3x3矩阵，最后一行固定)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    test_model()
