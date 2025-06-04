import torch
import torch.nn as nn
import torch.nn.functional as F
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
import numpy as np

class DDPMModel:
    """经典DDPM模型封装类"""
    
    def __init__(self, 
                 image_size=32,
                 channels=3,
                 dim=64,
                 dim_mults=(1, 2, 4, 8),
                 timesteps=1000,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        
        self.device = device
        self.image_size = image_size
        self.channels = channels
        self.timesteps = timesteps
        
        # 初始化U-Net模型
        self.unet = Unet(
            dim=dim,
            channels=channels,
            dim_mults=dim_mults
        ).to(device)
        
        # 初始化高斯扩散过程
        self.diffusion = GaussianDiffusion(
            self.unet,
            image_size=image_size,
            timesteps=timesteps,
            sampling_timesteps=min(timesteps, 250)  # 确保采样步数不超过总时间步数
        ).to(device)
        
        print(f"DDPM模型初始化完成:")
        print(f"- 图像尺寸: {image_size}x{image_size}")
        print(f"- 通道数: {channels}")
        print(f"- 时间步数: {timesteps}")
        print(f"- 设备: {device}")
        print(f"- 模型参数数量: {self.count_parameters():,}")
    
    def count_parameters(self):
        """计算模型参数数量"""
        return sum(p.numel() for p in self.unet.parameters() if p.requires_grad)
    
    def train_step(self, batch):
        """单步训练"""
        batch = batch.to(self.device)
        loss = self.diffusion(batch)
        return loss
    
    def sample(self, batch_size=16, return_all_timesteps=False):
        """生成样本"""
        with torch.no_grad():
            samples = self.diffusion.sample(
                batch_size=batch_size,
                return_all_timesteps=return_all_timesteps
            )
        return samples
    
    def save_model(self, path):
        """保存模型"""
        torch.save({
            'unet_state_dict': self.unet.state_dict(),
            'image_size': self.image_size,
            'channels': self.channels,
            'timesteps': self.timesteps
        }, path)
        print(f"模型已保存到: {path}")
    
    def load_model(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.unet.load_state_dict(checkpoint['unet_state_dict'])
        print(f"模型已从 {path} 加载")
    
    def interpolate(self, x1, x2, num_steps=10):
        """在两个图像之间进行插值"""
        with torch.no_grad():
            # 线性插值
            alphas = torch.linspace(0, 1, num_steps, device=self.device)
            interpolated_samples = []
            
            for alpha in alphas:
                # 在图像空间中直接插值
                interpolated = (1 - alpha) * x1 + alpha * x2
                interpolated_samples.append(interpolated)
            
            # 拼接成batch，保持4D张量格式 (batch_size, channels, height, width)
            return torch.cat(interpolated_samples, dim=0) 