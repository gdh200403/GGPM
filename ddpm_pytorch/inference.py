"""
DDPM推理脚本
"""
import os
import torch
import argparse
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np

from model import UNet
from ddpm import DDPM
from config import Config


def load_model(checkpoint_path):
    """加载训练好的模型"""
    checkpoint = torch.load(checkpoint_path, map_location=Config.device)
    
    # 创建模型
    model = UNet(
        in_channels=Config.channels,
        dim=Config.dim,
        dim_mults=Config.dim_mults
    ).to(Config.device)
    
    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 创建DDPM
    ddpm = DDPM(
        model=model,
        timesteps=Config.timesteps,
        beta_start=Config.beta_start,
        beta_end=Config.beta_end,
        beta_schedule=Config.beta_schedule,
        device=Config.device
    )
    
    print(f"模型加载成功，训练轮数: {checkpoint['epoch']}")
    return ddpm


def generate_samples(ddpm, num_samples=16, save_path='generated_samples.png'):
    """生成样本图像"""
    print(f"生成 {num_samples} 个样本...")
    
    with torch.no_grad():
        # 生成样本
        samples = ddpm.sample((num_samples, Config.channels, Config.image_size, Config.image_size))
        
        # 反标准化到[0, 1]
        samples = (samples + 1) / 2
        samples = torch.clamp(samples, 0, 1)
        
        # 保存图像网格
        save_image(
            samples,
            save_path,
            nrow=int(np.sqrt(num_samples)),
            normalize=False
        )
        
        print(f"样本已保存到: {save_path}")
        return samples


def generate_process_visualization(ddpm, save_path='generation_process.png'):
    """可视化生成过程"""
    print("生成过程可视化...")
    
    with torch.no_grad():
        # 生成一个样本并返回所有时间步
        samples = ddpm.sample(
            (1, Config.channels, Config.image_size, Config.image_size),
            return_all_timesteps=True
        )
        
        # 选择一些时间步进行可视化
        timesteps_to_show = [0, 50, 100, 200, 500, 999]
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        axes = axes.flatten()
        
        for i, t in enumerate(timesteps_to_show):
            img = samples[0, t].cpu()
            # 反标准化
            img = (img + 1) / 2
            img = torch.clamp(img, 0, 1)
            
            # 转换为numpy
            if img.shape[0] == 3:  # RGB
                img = img.permute(1, 2, 0).numpy()
            else:  # 灰度
                img = img.squeeze().numpy()
            
            axes[i].imshow(img)
            axes[i].set_title(f'Step {Config.timesteps - t}')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"生成过程可视化已保存到: {save_path}")


def interpolate_samples(ddpm, save_path='interpolation.png'):
    """在潜在空间中插值"""
    print("生成插值样本...")
    
    with torch.no_grad():
        # 生成两个随机噪声作为起点和终点
        device = next(ddpm.model.parameters()).device
        
        # 起始和结束噪声
        noise1 = torch.randn(1, Config.channels, Config.image_size, Config.image_size, device=device)
        noise2 = torch.randn(1, Config.channels, Config.image_size, Config.image_size, device=device)
        
        # 插值系数
        alphas = torch.linspace(0, 1, 8)
        interpolated_samples = []
        
        for alpha in alphas:
            # 在噪声空间中插值
            interpolated_noise = (1 - alpha) * noise1 + alpha * noise2
            
            # 从插值噪声生成样本
            sample = interpolated_noise.clone()
            for i in reversed(range(0, ddpm.timesteps)):
                t = torch.full((1,), i, device=device, dtype=torch.long)
                sample = ddpm.p_sample(sample, t)
            
            interpolated_samples.append(sample)
        
        # 拼接所有插值样本
        interpolated_samples = torch.cat(interpolated_samples, dim=0)
        
        # 反标准化
        interpolated_samples = (interpolated_samples + 1) / 2
        interpolated_samples = torch.clamp(interpolated_samples, 0, 1)
        
        # 保存
        save_image(
            interpolated_samples,
            save_path,
            nrow=8,
            normalize=False
        )
        
        print(f"插值样本已保存到: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='DDPM推理脚本')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--num_samples', type=int, default=16, help='生成样本数量')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='输出目录')
    parser.add_argument('--mode', type=str, default='sample', 
                       choices=['sample', 'process', 'interpolate', 'all'],
                       help='生成模式')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载模型
    ddpm = load_model(args.checkpoint)
    
    if args.mode == 'sample' or args.mode == 'all':
        # 生成样本
        save_path = os.path.join(args.output_dir, 'generated_samples.png')
        generate_samples(ddpm, args.num_samples, save_path)
    
    if args.mode == 'process' or args.mode == 'all':
        # 生成过程可视化
        save_path = os.path.join(args.output_dir, 'generation_process.png')
        generate_process_visualization(ddpm, save_path)
    
    if args.mode == 'interpolate' or args.mode == 'all':
        # 插值生成
        save_path = os.path.join(args.output_dir, 'interpolation.png')
        interpolate_samples(ddpm, save_path)
    
    print("推理完成！")


if __name__ == '__main__':
    main() 