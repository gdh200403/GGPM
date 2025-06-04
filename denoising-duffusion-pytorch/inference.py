import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from ddpm_model import DDPMModel

def load_model(checkpoint_path):
    """加载训练好的模型"""
    model = DDPMModel(
        image_size=32,
        channels=3,
        dim=64,
        dim_mults=(1, 2, 4, 8),
        timesteps=1000
    )
    
    model.load_model(checkpoint_path)
    model.unet.eval()
    
    return model

def generate_samples(model, num_samples=16, save_path='generated_samples.png'):
    """生成新样本"""
    print(f"正在生成 {num_samples} 个样本...")
    
    with torch.no_grad():
        samples = model.sample(batch_size=num_samples)
    
    # 保存样本
    save_image_grid(samples, save_path, title='Generated Samples')
    
    return samples

def generate_interpolation(model, num_steps=10, save_path='interpolation.png'):
    """生成插值序列"""
    print(f"正在生成插值序列 ({num_steps} 步)...")
    
    with torch.no_grad():
        # 生成两个随机样本
        samples = model.sample(batch_size=2)
        x1, x2 = samples[0:1], samples[1:2]
        
        # 执行插值
        interpolated = model.interpolate(x1, x2, num_steps=num_steps)
        
        # 将起始和结束样本添加到插值序列
        full_sequence = torch.cat([x1, interpolated, x2], dim=0)
    
    # 保存插值序列
    save_image_grid(full_sequence, save_path, nrow=num_steps+2, title='Interpolation Sequence')
    
    return full_sequence

def progressive_generation(model, save_path='progressive_generation.png'):
    """展示渐进式生成过程"""
    print("正在展示渐进式生成过程...")
    
    with torch.no_grad():
        # 生成完整的去噪过程
        samples = model.sample(batch_size=1, return_all_timesteps=True)
        
        # 选择几个关键时间步展示
        timesteps_to_show = [0, 50, 100, 200, 400, 600, 800, 999]
        selected_samples = []
        
        for t in timesteps_to_show:
            if t < len(samples):
                selected_samples.append(samples[t])
        
        selected_samples = torch.cat(selected_samples, dim=0)
    
    # 保存渐进式生成过程
    save_image_grid(selected_samples, save_path, nrow=len(timesteps_to_show), 
                   title='Progressive Generation Process')
    
    return selected_samples

def compare_with_real_data(model, num_samples=8):
    """与真实数据对比"""
    print("正在与真实数据进行对比...")
    
    # 加载真实CIFAR-10数据
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    
    # 获取一些真实样本
    real_samples = []
    for i in range(num_samples):
        real_samples.append(dataset[i][0])
    real_samples = torch.stack(real_samples)
    
    # 生成假样本
    with torch.no_grad():
        fake_samples = model.sample(batch_size=num_samples)
    
    # 合并真实和生成的样本
    combined = torch.cat([real_samples, fake_samples], dim=0)
    
    # 保存对比图
    save_image_grid(combined, 'real_vs_generated.png', nrow=num_samples,
                   title='Real (Top) vs Generated (Bottom)')
    
    return real_samples, fake_samples

def save_image_grid(images, path, nrow=4, title='Generated Images'):
    """保存图像网格"""
    # 将图像从[-1, 1]转换到[0, 1]
    images = (images + 1) / 2
    images = torch.clamp(images, 0, 1)
    
    # 创建网格
    grid = torchvision.utils.make_grid(images, nrow=nrow, padding=2)
    grid_np = grid.permute(1, 2, 0).cpu().numpy()
    
    # 保存图像
    plt.figure(figsize=(12, 8))
    plt.imshow(grid_np)
    plt.axis('off')
    plt.title(title)
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()
    
    print(f"图像已保存到: {path}")

def batch_generate(model, total_samples=100, batch_size=16, save_dir='generated_batch'):
    """批量生成大量样本"""
    print(f"正在批量生成 {total_samples} 个样本...")
    
    os.makedirs(save_dir, exist_ok=True)
    
    all_samples = []
    num_batches = (total_samples + batch_size - 1) // batch_size
    
    for i in range(num_batches):
        current_batch_size = min(batch_size, total_samples - i * batch_size)
        
        with torch.no_grad():
            samples = model.sample(batch_size=current_batch_size)
        
        all_samples.append(samples)
        
        # 保存每个批次
        batch_path = f'{save_dir}/batch_{i+1}.png'
        save_image_grid(samples, batch_path, title=f'Batch {i+1}')
        
        print(f"完成批次 {i+1}/{num_batches}")
    
    # 合并所有样本
    all_samples = torch.cat(all_samples, dim=0)
    
    # 保存总览图
    overview_samples = all_samples[:min(64, len(all_samples))]  # 最多显示64个
    save_image_grid(overview_samples, f'{save_dir}/overview.png', nrow=8,
                   title=f'Generated Samples Overview ({len(all_samples)} total)')
    
    return all_samples

def main():
    """主推理函数"""
    # 检查模型文件是否存在
    checkpoint_paths = [
        'checkpoints/ddpm_final.pt',
        'checkpoints/ddpm_epoch_50.pt',
        'checkpoints/ddpm_epoch_40.pt',
        'checkpoints/ddpm_epoch_30.pt'
    ]
    
    model_path = None
    for path in checkpoint_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if model_path is None:
        print("错误: 未找到训练好的模型文件!")
        print("请先运行 train.py 进行训练")
        return
    
    print(f"加载模型: {model_path}")
    model = load_model(model_path)
    
    # 创建输出目录
    os.makedirs('inference_results', exist_ok=True)
    
    # 执行各种推理任务
    print("\n=== 开始推理 ===")
    
    # 1. 生成基本样本
    generate_samples(model, num_samples=16, 
                    save_path='inference_results/basic_samples.png')
    
    # 2. 生成插值序列
    generate_interpolation(model, num_steps=8, 
                          save_path='inference_results/interpolation.png')
    
    # 3. 展示渐进式生成
    progressive_generation(model, 
                          save_path='inference_results/progressive.png')
    
    # 4. 与真实数据对比
    compare_with_real_data(model, num_samples=8)
    
    # 5. 批量生成
    batch_generate(model, total_samples=50, batch_size=16, 
                  save_dir='inference_results/batch_generation')
    
    print("\n=== 推理完成 ===")
    print("所有结果已保存到 inference_results/ 目录")

if __name__ == "__main__":
    main() 