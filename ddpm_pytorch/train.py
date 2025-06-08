"""
DDPM训练脚本
"""
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.utils import save_image
from tqdm import tqdm

from model import UNet
from ddpm import DDPM
from config import Config


def get_dataloader():
    """获取数据加载器"""
    transform = transforms.Compose([
        transforms.Resize(Config.image_size),
        transforms.CenterCrop(Config.image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # 标准化到[-1, 1]
    ])
    
    # 使用CIFAR-10数据集
    dataset = datasets.CIFAR10(
        root=Config.data_path,
        train=True,
        download=True,
        transform=transform
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    return dataloader


def train():
    """训练函数"""
    # 创建保存目录
    os.makedirs(Config.save_path, exist_ok=True)
    os.makedirs(Config.samples_path, exist_ok=True)
    
    # 获取数据
    dataloader = get_dataloader()
    print(f"数据集大小: {len(dataloader.dataset)}")
    print(f"批次数量: {len(dataloader)}")
    
    # 创建模型
    model = UNet(
        in_channels=Config.channels,
        dim=Config.dim,
        dim_mults=Config.dim_mults
    ).to(Config.device)
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")
    
    # 创建DDPM
    ddpm = DDPM(
        model=model,
        timesteps=Config.timesteps,
        beta_start=Config.beta_start,
        beta_end=Config.beta_end,
        beta_schedule=Config.beta_schedule,
        device=Config.device
    )
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate)
    
    # 训练循环
    global_step = 0
    
    for epoch in range(Config.epochs):
        model.train()
        epoch_loss = 0
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{Config.epochs}')
        
        for batch_idx, (data, _) in enumerate(pbar):
            data = data.to(Config.device)
            
            # 计算损失
            loss = ddpm.training_loss(data)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            global_step += 1
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{epoch_loss/(batch_idx+1):.4f}'
            })
            
            # 生成样本
            if global_step % 1000 == 0:
                model.eval()
                with torch.no_grad():
                    # 生成16个样本
                    samples = ddpm.sample((16, Config.channels, Config.image_size, Config.image_size))
                    # 反标准化
                    samples = (samples + 1) / 2
                    samples = torch.clamp(samples, 0, 1)
                    
                    # 保存样本
                    save_image(
                        samples,
                        os.path.join(Config.samples_path, f'sample_step_{global_step}.png'),
                        nrow=4,
                        normalize=False
                    )
                model.train()
        
        avg_loss = epoch_loss / len(dataloader)
        print(f'Epoch {epoch+1}, Average Loss: {avg_loss:.4f}')
        
        # 保存检查点
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'config': Config
            }
            
            torch.save(
                checkpoint,
                os.path.join(Config.save_path, f'checkpoint_epoch_{epoch+1}.pth')
            )
            print(f'检查点已保存: checkpoint_epoch_{epoch+1}.pth')
    
    # 保存最终模型
    final_checkpoint = {
        'epoch': Config.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': Config
    }
    
    torch.save(final_checkpoint, os.path.join(Config.save_path, 'final_model.pth'))
    print('训练完成！最终模型已保存。')


if __name__ == '__main__':
    print("开始DDPM训练...")
    print(f"使用设备: {Config.device}")
    print(f"图像尺寸: {Config.image_size}x{Config.image_size}")
    print(f"批次大小: {Config.batch_size}")
    print(f"时间步数: {Config.timesteps}")
    print(f"学习率: {Config.learning_rate}")
    print(f"训练轮数: {Config.epochs}")
    
    train() 