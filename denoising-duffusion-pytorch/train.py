import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
from ddpm_model import DDPMModel
from config import get_auto_config, print_config # 导入配置

def test_train_setup():
    """测试训练设置是否正常"""
    print("🔍 验证训练脚本设置...")
    
    try:
        # 测试数据加载器
        dataloader, dataset = get_cifar10_dataloader(batch_size=4, image_size=32)
        print(f"✅ 数据加载正常 - 数据集大小: {len(dataset)}")
        
        # 测试模型创建
        model = DDPMModel(
            image_size=32,
            channels=3,
            dim=32,  # 使用较小的模型进行测试
            dim_mults=(1, 2),
            timesteps=100
        )
        print("✅ 模型创建正常")
        
        # 测试一个批次的训练
        data_iter = iter(dataloader)
        test_batch, _ = next(data_iter)
        
        optimizer = torch.optim.Adam(model.unet.parameters(), lr=1e-4)
        
        model.unet.train()
        optimizer.zero_grad()
        loss = model.train_step(test_batch)
        loss.backward()
        optimizer.step()
        
        print(f"✅ 训练步骤正常 - 测试损失: {loss.item():.4f}")
        print("🎉 训练脚本验证通过！\n")
        
        return True
        
    except Exception as e:
        print(f"❌ 训练脚本验证失败: {e}")
        return False

def get_cifar10_dataloader(batch_size=32, image_size=32, num_workers=2, pin_memory=True): # 从config获取默认值
    """获取CIFAR-10数据加载器"""
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化到[-1, 1]
    ])
    
    dataset = torchvision.datasets.CIFAR10(
        root='./data', # 可以从config获取
        train=True,
        download=True,
        transform=transform
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return dataloader, dataset

def train_ddpm(config): # 接收配置对象作为参数
    """训练DDPM模型"""
    
    # 创建模型
    model = DDPMModel(
        image_size=config.IMAGE_SIZE,
        channels=config.CHANNELS,
        dim=config.DIM,
        dim_mults=config.DIM_MULTS,
        timesteps=config.TIMESTEPS
    )
    model.unet.to(config.DEVICE) # 将模型的unet部分移动到配置的设备
    
    # 设置优化器
    optimizer = optim.Adam(model.unet.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    
    # 获取数据加载器
    dataloader, dataset = get_cifar10_dataloader(
        batch_size=config.BATCH_SIZE,
        image_size=config.IMAGE_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    # 创建保存目录
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.SAMPLE_DIR, exist_ok=True)
    
    # 训练历史记录
    losses = []
    
    print(f"开始训练DDPM模型...")
    print_config(config) # 打印配置信息
        
    for epoch in range(config.EPOCHS):
        model.unet.train()
        epoch_losses = []
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{config.EPOCHS}')
        for batch_idx, (data, _) in enumerate(pbar):
            data = data.to(config.DEVICE) # 将数据移动到配置的设备
            optimizer.zero_grad()
            
            # 前向传播
            loss = model.train_step(data)
            
            # 反向传播
            loss.backward()
            if config.GRADIENT_CLIP > 0:
                 torch.nn.utils.clip_grad_norm_(model.unet.parameters(), config.GRADIENT_CLIP)
            optimizer.step()
            
            epoch_losses.append(loss.item())
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # 记录平均损失
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        
        print(f'Epoch {epoch+1}, 平均损失: {avg_loss:.4f}')
        
        # 保存检查点和生成样本
        if (epoch + 1) % config.SAVE_INTERVAL == 0:
            # 保存模型
            checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f'ddpm_epoch_{epoch+1}.pt')
            model.save_model(checkpoint_path)
            
            # 生成样本并保存
            model.unet.eval()
            with torch.no_grad():
                samples = model.sample(batch_size=config.INFERENCE_BATCH_SIZE) # 使用配置的推理批次大小和设备
                save_samples(samples, os.path.join(config.SAMPLE_DIR, f'epoch_{epoch+1}_samples.png'))
            
            # 保存损失曲线
            plot_losses(losses, os.path.join(config.SAMPLE_DIR, f'loss_curve_epoch_{epoch+1}.png'))
    
    # 保存最终模型
    final_path = os.path.join(config.CHECKPOINT_DIR, 'ddpm_final.pt')
    model.save_model(final_path)
    
    print("训练完成！")
    return model, losses

def save_samples(samples, path):
    """保存生成的样本图像"""
    try:
        # 确保保存目录存在
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # 将样本从[-1, 1]转换到[0, 1]
        samples = (samples + 1) / 2
        samples = torch.clamp(samples, 0, 1)
        
        # 创建网格图像
        grid = torchvision.utils.make_grid(samples, nrow=4, padding=2)
        
        # 转换为numpy数组并保存
        grid_np = grid.permute(1, 2, 0).cpu().numpy()
        
        plt.figure(figsize=(10, 10))
        plt.imshow(grid_np)
        plt.axis('off')
        plt.title('Generated Samples')
        plt.savefig(path, bbox_inches='tight', dpi=150)
        plt.close()
        
        print(f"样本已保存到: {path}")
    except Exception as e:
        print(f"保存样本时出错: {e}")

def plot_losses(losses, path):
    """绘制并保存损失曲线"""
    try:
        # 确保保存目录存在
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        plt.figure(figsize=(10, 6))
        plt.plot(losses)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig(path, bbox_inches='tight', dpi=150)
        plt.close()
        
        print(f"损失曲线已保存到: {path}")
    except Exception as e:
        print(f"保存损失曲线时出错: {e}")

if __name__ == "__main__":
    import sys
    
    # 获取配置
    config = get_auto_config()

    # 检查命令行参数
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        # 只运行验证测试
        # 注意：test_train_setup 也需要调整以接受或使用config
        if test_train_setup(): # 你可能需要修改 test_train_setup 以使用 config
            print("✅ 训练脚本准备就绪，可以开始完整训练")
        else:
            print("❌ 请修复问题后再开始训练")
        sys.exit()
    
    # 运行验证测试
    print("开始训练前验证...")
    # 注意：test_train_setup 也需要调整以接受或使用config
    if not test_train_setup(): # 你可能需要修改 test_train_setup 以使用 config
        print("❌ 验证失败，请检查环境设置")
        sys.exit(1)
    
    # 开始完整训练
    try:
        model, losses = train_ddpm(config) # 传递配置对象
        print("🎉 训练成功完成！")
    except Exception as e:
        print(f"❌ 训练过程中出错: {e}")
        import traceback
        traceback.print_exc()