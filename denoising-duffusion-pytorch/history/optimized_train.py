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
import time

# 检查是否支持混合精度训练
try:
    from torch.cuda.amp import autocast, GradScaler
    AMP_AVAILABLE = True
except ImportError:
    AMP_AVAILABLE = False

def get_optimized_cifar10_dataloader(batch_size=32, image_size=32, num_workers=None):
    """获取优化的CIFAR-10数据加载器"""
    
    # 自动设置最优的worker数量
    if num_workers is None:
        num_workers = min(8, os.cpu_count() or 4)  # 默认4个worker，避免None
    
    # 优化的数据变换（减少不必要的操作）
    transform = transforms.Compose([
        transforms.ToTensor(),  # 直接转换，CIFAR-10已经是32x32
        transforms.RandomHorizontalFlip(p=0.5),  # 明确概率
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    # 优化的DataLoader设置
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,  # 保持worker进程
        prefetch_factor=2,  # 预取因子
        drop_last=True  # 丢弃最后不完整的batch，保持一致性
    )
    
    return dataloader, dataset

def train_ddpm_optimized(
    epochs=50, 
    batch_size=32, 
    learning_rate=1e-4, 
    save_interval=10,
    use_amp=True,  # 混合精度训练
    gradient_accumulation_steps=1,  # 梯度累积
    compile_model=True,  # 模型编译加速
    efficient_checkpointing=True,  # 高效检查点
    fast_sampling_interval=5  # 快速采样间隔
):
    """优化版本的DDPM训练"""
    
    print("🚀 启动优化版本DDPM训练")
    print(f"⚡ 优化特性:")
    print(f"   - 混合精度训练: {'✅' if use_amp and AMP_AVAILABLE else '❌'}")
    print(f"   - 梯度累积: {'✅' if gradient_accumulation_steps > 1 else '❌'}")
    print(f"   - 模型编译: {'✅' if compile_model else '❌'}")
    print(f"   - 高效检查点: {'✅' if efficient_checkpointing else '❌'}")
    
    # 创建模型
    model = DDPMModel(
        image_size=32,
        channels=3,
        dim=64,
        dim_mults=(1, 2, 4, 8),
        timesteps=1000
    )
    
    # 模型编译加速（PyTorch 2.0+）
    if compile_model and hasattr(torch, 'compile'):
        try:
            compiled_unet = torch.compile(model.unet)
            model.unet = compiled_unet  # type: ignore
            print("✅ 模型编译成功")
        except Exception as e:
            print(f"⚠️ 模型编译失败: {e}")
    
    # 设置优化器（使用AdamW和更好的参数）
    optimizer = optim.AdamW(
        model.unet.parameters(), 
        lr=learning_rate,
        weight_decay=0.01,  # 权重衰减
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=epochs,
        eta_min=learning_rate * 0.1
    )
    
    # 混合精度训练
    scaler = GradScaler() if use_amp and AMP_AVAILABLE else None
    
    # 获取优化的数据加载器
    effective_batch_size = batch_size * gradient_accumulation_steps
    dataloader, dataset = get_optimized_cifar10_dataloader(
        batch_size=batch_size, 
        num_workers=8  # 增加worker数量
    )
    
    # 创建保存目录
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('samples', exist_ok=True)
    os.makedirs('optimized_training', exist_ok=True)
    
    # 训练历史记录
    losses = []
    best_loss = float('inf')
    
    print(f"📋 训练配置:")
    print(f"   - 训练轮数: {epochs}")
    print(f"   - 实际批次大小: {batch_size}")
    print(f"   - 有效批次大小: {effective_batch_size}")
    print(f"   - 初始学习率: {learning_rate}")
    print(f"   - 数据集大小: {len(dataset)}")
    print(f"   - 预计每epoch批次数: {len(dataloader)}")
    print("=" * 60)
    
    # 开始训练
    training_start_time = time.time()
    
    for epoch in range(epochs):
        model.unet.train()
        epoch_losses = []
        epoch_start_time = time.time()
        
        # 重置梯度累积
        optimizer.zero_grad()
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')
        for batch_idx, (data, _) in enumerate(pbar):
            
            # 混合精度前向传播
            if use_amp and scaler is not None:
                with autocast():
                    loss = model.train_step(data) / gradient_accumulation_steps
                
                # 混合精度反向传播
                scaler.scale(loss).backward()
                
                # 梯度累积
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                # 标准训练
                loss = model.train_step(data) / gradient_accumulation_steps
                loss.backward()
                
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
            
            epoch_losses.append(loss.item() * gradient_accumulation_steps)
            
            # 更新进度条
            current_lr = scheduler.get_last_lr()[0]
            pbar.set_postfix({
                'loss': f'{loss.item() * gradient_accumulation_steps:.4f}',
                'lr': f'{current_lr:.2e}'
            })
        
        # 学习率调度
        scheduler.step()
        
        # Epoch统计
        epoch_time = time.time() - epoch_start_time
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        
        print(f'📊 Epoch {epoch+1}/{epochs}:')
        print(f'   ⏱️  用时: {epoch_time:.2f}秒')
        print(f'   📉 平均损失: {avg_loss:.4f}')
        print(f'   📈 学习率: {current_lr:.2e}')
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            if efficient_checkpointing:
                best_path = 'checkpoints/ddpm_best.pt'
                model.save_model(best_path)
                print(f'   🏆 新的最佳模型已保存')
        
        # 快速采样监控
        if (epoch + 1) % fast_sampling_interval == 0:
            model.unet.eval()
            with torch.no_grad():
                # 更快的采样（减少样本数量）
                quick_samples = model.sample(batch_size=4)
                save_optimized_samples(
                    quick_samples, 
                    f'optimized_training/quick_epoch_{epoch+1}.png',
                    epoch=epoch+1,
                    loss=avg_loss
                )
            print(f'   📸 快速采样已保存')
        
        # 完整检查点保存
        if (epoch + 1) % save_interval == 0:
            checkpoint_path = f'checkpoints/ddpm_epoch_{epoch+1}.pt'
            model.save_model(checkpoint_path)
            
            # 生成完整样本
            model.unet.eval()
            with torch.no_grad():
                samples = model.sample(batch_size=16)
                save_optimized_samples(
                    samples, 
                    f'samples/epoch_{epoch+1}_samples.png',
                    epoch=epoch+1,
                    loss=avg_loss
                )
            
            # 保存训练曲线
            plot_optimized_losses(losses, f'samples/loss_curve_epoch_{epoch+1}.png')
            print(f'   💾 完整检查点已保存')
        
        print("-" * 50)
    
    # 训练完成
    total_time = time.time() - training_start_time
    
    # 保存最终模型
    final_path = 'checkpoints/ddpm_final_optimized.pt'
    model.save_model(final_path)
    
    print("🎉 优化训练完成！")
    print(f"⏱️  总训练时间: {total_time/3600:.2f}小时")
    print(f"📉 最佳损失: {best_loss:.4f}")
    print(f"💾 模型已保存到: {final_path}")
    
    return model, losses

def save_optimized_samples(samples, path, epoch=None, loss=None):
    """优化的样本保存函数"""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        samples = (samples + 1) / 2
        samples = torch.clamp(samples, 0, 1)
        
        grid = torchvision.utils.make_grid(samples, nrow=4, padding=2)
        grid_np = grid.permute(1, 2, 0).cpu().numpy()
        
        plt.figure(figsize=(10, 8))
        plt.imshow(grid_np)
        plt.axis('off')
        
        if epoch is not None and loss is not None:
            plt.title(f'Epoch {epoch} - Loss: {loss:.4f}', fontsize=14, fontweight='bold')
        
        plt.savefig(path, bbox_inches='tight', dpi=100)  # 降低DPI加速保存
        plt.close()
        
    except Exception as e:
        print(f"保存样本时出错: {e}")

def plot_optimized_losses(losses, path):
    """优化的损失绘制函数"""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        plt.figure(figsize=(10, 6))
        plt.plot(losses, linewidth=2)
        plt.title('Training Loss (Optimized)', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        
        # 添加最小值标记
        min_loss_idx = np.argmin(losses)
        min_loss = losses[min_loss_idx]
        plt.scatter([min_loss_idx], [min_loss], color='red', s=100, zorder=5)
        plt.annotate(f'Best: {min_loss:.4f}', 
                    xy=(float(min_loss_idx), float(min_loss)),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7))
        
        plt.savefig(path, bbox_inches='tight', dpi=100)
        plt.close()
        
    except Exception as e:
        print(f"保存损失曲线时出错: {e}")

if __name__ == "__main__":
    import sys
    
    # 解析命令行参数
    use_amp = '--no-amp' not in sys.argv
    compile_model = '--no-compile' not in sys.argv
    
    print("🚀 启动优化训练脚本")
    print(f"混合精度: {'开启' if use_amp else '关闭'}")
    print(f"模型编译: {'开启' if compile_model else '关闭'}")
    
    try:
        # 针对不同GPU优化批次大小
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            if '3090' in gpu_name or '4090' in gpu_name:
                batch_size = 32
                gradient_accumulation = 1
            elif '3060' in gpu_name:
                batch_size = 16
                gradient_accumulation = 2  # 通过梯度累积模拟更大batch
            else:
                batch_size = 16
                gradient_accumulation = 1
        else:
            batch_size = 8
            gradient_accumulation = 1
        
        model, losses = train_ddpm_optimized(
            epochs=50,
            batch_size=batch_size,
            learning_rate=2e-4,  # 稍微提高学习率
            save_interval=10,
            use_amp=use_amp,
            gradient_accumulation_steps=gradient_accumulation,
            compile_model=compile_model,
            fast_sampling_interval=5
        )
        
        print("🎉 优化训练成功完成！")
        
    except Exception as e:
        print(f"❌ 训练过程中出错: {e}")
        import traceback
        traceback.print_exc() 