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

def get_high_res_dataloader(
    dataset_name='cifar10', 
    image_size=64, 
    batch_size=16, 
    num_workers=None
):
    """获取高分辨率数据加载器"""
    
    if num_workers is None:
        num_workers = min(8, os.cpu_count() or 4)
    
    # 基础变换
    base_transforms = [
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
    ]
    
    # 数据增强（根据分辨率调整）
    augment_transforms = []
    if image_size >= 64:
        augment_transforms.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),  # 高分辨率可以加更多增强
        ])
    else:
        augment_transforms.append(transforms.RandomHorizontalFlip(p=0.5))
    
    # 最终变换
    final_transforms = [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
    
    transform = transforms.Compose(base_transforms + augment_transforms + final_transforms)
    
    # 选择数据集
    if dataset_name.lower() == 'cifar10':
        dataset = torchvision.datasets.CIFAR10(
            root='./data',
            train=True,
            download=True,
            transform=transform
        )
    elif dataset_name.lower() == 'cifar100':
        dataset = torchvision.datasets.CIFAR100(
            root='./data',
            train=True,
            download=True,
            transform=transform
        )
    elif dataset_name.lower() == 'celeba':
        # CelebA数据集（需要手动下载）
        try:
            dataset = torchvision.datasets.CelebA(
                root='./data',
                split='train',
                download=False,  # 通常需要手动下载
                transform=transform
            )
        except Exception as e:
            print(f"❌ CelebA数据集加载失败: {e}")
            print("请手动下载CelebA数据集或使用其他数据集")
            raise
    elif dataset_name.lower() == 'imagenet':
        # ImageNet的小版本或使用STL10作为替代
        try:
            dataset = torchvision.datasets.STL10(
                root='./data',
                split='train',
                download=True,
                transform=transform
            )
            print("使用STL10数据集作为高分辨率替代")
        except Exception as e:
            print(f"❌ 高分辨率数据集加载失败: {e}")
            raise
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2,
        drop_last=True
    )
    
    return dataloader, dataset

def get_model_config_for_resolution(image_size):
    """根据图像分辨率获取推荐的模型配置"""
    if image_size <= 32:
        return {
            'dim': 64,
            'dim_mults': (1, 2, 4, 8),
            'timesteps': 1000
        }
    elif image_size <= 64:
        return {
            'dim': 128,
            'dim_mults': (1, 2, 4, 8),
            'timesteps': 1000
        }
    elif image_size <= 128:
        return {
            'dim': 192,
            'dim_mults': (1, 1, 2, 2, 4, 4),
            'timesteps': 1000
        }
    elif image_size <= 256:
        return {
            'dim': 256,
            'dim_mults': (1, 1, 2, 2, 4, 4, 8),
            'timesteps': 1000
        }
    else:
        return {
            'dim': 320,
            'dim_mults': (1, 1, 2, 2, 4, 4, 8, 8),
            'timesteps': 1000
        }

def get_batch_size_for_resolution(image_size, gpu_memory_gb=12):
    """根据分辨率和GPU内存推荐批次大小"""
    if image_size <= 32:
        if gpu_memory_gb >= 24:
            return 64
        elif gpu_memory_gb >= 12:
            return 32
        else:
            return 16
    elif image_size <= 64:
        if gpu_memory_gb >= 24:
            return 32
        elif gpu_memory_gb >= 12:
            return 16
        else:
            return 8
    elif image_size <= 128:
        if gpu_memory_gb >= 24:
            return 16
        elif gpu_memory_gb >= 12:
            return 8
        else:
            return 4
    elif image_size <= 256:
        if gpu_memory_gb >= 24:
            return 8
        elif gpu_memory_gb >= 12:
            return 4
        else:
            return 2
    else:  # 512+
        if gpu_memory_gb >= 24:
            return 4
        elif gpu_memory_gb >= 12:
            return 2
        else:
            return 1

def train_ddpm_high_res(
    dataset_name='cifar10',
    image_size=64,
    epochs=50,
    batch_size=None,  # 自动根据分辨率计算
    learning_rate=1e-4,
    save_interval=10,
    use_amp=True,
    gradient_accumulation_steps=None,  # 自动计算
    compile_model=True,
    fast_sampling_interval=5
):
    """高分辨率DDPM训练"""
    
    print(f"🚀 启动高分辨率DDPM训练")
    print(f"📊 配置信息:")
    print(f"   - 数据集: {dataset_name.upper()}")
    print(f"   - 图像分辨率: {image_size}x{image_size}")
    
    # 获取GPU信息
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        # 估算GPU内存
        if '3060' in gpu_name:
            gpu_memory = 12
        elif '3090' in gpu_name or '4090' in gpu_name:
            gpu_memory = 24
        elif '3080' in gpu_name:
            gpu_memory = 10
        else:
            gpu_memory = 8  # 保守估计
        print(f"   - GPU: {gpu_name} ({gpu_memory}GB)")
    else:
        gpu_memory = 0
        print("   - 设备: CPU")
    
    # 自动配置批次大小
    if batch_size is None:
        batch_size = get_batch_size_for_resolution(image_size, gpu_memory)
    
    # 自动配置梯度累积
    if gradient_accumulation_steps is None:
        # 尝试保持有效批次大小在16-32之间
        target_effective_batch = 24
        gradient_accumulation_steps = max(1, target_effective_batch // batch_size)
    
    effective_batch_size = batch_size * gradient_accumulation_steps
    
    print(f"   - 实际批次大小: {batch_size}")
    print(f"   - 梯度累积步数: {gradient_accumulation_steps}")
    print(f"   - 有效批次大小: {effective_batch_size}")
    
    # 获取模型配置
    model_config = get_model_config_for_resolution(image_size)
    print(f"   - 模型维度: {model_config['dim']}")
    print(f"   - 层级倍数: {model_config['dim_mults']}")
    
    # 创建模型
    model = DDPMModel(
        image_size=image_size,
        channels=3,
        **model_config
    )
    
    # 显示模型参数数量
    total_params = sum(p.numel() for p in model.unet.parameters())
    print(f"   - 模型参数: {total_params:,}")
    
    # 模型编译加速
    if compile_model and hasattr(torch, 'compile'):
        try:
            compiled_unet = torch.compile(model.unet)
            model.unet = compiled_unet  # type: ignore
            print("✅ 模型编译成功")
        except Exception as e:
            print(f"⚠️ 模型编译失败: {e}")
    
    # 优化器设置（对高分辨率调整学习率）
    if image_size > 64:
        learning_rate = learning_rate * 0.8  # 高分辨率使用稍低的学习率
    
    optimizer = optim.AdamW(
        model.unet.parameters(),
        lr=learning_rate,
        weight_decay=0.01,
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
    
    # 获取数据加载器
    dataloader, dataset = get_high_res_dataloader(
        dataset_name=dataset_name,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=8
    )
    
    # 创建保存目录
    exp_name = f"{dataset_name}_{image_size}x{image_size}"
    os.makedirs(f'checkpoints/{exp_name}', exist_ok=True)
    os.makedirs(f'samples/{exp_name}', exist_ok=True)
    os.makedirs(f'training_progress/{exp_name}', exist_ok=True)
    
    print(f"📁 实验名称: {exp_name}")
    print("=" * 60)
    
    # 训练历史记录
    losses = []
    best_loss = float('inf')
    
    # 开始训练
    training_start_time = time.time()
    
    for epoch in range(epochs):
        model.unet.train()
        epoch_losses = []
        epoch_start_time = time.time()
        
        optimizer.zero_grad()
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')
        for batch_idx, (data, _) in enumerate(pbar):
            
            # 混合精度前向传播
            if use_amp and scaler is not None:
                with autocast():
                    loss = model.train_step(data) / gradient_accumulation_steps
                
                scaler.scale(loss).backward()
                
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
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
            best_path = f'checkpoints/{exp_name}/ddpm_best.pt'
            model.save_model(best_path)
            print(f'   🏆 新的最佳模型已保存')
        
        # 快速采样监控（减少样本数量以节省时间）
        if (epoch + 1) % fast_sampling_interval == 0:
            model.unet.eval()
            with torch.no_grad():
                # 高分辨率用更少的样本
                sample_count = max(4, 16 // (image_size // 32))
                quick_samples = model.sample(batch_size=sample_count)
                save_high_res_samples(
                    quick_samples,
                    f'training_progress/{exp_name}/quick_epoch_{epoch+1}.png',
                    epoch=epoch+1,
                    loss=avg_loss,
                    image_size=image_size
                )
            print(f'   📸 快速采样已保存')
        
        # 完整检查点保存
        if (epoch + 1) % save_interval == 0:
            checkpoint_path = f'checkpoints/{exp_name}/ddpm_epoch_{epoch+1}.pt'
            model.save_model(checkpoint_path)
            
            # 生成完整样本
            model.unet.eval()
            with torch.no_grad():
                sample_count = max(8, 16 // (image_size // 32))
                samples = model.sample(batch_size=sample_count)
                save_high_res_samples(
                    samples,
                    f'samples/{exp_name}/epoch_{epoch+1}_samples.png',
                    epoch=epoch+1,
                    loss=avg_loss,
                    image_size=image_size
                )
            
            # 保存训练曲线
            plot_high_res_losses(losses, f'samples/{exp_name}/loss_curve_epoch_{epoch+1}.png', exp_name)
            print(f'   💾 完整检查点已保存')
        
        print("-" * 50)
    
    # 训练完成
    total_time = time.time() - training_start_time
    
    # 保存最终模型
    final_path = f'checkpoints/{exp_name}/ddpm_final.pt'
    model.save_model(final_path)
    
    print("🎉 高分辨率训练完成！")
    print(f"⏱️  总训练时间: {total_time/3600:.2f}小时")
    print(f"📉 最佳损失: {best_loss:.4f}")
    print(f"💾 模型已保存到: {final_path}")
    
    return model, losses

def save_high_res_samples(samples, path, epoch=None, loss=None, image_size=64):
    """保存高分辨率样本"""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        samples = (samples + 1) / 2
        samples = torch.clamp(samples, 0, 1)
        
        # 根据图像大小调整网格
        if len(samples) <= 4:
            nrow = 2
        elif len(samples) <= 9:
            nrow = 3
        else:
            nrow = 4
        
        grid = torchvision.utils.make_grid(samples, nrow=nrow, padding=2)
        grid_np = grid.permute(1, 2, 0).cpu().numpy()
        
        # 根据分辨率调整图像大小
        fig_size = min(12, max(8, image_size // 16))
        plt.figure(figsize=(fig_size, fig_size))
        plt.imshow(grid_np)
        plt.axis('off')
        
        if epoch is not None and loss is not None:
            plt.title(f'Epoch {epoch} - Loss: {loss:.4f} ({image_size}x{image_size})', 
                     fontsize=14, fontweight='bold')
        
        # 高分辨率图像保存更高质量
        dpi = 150 if image_size >= 128 else 100
        plt.savefig(path, bbox_inches='tight', dpi=dpi)
        plt.close()
        
    except Exception as e:
        print(f"保存样本时出错: {e}")

def plot_high_res_losses(losses, path, exp_name):
    """绘制高分辨率训练损失"""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        plt.figure(figsize=(12, 6))
        plt.plot(losses, linewidth=2, color='blue')
        plt.title(f'Training Loss - {exp_name}', fontsize=16, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # 添加最小值标记
        if losses:
            min_loss_idx = np.argmin(losses)
            min_loss = losses[min_loss_idx]
            plt.scatter([min_loss_idx], [min_loss], color='red', s=100, zorder=5)
            plt.annotate(f'Best: {min_loss:.4f}', 
                        xy=(float(min_loss_idx), float(min_loss)),
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                        fontsize=10)
        
        plt.tight_layout()
        plt.savefig(path, bbox_inches='tight', dpi=150)
        plt.close()
        
    except Exception as e:
        print(f"保存损失曲线时出错: {e}")

if __name__ == "__main__":
    import sys
    
    # 解析命令行参数
    use_amp = '--no-amp' not in sys.argv
    compile_model = '--no-compile' not in sys.argv
    
    # 解析数据集和分辨率
    dataset_name = 'cifar10'
    image_size = 64
    
    for arg in sys.argv[1:]:
        if arg.startswith('--dataset='):
            dataset_name = arg.split('=')[1]
        elif arg.startswith('--size='):
            image_size = int(arg.split('=')[1])
    
    print("🚀 启动高分辨率训练脚本")
    print(f"数据集: {dataset_name}")
    print(f"分辨率: {image_size}x{image_size}")
    print(f"混合精度: {'开启' if use_amp else '关闭'}")
    print(f"模型编译: {'开启' if compile_model else '关闭'}")
    print("\n可用选项:")
    print("  --dataset=cifar10/cifar100/celeba/imagenet")
    print("  --size=32/64/128/256")
    print("  --no-amp (关闭混合精度)")
    print("  --no-compile (关闭模型编译)")
    
    try:
        model, losses = train_ddpm_high_res(
            dataset_name=dataset_name,
            image_size=image_size,
            epochs=50,
            use_amp=use_amp,
            compile_model=compile_model
        )
        
        print("🎉 高分辨率训练成功完成！")
        
    except Exception as e:
        print(f"❌ 训练过程中出错: {e}")
        import traceback
        traceback.print_exc() 