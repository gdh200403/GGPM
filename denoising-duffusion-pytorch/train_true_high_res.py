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

def get_native_high_res_dataloader(
    dataset_name='stl10', 
    batch_size=16, 
    num_workers=None,
    use_upsampling=False,
    target_size=None
):
    """获取原生高分辨率数据加载器"""
    
    if num_workers is None:
        num_workers = min(8, os.cpu_count() or 4)
    
    # 根据数据集确定原生分辨率
    if dataset_name.lower() == 'cifar10':
        native_size = 32
        if not use_upsampling:
            print("⚠️ CIFAR-10原生分辨率为32x32")
            print("   建议使用STL-10(96x96)或CelebA(64x64)获得真正的高分辨率")
            print("   或者设置use_upsampling=True进行上采样")
    elif dataset_name.lower() == 'stl10':
        native_size = 96
    elif dataset_name.lower() == 'celeba':
        native_size = 64  # CelebA原始是更高的，但通常crop到64x64
    else:
        native_size = 64  # 默认
    
    # 决定最终的图像尺寸
    if target_size is None:
        image_size = native_size
    else:
        image_size = target_size
        if target_size > native_size and not use_upsampling:
            print(f"⚠️ 警告：目标尺寸{target_size}大于原生尺寸{native_size}")
            print("   这会导致上采样，建议使用原生尺寸或更高分辨率数据集")
    
    print(f"📊 数据集信息:")
    print(f"   - 数据集: {dataset_name.upper()}")
    print(f"   - 原生分辨率: {native_size}x{native_size}")
    print(f"   - 训练分辨率: {image_size}x{image_size}")
    print(f"   - 是否上采样: {'是' if image_size > native_size else '否'}")
    
    # 构建变换管道
    transforms_list = []
    
    # 基础尺寸调整
    if image_size != native_size:
        if image_size < native_size:
            # 下采样：先resize再crop
            transforms_list.extend([
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size)
            ])
        else:
            # 上采样：发出警告
            if use_upsampling:
                transforms_list.extend([
                    transforms.Resize(image_size),
                    transforms.CenterCrop(image_size)
                ])
                print("🔄 正在进行上采样，图像质量可能受影响")
            else:
                raise ValueError(f"目标尺寸{image_size}大于原生尺寸{native_size}，请设置use_upsampling=True")
    
    # 数据增强
    augment_transforms = []  # type: ignore
    augment_transforms.append(transforms.RandomHorizontalFlip(p=0.5))
    
    # 对于高分辨率图像，可以添加更多增强
    if image_size >= 64:
        augment_transforms.append(transforms.RandomRotation(5))  # 轻微旋转
        # augment_transforms.append(transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05))
    
    # 最终变换
    final_transforms = [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
    
    # 组合所有变换
    transform = transforms.Compose(transforms_list + augment_transforms + final_transforms)
    
    # 创建数据集
    if dataset_name.lower() == 'cifar10':
        dataset = torchvision.datasets.CIFAR10(
            root='./data',
            train=True,
            download=True,
            transform=transform
        )
    elif dataset_name.lower() == 'stl10':
        dataset = torchvision.datasets.STL10(
            root='./data',
            split='train',
            download=True,
            transform=transform
        )
        print("✅ 使用STL-10数据集，原生96x96分辨率")
    elif dataset_name.lower() == 'celeba':
        try:
            # CelebA的特殊处理
            celeba_transform = transforms.Compose([
                transforms.CenterCrop(178),  # CelebA推荐的crop
                transforms.Resize(image_size),
                *augment_transforms,
                *final_transforms
            ])
            
            dataset = torchvision.datasets.CelebA(
                root='./data',
                split='train',
                download=False,  # 需要手动下载
                transform=celeba_transform
            )
            print("✅ 使用CelebA数据集，高质量人脸图像")
        except Exception as e:
            print(f"❌ CelebA加载失败: {e}")
            print("💡 CelebA需要手动下载，请参考：")
            print("   https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html")
            raise
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")
    
    # 创建数据加载器
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
    
    return dataloader, dataset, image_size

def train_true_high_res_ddpm(
    dataset_name='stl10',
    target_size=None,  # None表示使用原生分辨率
    use_upsampling=False,
    epochs=50,
    batch_size=None,
    learning_rate=1e-4,
    save_interval=10,
    use_amp=True,
    compile_model=True
):
    """训练真正的高分辨率DDPM模型"""
    
    print("🎯 真正的高分辨率DDPM训练")
    print("=" * 50)
    
    # 获取数据加载器
    dataloader, dataset, image_size = get_native_high_res_dataloader(
        dataset_name=dataset_name,
        target_size=target_size,
        use_upsampling=use_upsampling,
        batch_size=batch_size or 16
    )
    
    # 获取GPU信息并调整批次大小
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        if '3060' in gpu_name:
            gpu_memory = 12
        elif '3090' in gpu_name or '4090' in gpu_name:
            gpu_memory = 24
        elif '3080' in gpu_name:
            gpu_memory = 10
        else:
            gpu_memory = 8
        print(f"🖥️  GPU: {gpu_name} ({gpu_memory}GB)")
    else:
        gpu_memory = 0
        print("🖥️  设备: CPU")
    
    # 智能批次大小调整
    if batch_size is None:
        if image_size <= 32:
            batch_size = 32 if gpu_memory >= 12 else 16
        elif image_size <= 64:
            batch_size = 16 if gpu_memory >= 12 else 8
        elif image_size <= 96:
            batch_size = 12 if gpu_memory >= 12 else 6
        else:
            batch_size = 8 if gpu_memory >= 12 else 4
    
    # 重新创建dataloader with正确的batch_size
    dataloader, dataset, image_size = get_native_high_res_dataloader(
        dataset_name=dataset_name,
        target_size=target_size,
        use_upsampling=use_upsampling,
        batch_size=batch_size
    )
    
    # 根据分辨率配置模型
    if image_size <= 32:
        model_config = {'dim': 64, 'dim_mults': (1, 2, 4, 8)}
    elif image_size <= 64:
        model_config = {'dim': 128, 'dim_mults': (1, 2, 4, 8)}
    elif image_size <= 96:
        model_config = {'dim': 160, 'dim_mults': (1, 2, 4, 8)}
    else:
        model_config = {'dim': 192, 'dim_mults': (1, 1, 2, 2, 4, 4)}
    
    print(f"🏗️  模型配置:")
    print(f"   - 批次大小: {batch_size}")
    print(f"   - 模型维度: {model_config['dim']}")
    print(f"   - 层级倍数: {model_config['dim_mults']}")
    
    # 创建模型
    model = DDPMModel(
        image_size=image_size,
        channels=3,
        timesteps=1000,
        **model_config
    )
    
    total_params = sum(p.numel() for p in model.unet.parameters())
    print(f"   - 模型参数: {total_params:,}")
    
    # 模型编译
    if compile_model and hasattr(torch, 'compile'):
        try:
            compiled_unet = torch.compile(model.unet)
            model.unet = compiled_unet  # type: ignore
            print("✅ 模型编译成功")
        except Exception as e:
            print(f"⚠️ 模型编译失败: {e}")
    
    # 优化器
    optimizer = optim.AdamW(
        model.unet.parameters(),
        lr=learning_rate,
        weight_decay=0.01,
        betas=(0.9, 0.999)
    )
    
    # 学习率调度
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=learning_rate * 0.1
    )
    
    # 混合精度
    scaler = GradScaler() if use_amp and AMP_AVAILABLE else None
    
    # 创建保存目录
    exp_name = f"{dataset_name}_native_{image_size}x{image_size}"
    if use_upsampling and target_size:
        exp_name = f"{dataset_name}_upsampled_{image_size}x{image_size}"
    
    os.makedirs(f'checkpoints/{exp_name}', exist_ok=True)
    os.makedirs(f'samples/{exp_name}', exist_ok=True)
    os.makedirs(f'training_progress/{exp_name}', exist_ok=True)
    
    print(f"📁 实验名称: {exp_name}")
    print("=" * 60)
    
    # 训练循环
    losses = []
    best_loss = float('inf')
    training_start_time = time.time()
    
    for epoch in range(epochs):
        model.unet.train()
        epoch_losses = []
        epoch_start_time = time.time()
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')
        for batch_idx, (data, _) in enumerate(pbar):
            optimizer.zero_grad()
            
            if use_amp and scaler is not None:
                with autocast():
                    loss = model.train_step(data)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = model.train_step(data)
                loss.backward()
                optimizer.step()
            
            epoch_losses.append(loss.item())
            
            current_lr = scheduler.get_last_lr()[0]
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
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
        
        # 快速采样
        if (epoch + 1) % 5 == 0:
            model.unet.eval()
            with torch.no_grad():
                sample_count = min(9, batch_size)
                quick_samples = model.sample(batch_size=sample_count)
                save_native_samples(
                    quick_samples,
                    f'training_progress/{exp_name}/quick_epoch_{epoch+1}.png',
                    epoch=epoch+1,
                    loss=avg_loss,
                    image_size=image_size,
                    is_native=not use_upsampling
                )
            print(f'   📸 快速采样已保存')
        
        # 检查点保存
        if (epoch + 1) % save_interval == 0:
            checkpoint_path = f'checkpoints/{exp_name}/ddpm_epoch_{epoch+1}.pt'
            model.save_model(checkpoint_path)
            
            model.unet.eval()
            with torch.no_grad():
                samples = model.sample(batch_size=16)
                save_native_samples(
                    samples,
                    f'samples/{exp_name}/epoch_{epoch+1}_samples.png',
                    epoch=epoch+1,
                    loss=avg_loss,
                    image_size=image_size,
                    is_native=not use_upsampling
                )
            
            plot_native_losses(losses, f'samples/{exp_name}/loss_curve_epoch_{epoch+1}.png', exp_name)
            print(f'   💾 完整检查点已保存')
        
        print("-" * 50)
    
    # 训练完成
    total_time = time.time() - training_start_time
    final_path = f'checkpoints/{exp_name}/ddpm_final.pt'
    model.save_model(final_path)
    
    print("🎉 真正的高分辨率训练完成！")
    print(f"⏱️  总训练时间: {total_time/3600:.2f}小时")
    print(f"📉 最佳损失: {best_loss:.4f}")
    print(f"💾 模型已保存到: {final_path}")
    
    return model, losses

def save_native_samples(samples, path, epoch=None, loss=None, image_size=64, is_native=True):
    """保存原生高分辨率样本"""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        samples = (samples + 1) / 2
        samples = torch.clamp(samples, 0, 1)
        
        nrow = 3 if len(samples) <= 9 else 4
        grid = torchvision.utils.make_grid(samples, nrow=nrow, padding=2)
        grid_np = grid.permute(1, 2, 0).cpu().numpy()
        
        fig_size = max(8, image_size // 12)
        plt.figure(figsize=(fig_size, fig_size))
        plt.imshow(grid_np)
        plt.axis('off')
        
        title = f'Epoch {epoch} - Loss: {loss:.4f} ({image_size}x{image_size})'
        if is_native:
            title += ' [原生分辨率]'
        else:
            title += ' [上采样]'
        
        if epoch is not None and loss is not None:
            plt.title(title, fontsize=14, fontweight='bold')
        
        dpi = 200 if image_size >= 64 else 150
        plt.savefig(path, bbox_inches='tight', dpi=dpi)
        plt.close()
        
    except Exception as e:
        print(f"保存样本时出错: {e}")

def plot_native_losses(losses, path, exp_name):
    """绘制原生高分辨率训练损失"""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        plt.figure(figsize=(12, 6))
        plt.plot(losses, linewidth=2, color='blue')
        plt.title(f'Training Loss - {exp_name}', fontsize=16, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        if losses:
            min_loss_idx = np.argmin(losses)
            min_loss = losses[min_loss_idx]
            plt.scatter([min_loss_idx], [min_loss], color='red', s=100, zorder=5)
            plt.annotate(f'Best: {min_loss:.4f}', 
                        xy=(float(min_loss_idx), float(min_loss)),
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7))
        
        plt.tight_layout()
        plt.savefig(path, bbox_inches='tight', dpi=150)
        plt.close()
        
    except Exception as e:
        print(f"保存损失曲线时出错: {e}")

if __name__ == "__main__":
    import sys
    
    print("🎯 真正的高分辨率DDPM训练")
    print("推荐的数据集组合:")
    print("  1. STL-10 (原生96x96) - python train_true_high_res.py stl10")
    print("  2. CelebA (原生64x64) - python train_true_high_res.py celeba")
    print("  3. CIFAR-10 (原生32x32) - python train_true_high_res.py cifar10")
    print()
    
    # 解析参数
    dataset_name = sys.argv[1] if len(sys.argv) > 1 else 'stl10'
    target_size = int(sys.argv[2]) if len(sys.argv) > 2 else None
    use_upsampling = '--allow-upsampling' in sys.argv
    
    print(f"选择的数据集: {dataset_name}")
    if target_size:
        print(f"目标分辨率: {target_size}x{target_size}")
    print(f"允许上采样: {'是' if use_upsampling else '否'}")
    print("=" * 60)
    
    try:
        model, losses = train_true_high_res_ddpm(
            dataset_name=dataset_name,
            target_size=target_size,
            use_upsampling=use_upsampling,
            epochs=50,
            use_amp='--no-amp' not in sys.argv,
            compile_model='--no-compile' not in sys.argv
        )
        
        print("🎉 真正的高分辨率训练成功完成！")
        
    except Exception as e:
        print(f"❌ 训练过程中出错: {e}")
        import traceback
        traceback.print_exc() 