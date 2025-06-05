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

def get_improved_high_res_dataloader(
    dataset_name='stl10', 
    batch_size=16, 
    num_workers=None,
    use_upsampling=False,
    target_size=None,
    progressive_training=False,
    current_size=None
):
    """获取改进的高分辨率数据加载器，支持渐进式训练"""
    
    if num_workers is None:
        num_workers = min(6, os.cpu_count() or 4)  # 稍微保守一些
    
    # 根据数据集确定原生分辨率
    if dataset_name.lower() == 'cifar10':
        native_size = 32
    elif dataset_name.lower() == 'stl10':
        native_size = 96
    elif dataset_name.lower() == 'celeba':
        native_size = 64
    else:
        native_size = 64
    
    # 渐进式训练：从小尺寸开始
    if progressive_training and current_size is not None:
        image_size = current_size
        print(f"🔄 渐进式训练: 当前分辨率 {image_size}x{image_size}")
    else:
        image_size = target_size if target_size else native_size
    
    print(f"📊 改进的数据集配置:")
    print(f"   - 数据集: {dataset_name.upper()}")
    print(f"   - 原生分辨率: {native_size}x{native_size}")
    print(f"   - 训练分辨率: {image_size}x{image_size}")
    print(f"   - 渐进式训练: {'是' if progressive_training else '否'}")
    
    # 构建变换管道
    transforms_list = []
    
    # 智能尺寸调整策略
    if image_size != native_size:
        if image_size < native_size:
            # 下采样：保持质量
            transforms_list.extend([
                transforms.Resize(int(image_size * 1.1)),  # 稍微大一点再crop
                transforms.CenterCrop(image_size)
            ])
        else:
            # 上采样：提高质量
            if use_upsampling:
                transforms_list.extend([
                    transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
                    transforms.CenterCrop(image_size)
                ])
                print("🔄 使用双三次插值上采样")
            else:
                raise ValueError(f"目标尺寸{image_size}大于原生尺寸{native_size}，请设置use_upsampling=True")
    
    # 增强的数据增强策略
    augment_transforms = []  # type: ignore
    augment_transforms.append(transforms.RandomHorizontalFlip(p=0.5))
    
    # 根据分辨率调整增强强度
    if image_size >= 64:
        augment_transforms.append(transforms.RandomRotation(degrees=3))  # 轻微旋转
        # 轻微颜色增强，有助于模型泛化
        augment_transforms.append(transforms.ColorJitter(
            brightness=0.05,
            contrast=0.05,
            saturation=0.05,
            hue=0.025
        ))
    
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
    elif dataset_name.lower() == 'celeba':
        try:
            celeba_transform = transforms.Compose([
                transforms.CenterCrop(178),
                transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
                *augment_transforms,
                *final_transforms
            ])
            
            dataset = torchvision.datasets.CelebA(
                root='./data',
                split='train',
                download=False,
                transform=celeba_transform
            )
        except Exception as e:
            print(f"❌ CelebA加载失败: {e}")
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

def get_improved_model_config(image_size, quality_mode='balanced'):
    """获取改进的模型配置"""
    
    if quality_mode == 'fast':
        # 快速训练模式
        if image_size <= 32:
            return {'dim': 64, 'dim_mults': (1, 2, 4, 8)}
        elif image_size <= 64:
            return {'dim': 96, 'dim_mults': (1, 2, 4, 8)}
        elif image_size <= 96:
            return {'dim': 128, 'dim_mults': (1, 2, 4, 8)}
        else:
            return {'dim': 160, 'dim_mults': (1, 2, 4, 8)}
    
    elif quality_mode == 'balanced':
        # 平衡模式（推荐）
        if image_size <= 32:
            return {'dim': 128, 'dim_mults': (1, 2, 4, 8)}
        elif image_size <= 64:
            return {'dim': 160, 'dim_mults': (1, 2, 4, 8)}
        elif image_size <= 96:
            return {'dim': 192, 'dim_mults': (1, 2, 4, 8)}
        else:
            return {'dim': 256, 'dim_mults': (1, 1, 2, 2, 4, 4)}
    
    else:  # quality_mode == 'high'
        # 高质量模式
        if image_size <= 32:
            return {'dim': 192, 'dim_mults': (1, 2, 4, 8)}
        elif image_size <= 64:
            return {'dim': 256, 'dim_mults': (1, 2, 4, 8)}
        elif image_size <= 96:
            return {'dim': 320, 'dim_mults': (1, 2, 4, 8)}
        else:
            return {'dim': 384, 'dim_mults': (1, 1, 2, 2, 4, 4)}

def train_improved_high_res_ddpm(
    dataset_name='stl10',
    target_size=None,
    use_upsampling=False,
    epochs=150,  # 大幅增加训练轮数
    batch_size=None,
    learning_rate=2e-4,  # 稍微提高学习率
    save_interval=20,
    use_amp=True,
    compile_model=True,
    quality_mode='balanced',  # 'fast', 'balanced', 'high'
    progressive_training=False,  # 渐进式训练
    warmup_epochs=20,  # 学习率热身
    early_sample_freq=3  # 早期更频繁采样
):
    """改进的高分辨率DDPM训练，解决效果不理想问题"""
    
    print("🎯 改进的高分辨率DDPM训练")
    print("重点解决生成效果不理想的问题")
    print("=" * 60)
    
    # 获取GPU信息并智能调整配置
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        if '3060' in gpu_name:
            gpu_memory = 12
            if quality_mode == 'high':
                print("⚠️ RTX 3060检测到高质量模式，建议使用balanced模式")
                quality_mode = 'balanced'
        elif '3090' in gpu_name or '4090' in gpu_name:
            gpu_memory = 24
        elif '3080' in gpu_name:
            gpu_memory = 10
        else:
            gpu_memory = 8
            if quality_mode == 'high':
                quality_mode = 'balanced'
        print(f"🖥️  GPU: {gpu_name} ({gpu_memory}GB)")
    else:
        gpu_memory = 0
        quality_mode = 'fast'
        print("🖥️  设备: CPU (强制快速模式)")
    
    # 获取数据加载器并确定图像尺寸
    dataloader, dataset, image_size = get_improved_high_res_dataloader(
        dataset_name=dataset_name,
        target_size=target_size,
        use_upsampling=use_upsampling,
        batch_size=batch_size or 16
    )
    
    # 智能批次大小调整
    if batch_size is None:
        memory_factor = gpu_memory / 12  # 以12GB为基准
        if image_size <= 32:
            batch_size = max(4, int(32 * memory_factor))
        elif image_size <= 64:
            batch_size = max(4, int(16 * memory_factor))
        elif image_size <= 96:
            batch_size = max(2, int(8 * memory_factor))
        else:
            batch_size = max(2, int(4 * memory_factor))
    
    # 重新创建dataloader with正确的batch_size
    dataloader, dataset, image_size = get_improved_high_res_dataloader(
        dataset_name=dataset_name,
        target_size=target_size,
        use_upsampling=use_upsampling,
        batch_size=batch_size
    )
    
    # 获取改进的模型配置
    model_config = get_improved_model_config(image_size, quality_mode)
    
    print(f"🏗️  改进的模型配置:")
    print(f"   - 质量模式: {quality_mode}")
    print(f"   - 批次大小: {batch_size}")
    print(f"   - 模型维度: {model_config['dim']}")
    print(f"   - 层级倍数: {model_config['dim_mults']}")
    print(f"   - 训练轮数: {epochs}")
    print(f"   - 热身轮数: {warmup_epochs}")
    
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
            compiled_unet = torch.compile(model.unet, mode='reduce-overhead')
            model.unet = compiled_unet  # type: ignore
            print("✅ 模型编译成功")
        except Exception as e:
            print(f"⚠️ 模型编译失败: {e}")
    
    # 改进的优化器配置
    optimizer = optim.AdamW(
        model.unet.parameters(),
        lr=learning_rate,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # 改进的学习率调度策略
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, 
        start_factor=0.1, 
        end_factor=1.0, 
        total_iters=warmup_epochs
    )
    
    main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=epochs - warmup_epochs, 
        eta_min=learning_rate * 0.01
    )
    
    # 混合精度
    scaler = GradScaler() if use_amp and AMP_AVAILABLE else None
    if scaler:
        print("✅ 混合精度训练启用")
    
    # 创建保存目录
    exp_name = f"{dataset_name}_improved_{image_size}x{image_size}_{quality_mode}"
    os.makedirs(f'checkpoints/{exp_name}', exist_ok=True)
    os.makedirs(f'samples/{exp_name}', exist_ok=True)
    os.makedirs(f'training_progress/{exp_name}', exist_ok=True)
    
    print(f"📁 实验名称: {exp_name}")
    print("=" * 70)
    
    # 训练循环
    losses = []
    best_loss = float('inf')
    training_start_time = time.time()
    
    for epoch in range(epochs):
        model.unet.train()
        epoch_losses = []
        epoch_start_time = time.time()
        
        # 学习率调度
        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            main_scheduler.step()
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')
        for batch_idx, (data, _) in enumerate(pbar):
            optimizer.zero_grad()
            
            if use_amp and scaler is not None:
                with autocast():
                    loss = model.train_step(data)
                scaler.scale(loss).backward()
                
                # 梯度裁剪
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.unet.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = model.train_step(data)
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.unet.parameters(), max_norm=1.0)
                
                optimizer.step()
            
            epoch_losses.append(loss.item())
            
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{current_lr:.2e}',
                'mode': quality_mode
            })
        
        # Epoch统计
        epoch_time = time.time() - epoch_start_time
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        
        print(f'📊 Epoch {epoch+1}/{epochs}:')
        print(f'   ⏱️  用时: {epoch_time:.2f}秒')
        print(f'   📉 平均损失: {avg_loss:.4f}')
        print(f'   📈 学习率: {current_lr:.2e}')
        
        # 改进的质量评估
        if avg_loss > 0.8:
            quality = "🔴 初始噪声阶段"
        elif avg_loss > 0.5:
            quality = "🟡 基础结构形成"
        elif avg_loss > 0.3:
            quality = "🟢 形状细节显现"
        elif avg_loss > 0.15:
            quality = "🔵 可辨识物体"
        elif avg_loss > 0.08:
            quality = "🟣 高质量细节"
        else:
            quality = "⭐ 超高质量"
        
        print(f'   🎯 质量评估: {quality}')
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = f'checkpoints/{exp_name}/ddpm_best.pt'
            model.save_model(best_path)
            print(f'   🏆 新的最佳模型已保存 (Loss: {best_loss:.4f})')
        
        # 改进的采样策略：早期更频繁，后期less frequent
        sample_freq = early_sample_freq if epoch < epochs // 3 else 5
        if (epoch + 1) % sample_freq == 0:
            model.unet.eval()
            with torch.no_grad():
                # 多样化采样
                sample_count = min(9, batch_size)
                quick_samples = model.sample(batch_size=sample_count)
                save_improved_samples(
                    quick_samples,
                    f'training_progress/{exp_name}/quick_epoch_{epoch+1}.png',
                    epoch=epoch+1,
                    loss=avg_loss,
                    image_size=image_size,
                    quality=quality,
                    dataset_name=dataset_name
                )
            print(f'   📸 快速采样已保存')
        
        # 定期保存和详细采样
        if (epoch + 1) % save_interval == 0:
            checkpoint_path = f'checkpoints/{exp_name}/ddpm_epoch_{epoch+1}.pt'
            model.save_model(checkpoint_path)
            
            model.unet.eval()
            with torch.no_grad():
                # 生成更多样本用于质量评估
                samples = model.sample(batch_size=16)
                save_improved_samples(
                    samples,
                    f'samples/{exp_name}/epoch_{epoch+1}_samples.png',
                    epoch=epoch+1,
                    loss=avg_loss,
                    image_size=image_size,
                    quality=quality,
                    dataset_name=dataset_name
                )
                
                # 生成额外的对比样本
                if epoch > epochs // 2:  # 后期进行quality对比
                    try:
                        # 生成更多样本用于对比
                        extra_samples = model.sample(batch_size=4)
                        save_improved_samples(
                            extra_samples,
                            f'samples/{exp_name}/epoch_{epoch+1}_extra.png',
                            epoch=epoch+1,
                            loss=avg_loss,
                            image_size=image_size,
                            quality=quality + " (额外样本)",
                            dataset_name=dataset_name
                        )
                    except Exception as e:
                        print(f"   ⚠️ 额外采样失败: {e}")
            
            plot_improved_losses(losses, f'samples/{exp_name}/loss_curve_epoch_{epoch+1}.png', exp_name)
            print(f'   💾 完整检查点已保存')
        
        # 损失趋势分析
        if len(losses) >= 10:
            recent_trend = np.mean(losses[-5:]) - np.mean(losses[-10:-5])
            if recent_trend > 0.01:
                print(f'   ⚠️ 损失上升趋势检测到 (+{recent_trend:.4f})')
            elif recent_trend < -0.01:
                print(f'   ✅ 损失下降良好 ({recent_trend:.4f})')
        
        print("-" * 60)
    
    # 训练完成
    total_time = time.time() - training_start_time
    final_path = f'checkpoints/{exp_name}/ddpm_final.pt'
    model.save_model(final_path)
    
    # 最终质量评估
    model.unet.eval()
    with torch.no_grad():
        print("🎯 生成最终质量评估样本...")
        final_samples = model.sample(batch_size=25)  # 5x5网格
        save_improved_samples(
            final_samples,
            f'samples/{exp_name}/final_quality_assessment.png',
            epoch=epochs,
            loss=best_loss,
            image_size=image_size,
            quality="最终质量评估",
            dataset_name=dataset_name
        )
    
    print("🎉 改进的高分辨率训练完成！")
    print(f"⏱️  总训练时间: {total_time/3600:.2f}小时")
    print(f"📉 最佳损失: {best_loss:.4f}")
    print(f"💾 模型已保存到: {final_path}")
    print(f"🖼️  最终样本: samples/{exp_name}/final_quality_assessment.png")
    
    return model, losses

def save_improved_samples(samples, path, epoch=None, loss=None, image_size=64, quality="", dataset_name=""):
    """保存改进的样本，增强可视化"""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        samples = (samples + 1) / 2
        samples = torch.clamp(samples, 0, 1)
        
        # 动态调整网格
        if len(samples) <= 4:
            nrow = 2
        elif len(samples) <= 9:
            nrow = 3
        elif len(samples) <= 16:
            nrow = 4
        else:
            nrow = 5
        
        grid = torchvision.utils.make_grid(samples, nrow=nrow, padding=3, pad_value=1.0)
        grid_np = grid.permute(1, 2, 0).cpu().numpy()
        
        # 动态调整图像大小
        fig_size = max(10, min(16, image_size // 8))
        plt.figure(figsize=(fig_size, fig_size))
        plt.imshow(grid_np)
        plt.axis('off')
        
        # 改进的标题
        title_parts = []
        if dataset_name:
            title_parts.append(f'{dataset_name.upper()}')
        if epoch is not None:
            title_parts.append(f'Epoch {epoch}')
        if loss is not None:
            title_parts.append(f'Loss: {loss:.4f}')
        title_parts.append(f'({image_size}x{image_size})')
        if quality:
            title_parts.append(f'\n{quality}')
        
        title = ' - '.join(title_parts[:4])
        if len(title_parts) > 4:
            title += title_parts[4]
        
        plt.title(title, fontsize=12, fontweight='bold', pad=20)
        
        # 高质量保存
        dpi = 300 if image_size >= 96 else 200
        plt.savefig(path, bbox_inches='tight', dpi=dpi, facecolor='white')
        plt.close()
        
    except Exception as e:
        print(f"保存样本时出错: {e}")

def plot_improved_losses(losses, path, exp_name):
    """绘制改进的训练损失曲线"""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        plt.figure(figsize=(14, 8))
        
        # 原始损失曲线
        plt.subplot(2, 1, 1)
        plt.plot(losses, linewidth=2, color='blue', alpha=0.7, label='训练损失')
        
        # 添加移动平均
        if len(losses) > 10:
            window = min(10, len(losses) // 5)
            moving_avg = np.convolve(losses, np.ones(window)/window, mode='valid')
            plt.plot(range(window-1, len(losses)), moving_avg, 
                    linewidth=3, color='red', label=f'{window}轮移动平均')
        
        plt.title(f'Training Loss - {exp_name}', fontsize=16, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # 添加质量阶段标记
        plt.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='初始噪声')
        plt.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='基础结构')
        plt.axhline(y=0.3, color='yellow', linestyle='--', alpha=0.5, label='形状细节')
        plt.axhline(y=0.15, color='green', linestyle='--', alpha=0.5, label='可辨识物体')
        plt.axhline(y=0.08, color='blue', linestyle='--', alpha=0.5, label='高质量')
        
        # 损失改善率
        plt.subplot(2, 1, 2)
        if len(losses) > 1:
            improvements = np.diff(losses)
            plt.plot(improvements, linewidth=2, color='green', alpha=0.7)
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            plt.title('Loss Improvement Rate', fontsize=14)
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Loss Change', fontsize=12)
            plt.grid(True, alpha=0.3)
        
        # 标记最佳点
        if losses:
            min_loss_idx = np.argmin(losses)
            min_loss = losses[min_loss_idx]
            plt.subplot(2, 1, 1)
            plt.scatter([min_loss_idx], [min_loss], color='red', s=100, zorder=5)
            plt.annotate(f'Best: {min_loss:.4f}', 
                        xy=(float(min_loss_idx), float(min_loss)),
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(path, bbox_inches='tight', dpi=150, facecolor='white')
        plt.close()
        
    except Exception as e:
        print(f"保存损失曲线时出错: {e}")

if __name__ == "__main__":
    import sys
    
    print("🎯 改进的高分辨率DDPM训练")
    print("专门解决生成效果不理想的问题")
    print()
    print("推荐配置:")
    print("  1. STL-10平衡模式: python train_improved_high_res.py stl10 --quality balanced")
    print("  2. STL-10高质量: python train_improved_high_res.py stl10 --quality high")
    print("  3. CelebA人脸: python train_improved_high_res.py celeba --quality balanced")
    print()
    
    # 解析参数
    dataset_name = sys.argv[1] if len(sys.argv) > 1 else 'stl10'
    
    # 解析质量模式
    quality_mode = 'balanced'
    if '--quality' in sys.argv:
        quality_idx = sys.argv.index('--quality')
        if quality_idx + 1 < len(sys.argv):
            quality_mode = sys.argv[quality_idx + 1]
    
    # 解析训练轮数
    epochs = 150
    if '--epochs' in sys.argv:
        epochs_idx = sys.argv.index('--epochs')
        if epochs_idx + 1 < len(sys.argv):
            epochs = int(sys.argv[epochs_idx + 1])
    
    print(f"选择的数据集: {dataset_name}")
    print(f"质量模式: {quality_mode}")
    print(f"训练轮数: {epochs}")
    print("=" * 60)
    
    try:
        model, losses = train_improved_high_res_ddpm(
            dataset_name=dataset_name,
            epochs=epochs,
            quality_mode=quality_mode,
            use_amp='--no-amp' not in sys.argv,
            compile_model='--no-compile' not in sys.argv,
            progressive_training='--progressive' in sys.argv
        )
        
        print("🎉 改进的高分辨率训练成功完成！")
        print("💡 建议检查最终质量评估图像以验证效果")
        
    except Exception as e:
        print(f"❌ 训练过程中出错: {e}")
        import traceback
        traceback.print_exc()