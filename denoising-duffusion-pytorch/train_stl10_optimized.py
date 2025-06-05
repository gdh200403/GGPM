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

def get_stl10_optimized_dataloader(batch_size=8, num_workers=None):
    """获取优化的STL-10数据加载器"""
    
    if num_workers is None:
        num_workers = min(8, os.cpu_count() or 4)
    
    print(f"📊 STL-10优化数据集配置:")
    print(f"   - 原生分辨率: 96x96")
    print(f"   - 批次大小: {batch_size}")
    print(f"   - 工作进程: {num_workers}")
    
    # STL-10特化的数据预处理
    transform = transforms.Compose([
        # STL-10已经是96x96，无需resize
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=5),  # 轻微旋转
        # 轻微的颜色增强 - STL-10图像质量较高，增强要温和
        transforms.ColorJitter(
            brightness=0.05,  # 很轻微的亮度调整
            contrast=0.05,    # 很轻微的对比度调整
            saturation=0.05,  # 很轻微的饱和度调整
            hue=0.02         # 很轻微的色调调整
        ),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化到[-1,1]
    ])
    
    # 创建STL-10数据集
    dataset = torchvision.datasets.STL10(
        root='./data',
        split='train',  # 使用训练集
        download=True,
        transform=transform
    )
    
    print(f"   - 训练样本数: {len(dataset)}")
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=3,  # 增加预取因子
        drop_last=True
    )
    
    return dataloader, dataset

def train_stl10_optimized_ddpm(
    epochs=100,  # 增加训练轮数
    batch_size=None,
    learning_rate=1e-4,
    save_interval=10,
    use_amp=True,
    compile_model=True,
    use_ema=True,  # 使用指数移动平均
    warmup_epochs=5  # 学习率热身
):
    """优化的STL-10 DDPM训练"""
    
    print("🎯 STL-10优化DDPM训练")
    print("=" * 60)
    
    # 获取GPU信息并调整批次大小
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        if '3060' in gpu_name:
            gpu_memory = 12
            recommended_batch = 6  # 保守一些
        elif '3090' in gpu_name or '4090' in gpu_name:
            gpu_memory = 24
            recommended_batch = 12
        elif '3080' in gpu_name:
            gpu_memory = 10
            recommended_batch = 6
        else:
            gpu_memory = 8
            recommended_batch = 4
        print(f"🖥️  GPU: {gpu_name} ({gpu_memory}GB)")
    else:
        gpu_memory = 0
        recommended_batch = 2
        print("🖥️  设备: CPU")
    
    if batch_size is None:
        batch_size = recommended_batch
    
    # 获取数据加载器
    dataloader, dataset = get_stl10_optimized_dataloader(batch_size=batch_size)
    
    # STL-10专用的强化模型配置
    model_config = {
        'dim': 256,  # 显著增加基础维度
        'dim_mults': (1, 1, 2, 2, 4, 4, 8),  # 更深的层级结构
        # 'flash_attn': True,  # 如果可用的话
    }
    
    print(f"🏗️  STL-10优化模型配置:")
    print(f"   - 批次大小: {batch_size}")
    print(f"   - 模型维度: {model_config['dim']}")
    print(f"   - 层级倍数: {model_config['dim_mults']}")
    print(f"   - 总轮数: {epochs}")
    
    # 创建模型
    model = DDPMModel(
        image_size=96,
        channels=3,
        timesteps=1000,  # 保持1000步
        **model_config
    )
    
    total_params = sum(p.numel() for p in model.unet.parameters())
    print(f"   - 模型参数: {total_params:,}")
    
    # 模型编译
    if compile_model and hasattr(torch, 'compile'):
        try:
            compiled_unet = torch.compile(model.unet, mode='reduce-overhead')
            model.unet = compiled_unet  # type: ignore
            print("✅ 模型编译成功 (reduce-overhead模式)")
        except Exception as e:
            print(f"⚠️ 模型编译失败: {e}")
    
    # 优化器配置
    optimizer = optim.AdamW(
        model.unet.parameters(),
        lr=learning_rate,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # 更复杂的学习率调度
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, 
        start_factor=0.1, 
        end_factor=1.0, 
        total_iters=warmup_epochs
    )
    
    main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=epochs - warmup_epochs, 
        eta_min=learning_rate * 0.01  # 更低的最小学习率
    )
    
    # 指数移动平均（EMA）
    if use_ema:
        try:
            from torch_ema import ExponentialMovingAverage
            ema = ExponentialMovingAverage(model.unet.parameters(), decay=0.995)
            print("✅ EMA初始化成功")
        except ImportError:
            print("⚠️ torch_ema未安装，跳过EMA")
            print("💡 可以通过 pip install torch-ema 安装")
            use_ema = False
            ema = None
    else:
        ema = None
    
    # 混合精度
    scaler = GradScaler() if use_amp and AMP_AVAILABLE else None
    if scaler:
        print("✅ 混合精度训练启用")
    
    # 创建保存目录
    exp_name = "stl10_optimized_96x96"
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
            
            # 更新EMA
            if use_ema and ema is not None:
                ema.update()
            
            epoch_losses.append(loss.item())
            
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{current_lr:.2e}',
                'ema': 'ON' if use_ema else 'OFF'
            })
        
        # Epoch统计
        epoch_time = time.time() - epoch_start_time
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        
        print(f'📊 Epoch {epoch+1}/{epochs}:')
        print(f'   ⏱️  用时: {epoch_time:.2f}秒')
        print(f'   📉 平均损失: {avg_loss:.4f}')
        print(f'   📈 学习率: {current_lr:.2e}')
        
        # 损失质量评估
        if avg_loss > 0.5:
            quality = "🔴 纯噪声阶段"
        elif avg_loss > 0.3:
            quality = "🟡 形状形成阶段"
        elif avg_loss > 0.15:
            quality = "🟢 可辨识物体阶段"
        elif avg_loss > 0.08:
            quality = "🔵 清晰图像阶段"
        else:
            quality = "🟣 高质量阶段"
        
        print(f'   🎯 质量评估: {quality}')
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = f'checkpoints/{exp_name}/ddpm_best.pt'
            
            # 使用EMA权重保存最佳模型
            if use_ema and ema is not None:
                with ema.average_parameters():
                    model.save_model(best_path)
            else:
                model.save_model(best_path)
            print(f'   🏆 新的最佳模型已保存 (Loss: {best_loss:.4f})')
        
        # 定期采样
        if (epoch + 1) % 5 == 0:
            model.unet.eval()
            with torch.no_grad():
                # 使用EMA权重进行采样
                if use_ema and ema is not None:
                    with ema.average_parameters():
                        quick_samples = model.sample(batch_size=9)
                else:
                    quick_samples = model.sample(batch_size=9)
                
                save_stl10_samples(
                    quick_samples,
                    f'training_progress/{exp_name}/quick_epoch_{epoch+1}.png',
                    epoch=epoch+1,
                    loss=avg_loss,
                    quality=quality
                )
            print(f'   📸 快速采样已保存')
        
        # 检查点保存
        if (epoch + 1) % save_interval == 0:
            checkpoint_path = f'checkpoints/{exp_name}/ddpm_epoch_{epoch+1}.pt'
            
            # 保存常规权重
            model.save_model(checkpoint_path)
            
            # 如果使用EMA，也保存EMA权重
            if use_ema and ema is not None:
                ema_path = f'checkpoints/{exp_name}/ddpm_ema_epoch_{epoch+1}.pt'
                with ema.average_parameters():
                    model.save_model(ema_path)
            
            # 生成高质量样本
            model.unet.eval()
            with torch.no_grad():
                if use_ema and ema is not None:
                    with ema.average_parameters():
                        samples = model.sample(batch_size=16)
                else:
                    samples = model.sample(batch_size=16)
                
                save_stl10_samples(
                    samples,
                    f'samples/{exp_name}/epoch_{epoch+1}_samples.png',
                    epoch=epoch+1,
                    loss=avg_loss,
                    quality=quality
                )
            
            plot_stl10_losses(losses, f'samples/{exp_name}/loss_curve_epoch_{epoch+1}.png', exp_name)
            print(f'   💾 完整检查点已保存')
        
        print("-" * 60)
    
    # 训练完成
    total_time = time.time() - training_start_time
    final_path = f'checkpoints/{exp_name}/ddpm_final.pt'
    
    # 使用EMA权重保存最终模型
    if use_ema and ema is not None:
        with ema.average_parameters():
            model.save_model(final_path)
        
        # 也保存非EMA版本
        regular_path = f'checkpoints/{exp_name}/ddpm_final_regular.pt'
        model.save_model(regular_path)
    else:
        model.save_model(final_path)
    
    print("🎉 STL-10优化训练完成！")
    print(f"⏱️  总训练时间: {total_time/3600:.2f}小时")
    print(f"📉 最佳损失: {best_loss:.4f}")
    print(f"💾 最终模型已保存到: {final_path}")
    
    return model, losses

def save_stl10_samples(samples, path, epoch=None, loss=None, quality=""):
    """保存STL-10样本"""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        samples = (samples + 1) / 2
        samples = torch.clamp(samples, 0, 1)
        
        nrow = 3 if len(samples) <= 9 else 4
        grid = torchvision.utils.make_grid(samples, nrow=nrow, padding=2)
        grid_np = grid.permute(1, 2, 0).cpu().numpy()
        
        plt.figure(figsize=(12, 12))
        plt.imshow(grid_np)
        plt.axis('off')
        
        title = f'STL-10 Epoch {epoch} - Loss: {loss:.4f} (96x96)'
        if quality:
            title += f'\n{quality}'
        
        if epoch is not None and loss is not None:
            plt.title(title, fontsize=14, fontweight='bold', pad=20)
        
        plt.savefig(path, bbox_inches='tight', dpi=200)
        plt.close()
        
    except Exception as e:
        print(f"保存STL-10样本时出错: {e}")

def plot_stl10_losses(losses, path, exp_name):
    """绘制STL-10训练损失"""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        plt.figure(figsize=(12, 8))
        plt.plot(losses, linewidth=2, color='blue')
        plt.title(f'STL-10 Training Loss - {exp_name}', fontsize=16, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # 添加质量阶段标记
        plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='纯噪声阶段')
        plt.axhline(y=0.3, color='orange', linestyle='--', alpha=0.7, label='形状形成阶段')
        plt.axhline(y=0.15, color='green', linestyle='--', alpha=0.7, label='可辨识物体阶段')
        plt.axhline(y=0.08, color='blue', linestyle='--', alpha=0.7, label='清晰图像阶段')
        
        if losses:
            min_loss_idx = np.argmin(losses)
            min_loss = losses[min_loss_idx]
            plt.scatter([min_loss_idx], [min_loss], color='red', s=100, zorder=5)
            plt.annotate(f'Best: {min_loss:.4f}', 
                        xy=(float(min_loss_idx), float(min_loss)),
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7))
        
        plt.legend()
        plt.tight_layout()
        plt.savefig(path, bbox_inches='tight', dpi=150)
        plt.close()
        
    except Exception as e:
        print(f"保存STL-10损失曲线时出错: {e}")

if __name__ == "__main__":
    import sys
    
    print("🎯 STL-10优化DDPM训练")
    print("专门针对STL-10 (96x96) 的增强配置")
    print("=" * 70)
    
    # 解析参数
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else None
    
    print(f"训练轮数: {epochs}")
    print(f"批次大小: {batch_size or '自动'}")
    print("=" * 70)
    
    try:
        model, losses = train_stl10_optimized_ddpm(
            epochs=epochs,
            batch_size=batch_size,
            use_amp='--no-amp' not in sys.argv,
            compile_model='--no-compile' not in sys.argv,
            use_ema='--no-ema' not in sys.argv
        )
        
        print("🎉 STL-10优化训练成功完成！")
        print("💡 建议运行推理查看生成质量：")
        print("   python inference_high_res.py --model checkpoints/stl10_optimized_96x96/ddpm_best.pt")
        
    except Exception as e:
        print(f"❌ 训练过程中出错: {e}")
        import traceback
        traceback.print_exc() 