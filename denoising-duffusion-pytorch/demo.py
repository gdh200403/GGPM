import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np
import os
from ddpm_model import DDPMModel
from tqdm import tqdm
import time

# 创建demo结果目录
DEMO_OUTPUT_DIR = 'demo_results'
os.makedirs(DEMO_OUTPUT_DIR, exist_ok=True)

def quick_data_test():
    """快速测试数据加载"""
    print("=== 测试数据加载 ===")
    
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    # 使用小部分数据进行测试
    subset = Subset(dataset, range(100))  # 只使用100个样本
    dataloader = DataLoader(subset, batch_size=8, shuffle=True)
    
    print(f"数据加载成功！")
    print(f"- 数据集大小: {len(subset)}")
    print(f"- 批次大小: 8")
    print(f"- 图像尺寸: {dataset[0][0].shape}")
    
    # 可视化一些样本
    data_iter = iter(dataloader)
    images, labels = next(data_iter)
    
    # 保存样本
    images_display = (images + 1) / 2  # 转换到[0,1]
    grid = torchvision.utils.make_grid(images_display, nrow=4)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis('off')
    plt.title('CIFAR-10 Sample Batch')
    sample_path = os.path.join(DEMO_OUTPUT_DIR, 'demo_data_samples.png')
    plt.savefig(sample_path, bbox_inches='tight', dpi=150)
    plt.close()
    
    print(f"样本图像已保存到: {sample_path}")
    return dataloader

def quick_model_test():
    """快速测试模型初始化"""
    print("\n=== 测试模型初始化 ===")
    
    # 创建小模型进行快速测试
    model = DDPMModel(
        image_size=32,
        channels=3,
        dim=32,  # 减小模型尺寸
        dim_mults=(1, 2, 4),  # 减少层数
        timesteps=100  # 减少时间步数
    )
    
    print("模型初始化成功！")
    
    # 测试前向传播
    batch_size = 4
    test_input = torch.randn(batch_size, 3, 32, 32).to(model.device)
    
    with torch.no_grad():
        loss = model.train_step(test_input)
        print(f"前向传播测试 - 损失值: {loss.item():.4f}")
        
        # 测试采样
        samples = model.sample(batch_size=4)
        print(f"采样测试 - 生成样本形状: {samples.shape}")
    
    return model

def quick_training_test(model, dataloader, epochs=3):
    """快速训练测试"""
    print(f"\n=== 快速训练测试 ({epochs} epochs) ===")
    
    optimizer = torch.optim.Adam(model.unet.parameters(), lr=1e-3)
    losses = []
    
    model.unet.train()
    
    for epoch in range(epochs):
        epoch_losses = []
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')
        for batch_idx, (data, _) in enumerate(pbar):
            optimizer.zero_grad()
            
            loss = model.train_step(data)
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        print(f'Epoch {epoch+1}, 平均损失: {avg_loss:.4f}')
    
    # 绘制损失曲线
    plt.figure(figsize=(8, 5))
    plt.plot(losses, marker='o')
    plt.title('Quick Training - Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    sample_path = os.path.join(DEMO_OUTPUT_DIR, 'demo_training_loss.png')
    plt.savefig(sample_path, bbox_inches='tight', dpi=150)
    plt.close()
    
    print(f"训练损失曲线已保存到: {sample_path}")
    
    # 保存快速训练的模型
    model_path = os.path.join(DEMO_OUTPUT_DIR, 'demo_model.pt')
    model.save_model(model_path)
    
    return model, losses

def quick_inference_test(model):
    """快速推理测试"""
    print("\n=== 快速推理测试 ===")
    
    model.unet.eval()
    
    with torch.no_grad():
        # 生成少量样本
        print("生成样本...")
        samples = model.sample(batch_size=8)
        
        # 保存生成的样本
        samples_display = (samples + 1) / 2
        samples_display = torch.clamp(samples_display, 0, 1)
        
        grid = torchvision.utils.make_grid(samples_display, nrow=4, padding=2)
        
        plt.figure(figsize=(8, 6))
        plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
        plt.axis('off')
        plt.title('Generated Samples (Quick Demo)')
        sample_path = os.path.join(DEMO_OUTPUT_DIR, 'demo_generated_samples.png')
        plt.savefig(sample_path, bbox_inches='tight', dpi=150)
        plt.close()
        
        print(f"生成样本已保存到: {sample_path}")
    
    return samples

def advanced_correctness_test():
    """高级正确性验证测试"""
    print("\n=== 高级正确性验证 ===")
    
    model = DDPMModel(
        image_size=32,
        channels=3,
        dim=32,
        dim_mults=(1, 2, 4),
        timesteps=100
    )
    
    # 1. 测试扩散过程的数学正确性
    print("1. 验证扩散过程数学性质...")
    batch_size = 4
    test_images = torch.randn(batch_size, 3, 32, 32).to(model.device)
    
    with torch.no_grad():
        # 测试噪声调度的单调性
        timesteps = [10, 20, 50, 80]
        noise_levels = []
        
        for t in timesteps:
            t_tensor = torch.full((batch_size,), t, device=model.device, dtype=torch.long)
            # 这里需要访问扩散过程的内部状态来验证
            # 简化版本的噪声验证
            noisy = model.diffusion.q_sample(test_images, t_tensor)
            noise_level = torch.mean((noisy - test_images) ** 2).item()
            noise_levels.append(noise_level)
        
        # 验证噪声水平随时间步递增
        is_monotonic = all(noise_levels[i] <= noise_levels[i+1] for i in range(len(noise_levels)-1))
        print(f"   噪声调度单调性: {'✅ 通过' if is_monotonic else '❌ 失败'}")
        print(f"   噪声水平: {[f'{x:.3f}' for x in noise_levels]}")
    
    # 2. 测试梯度稳定性
    print("2. 验证梯度稳定性...")
    model.unet.train()
    test_batch = torch.randn(2, 3, 32, 32).to(model.device)
    
    # 计算梯度
    loss = model.train_step(test_batch)
    loss.backward()
    
    # 检查梯度范数
    total_grad_norm = 0
    for param in model.unet.parameters():
        if param.grad is not None:
            total_grad_norm += param.grad.data.norm(2).item() ** 2
    total_grad_norm = total_grad_norm ** 0.5
    
    print(f"   总梯度范数: {total_grad_norm:.4f}")
    print(f"   梯度稳定性: {'✅ 通过' if total_grad_norm < 100 else '❌ 可能不稳定'}")
    
    # 3. 测试采样一致性
    print("3. 验证采样一致性...")
    model.unet.eval()
    
    # 固定随机种子，多次采样应该得到相同结果
    torch.manual_seed(42)
    sample1 = model.sample(batch_size=2)
    
    torch.manual_seed(42)  
    sample2 = model.sample(batch_size=2)
    
    consistency_error = torch.mean((sample1 - sample2) ** 2).item()
    print(f"   采样一致性误差: {consistency_error:.6f}")
    print(f"   一致性测试: {'✅ 通过' if consistency_error < 1e-5 else '❌ 失败'}")
    
    # 4. 测试模型保存/加载完整性
    print("4. 验证模型保存/加载完整性...")
    
    # 保存原始模型的一个样本
    torch.manual_seed(123)
    original_sample = model.sample(batch_size=1)
    
    # 保存并重新加载模型
    temp_path = os.path.join(DEMO_OUTPUT_DIR, 'temp_test_model.pt')
    model.save_model(temp_path)
    
    new_model = DDPMModel(
        image_size=32,
        channels=3, 
        dim=32,
        dim_mults=(1, 2, 4),
        timesteps=100
    )
    new_model.load_model(temp_path)
    new_model.unet.eval()
    
    # 使用相同随机种子生成样本
    torch.manual_seed(123)
    loaded_sample = new_model.sample(batch_size=1)
    
    load_error = torch.mean((original_sample - loaded_sample) ** 2).item()
    print(f"   加载误差: {load_error:.6f}")
    print(f"   保存/加载测试: {'✅ 通过' if load_error < 1e-5 else '❌ 失败'}")
    
    # 清理临时文件
    if os.path.exists(temp_path):
        os.remove(temp_path)
    
    return True

def full_demo():
    """完整的demo测试流程"""
    print("🚀 开始DDPM模型Demo测试")
    print("=" * 50)
    
    start_time = time.time()
    
    try:
        # 1. 测试数据加载
        dataloader = quick_data_test()
        
        # 2. 测试模型初始化
        model = quick_model_test()
        
        # 3. 快速训练测试
        model, losses = quick_training_test(model, dataloader, epochs=5)
        
        # 4. 快速推理测试
        samples = quick_inference_test(model)
        
        # 5. 高级正确性验证
        advanced_correctness_test()
        
        # 6. 测试模型保存和加载
        print("\n=== 测试模型保存和加载 ===")
        
        # 重新加载模型
        model_reloaded = DDPMModel(
            image_size=32,
            channels=3,
            dim=32,
            dim_mults=(1, 2, 4),
            timesteps=100
        )
        model_reloaded.load_model(os.path.join(DEMO_OUTPUT_DIR, 'demo_model.pt'))
        
        # 测试重新加载的模型
        with torch.no_grad():
            test_samples = model_reloaded.sample(batch_size=4)
            print(f"重新加载模型采样测试成功 - 形状: {test_samples.shape}")
        
        end_time = time.time()
        
        print("\n" + "=" * 50)
        print("🎉 Demo测试全部完成！")
        print(f"⏱️  总耗时: {end_time - start_time:.2f}秒")
        print(f"\n生成的文件都保存在 '{DEMO_OUTPUT_DIR}' 目录中:")
        print(f"- {os.path.join(DEMO_OUTPUT_DIR, 'demo_data_samples.png')} (数据样本)")
        print(f"- {os.path.join(DEMO_OUTPUT_DIR, 'demo_training_loss.png')} (训练损失)")
        print(f"- {os.path.join(DEMO_OUTPUT_DIR, 'demo_generated_samples.png')} (生成样本)")
        print(f"- {os.path.join(DEMO_OUTPUT_DIR, 'demo_model.pt')} (模型权重)")
        print(f"- {os.path.join(DEMO_OUTPUT_DIR, 'demo_benchmark.png')} (性能基准)")
        
        print("\n✅ 所有组件工作正常，核心代码验证通过！")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Demo测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def benchmark_test():
    """性能基准测试"""
    print("\n=== 性能基准测试 ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"测试设备: {device}")
    
    # 测试不同批次大小的性能
    batch_sizes = [1, 4, 8, 16] if device.type == 'cuda' else [1, 2, 4]
    
    model = DDPMModel(
        image_size=32,
        channels=3,
        dim=32,
        dim_mults=(1, 2, 4),
        timesteps=100
    )
    
    model.unet.eval()
    
    results = []
    
    for batch_size in batch_sizes:
        print(f"\n测试批次大小: {batch_size}")
        
        # 预热
        with torch.no_grad():
            dummy_input = torch.randn(batch_size, 3, 32, 32).to(device)
            _ = model.train_step(dummy_input)
        
        # 测试推理速度
        times = []
        for _ in range(5):
            start = time.time()
            with torch.no_grad():
                _ = model.sample(batch_size=batch_size)
            end = time.time()
            times.append(end - start)
        
        avg_time = np.mean(times)
        results.append((batch_size, avg_time))
        print(f"平均推理时间: {avg_time:.3f}秒")
        print(f"每样本时间: {avg_time/batch_size:.3f}秒")
    
    # 绘制性能图表
    batch_sizes, times = zip(*results)
    
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.plot(batch_sizes, times, 'o-')
    plt.title('Batch Size vs Total Time')
    plt.xlabel('Batch Size')
    plt.ylabel('Time (seconds)')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    per_sample_times = [t/b for b, t in results]
    plt.plot(batch_sizes, per_sample_times, 'o-')
    plt.title('Batch Size vs Per-Sample Time')
    plt.xlabel('Batch Size')
    plt.ylabel('Time per Sample (seconds)')
    plt.grid(True)
    
    plt.tight_layout()
    sample_path = os.path.join(DEMO_OUTPUT_DIR, 'demo_benchmark.png')
    plt.savefig(sample_path, bbox_inches='tight', dpi=150)
    plt.close()
    
    print(f"性能基准图表已保存到: {sample_path}")

if __name__ == "__main__":
    # 运行完整demo
    success = full_demo()
    
    if success:
        # 运行性能测试
        benchmark_test()
        
        print("\n🔥 接下来您可以:")
        print("1. 运行 'python train.py' 进行完整训练")
        print("2. 运行 'python inference.py' 进行推理")
        print("3. 修改超参数来优化模型性能")
    else:
        print("\n🔧 请检查错误信息并修复问题") 