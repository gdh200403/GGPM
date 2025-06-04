import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from ddpm_model import DDPMModel
import time
from PIL import Image

def load_high_res_model(model_path, device='auto'):
    """加载高分辨率模型"""
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"📂 加载模型: {model_path}")
    print(f"🖥️  设备: {device}")
    
    # 尝试从路径推断配置
    filename = os.path.basename(model_path)
    if 'cifar10_64x64' in filename:
        image_size = 64
        config = {'dim': 128, 'dim_mults': (1, 2, 4, 8)}
    elif 'cifar10_128x128' in filename:
        image_size = 128
        config = {'dim': 192, 'dim_mults': (1, 1, 2, 2, 4, 4)}
    elif 'cifar10_256x256' in filename:
        image_size = 256
        config = {'dim': 256, 'dim_mults': (1, 1, 2, 2, 4, 4, 8)}
    else:
        # 默认配置
        image_size = 64
        config = {'dim': 128, 'dim_mults': (1, 2, 4, 8)}
        print("⚠️ 无法从文件名推断配置，使用默认64x64配置")
    
    model = DDPMModel(
        image_size=image_size,
        channels=3,
        timesteps=1000,
        device=device,
        **config
    )
    
    model.load_model(model_path)
    model.unet.eval()
    
    # 显示模型信息
    total_params = sum(p.numel() for p in model.unet.parameters())
    print(f"📊 模型信息:")
    print(f"   - 图像尺寸: {image_size}x{image_size}")
    print(f"   - 参数数量: {total_params:,}")
    print(f"   - 模型维度: {config['dim']}")
    
    return model, image_size

def generate_high_res_samples(
    model, 
    num_samples=16, 
    save_path=None, 
    sampling_steps=None,
    guidance_scale=1.0,
    show_progress=True
):
    """生成高分辨率样本"""
    
    print(f"🎨 开始生成 {num_samples} 个高分辨率样本")
    
    device = model.device
    
    # 根据模型自动调整采样步数
    if sampling_steps is None:
        if model.image_size <= 64:
            sampling_steps = 250
        elif model.image_size <= 128:
            sampling_steps = 500
        else:
            sampling_steps = 1000
    
    print(f"⚙️  采样步数: {sampling_steps}")
    print(f"🎯 引导强度: {guidance_scale}")
    
    start_time = time.time()
    
    with torch.no_grad():
        # 使用模型的diffusion对象进行采样
        samples = model.diffusion.sample(batch_size=num_samples)
    
    generation_time = time.time() - start_time
    print(f"⏱️  生成用时: {generation_time:.2f}秒")
    print(f"🚀 生成速度: {num_samples/generation_time:.1f} 样本/秒")
    
    # 保存样本
    if save_path:
        save_high_quality_samples(samples, save_path, model.image_size)
        print(f"💾 样本已保存到: {save_path}")
    
    return samples

def save_high_quality_samples(samples, path, image_size):
    """保存高质量样本"""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    
    # 转换到[0,1]范围
    samples = (samples + 1) / 2
    samples = torch.clamp(samples, 0, 1)
    
    # 根据样本数量调整网格
    n_samples = len(samples)
    if n_samples <= 4:
        nrow = 2
    elif n_samples <= 9:
        nrow = 3
    elif n_samples <= 16:
        nrow = 4
    elif n_samples <= 25:
        nrow = 5
    else:
        nrow = int(np.sqrt(n_samples))
    
    # 创建网格
    grid = torchvision.utils.make_grid(samples, nrow=nrow, padding=4)
    grid_np = grid.permute(1, 2, 0).cpu().numpy()
    
    # 根据分辨率调整显示大小
    fig_size = min(20, max(10, image_size // 8))
    plt.figure(figsize=(fig_size, fig_size))
    plt.imshow(grid_np)
    plt.axis('off')
    plt.title(f'Generated Samples ({image_size}x{image_size})', 
              fontsize=16, fontweight='bold', pad=20)
    
    # 高质量保存
    dpi = 300 if image_size >= 128 else 200
    plt.savefig(path, bbox_inches='tight', dpi=dpi, facecolor='white')
    plt.close()
    
    # 同时保存单独的图片
    base_name = os.path.splitext(path)[0]
    individual_dir = f"{base_name}_individual"
    os.makedirs(individual_dir, exist_ok=True)
    
    for i, sample in enumerate(samples):
        sample_np = sample.permute(1, 2, 0).cpu().numpy()
        sample_np = (sample_np * 255).astype(np.uint8)
        img = Image.fromarray(sample_np)
        img.save(f"{individual_dir}/sample_{i+1:03d}.png")

def interpolate_high_res(model, num_steps=10, save_path=None):
    """高分辨率插值"""
    print(f"🔄 生成高分辨率插值序列 ({num_steps} 步)")
    
    device = model.device
    
    with torch.no_grad():
        # 生成两个随机样本
        sample1 = model.sample(batch_size=1)
        sample2 = model.sample(batch_size=1)
        
        # 使用模型的插值方法
        interpolated = model.interpolate(sample1, sample2, num_steps=num_steps)
    
    if save_path:
        save_interpolation_grid(interpolated, save_path, model.image_size)
        print(f"💾 插值序列已保存到: {save_path}")
    
    return interpolated

def save_interpolation_grid(samples, path, image_size):
    """保存插值网格"""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    
    samples = (samples + 1) / 2
    samples = torch.clamp(samples, 0, 1)
    
    # 单行网格显示插值过程
    grid = torchvision.utils.make_grid(samples, nrow=len(samples), padding=2)
    grid_np = grid.permute(1, 2, 0).cpu().numpy()
    
    plt.figure(figsize=(len(samples) * 2, 2))
    plt.imshow(grid_np)
    plt.axis('off')
    plt.title(f'Interpolation Sequence ({image_size}x{image_size})', 
              fontsize=14, fontweight='bold')
    
    plt.savefig(path, bbox_inches='tight', dpi=200, facecolor='white')
    plt.close()

def compare_resolutions(model_paths, num_samples=4):
    """比较不同分辨率的生成效果"""
    print("📊 比较不同分辨率的生成效果")
    
    results = {}
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            try:
                model, image_size = load_high_res_model(model_path)
                samples = generate_high_res_samples(
                    model, 
                    num_samples=num_samples, 
                    show_progress=False
                )
                results[f"{image_size}x{image_size}"] = samples
                print(f"✅ {image_size}x{image_size} 模型测试完成")
            except Exception as e:
                print(f"❌ {model_path} 加载失败: {e}")
        else:
            print(f"⚠️ 模型文件不存在: {model_path}")
    
    if results:
        # 保存比较图
        save_comparison_grid(results, "resolution_comparison.png")
        print("💾 分辨率比较图已保存")
    
    return results

def save_comparison_grid(results, path):
    """保存分辨率比较网格"""
    if not results:
        return
    
    fig, axes = plt.subplots(len(results), 4, figsize=(16, 4 * len(results)))
    if len(results) == 1:
        axes = axes.reshape(1, -1)
    
    for row, (resolution, samples) in enumerate(results.items()):
        samples = (samples + 1) / 2
        samples = torch.clamp(samples, 0, 1)
        
        for col in range(min(4, len(samples))):
            sample_np = samples[col].permute(1, 2, 0).cpu().numpy()
            axes[row, col].imshow(sample_np)
            axes[row, col].axis('off')
            if col == 0:
                axes[row, col].set_ylabel(resolution, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight', dpi=200, facecolor='white')
    plt.close()

def main():
    """主函数 - 高分辨率推理示例"""
    import argparse
    
    parser = argparse.ArgumentParser(description='高分辨率DDPM推理')
    parser.add_argument('--model', type=str, required=True, help='模型路径')
    parser.add_argument('--samples', type=int, default=16, help='生成样本数量')
    parser.add_argument('--steps', type=int, default=None, help='采样步数')
    parser.add_argument('--output', type=str, default='high_res_samples.png', help='输出路径')
    parser.add_argument('--interpolate', action='store_true', help='生成插值序列')
    parser.add_argument('--compare', nargs='+', help='比较多个模型')
    
    args = parser.parse_args()
    
    if args.compare:
        # 比较模式
        compare_resolutions(args.compare)
    else:
        # 单模型生成
        model, image_size = load_high_res_model(args.model)
        
        # 生成样本
        samples = generate_high_res_samples(
            model, 
            num_samples=args.samples,
            save_path=args.output,
            sampling_steps=args.steps
        )
        
        # 插值（如果请求）
        if args.interpolate:
            interpolate_path = f"interpolation_{image_size}x{image_size}.png"
            interpolate_high_res(model, save_path=interpolate_path)

if __name__ == "__main__":
    # 如果直接运行，使用交互模式
    print("🎨 高分辨率DDPM推理工具")
    print("=" * 50)
    
    # 查找可用的模型
    checkpoint_dirs = [d for d in os.listdir('.') if d.startswith('checkpoints') and os.path.isdir(d)]
    
    if not checkpoint_dirs:
        print("❌ 未找到checkpoints目录")
        exit(1)
    
    # 扫描高分辨率模型
    high_res_models = []
    for checkpoint_dir in checkpoint_dirs:
        for root, dirs, files in os.walk(checkpoint_dir):
            for file in files:
                if file.endswith('.pt') and any(res in root for res in ['64x64', '128x128', '256x256']):
                    high_res_models.append(os.path.join(root, file))
    
    if not high_res_models:
        print("❌ 未找到高分辨率模型文件")
        print("请先训练高分辨率模型：python train_high_res.py --dataset=cifar10 --size=64")
        exit(1)
    
    print("📂 发现的高分辨率模型:")
    for i, model_path in enumerate(high_res_models):
        print(f"   {i+1}. {model_path}")
    
    try:
        choice = int(input("\n选择模型 (输入编号): ")) - 1
        if 0 <= choice < len(high_res_models):
            model_path = high_res_models[choice]
            
            model, image_size = load_high_res_model(model_path)
            
            print("\n🎯 生成选项:")
            print("1. 生成样本")
            print("2. 生成插值序列") 
            print("3. 批量生成")
            
            option = input("选择操作 (1-3): ")
            
            if option == "1":
                num_samples = int(input("样本数量 (默认16): ") or "16")
                output_path = f"generated_samples_{image_size}x{image_size}.png"
                generate_high_res_samples(model, num_samples=num_samples, save_path=output_path)
                
            elif option == "2":
                interpolate_high_res(model, save_path=f"interpolation_{image_size}x{image_size}.png")
                
            elif option == "3":
                batch_size = int(input("每批样本数 (默认16): ") or "16")
                num_batches = int(input("批次数量 (默认5): ") or "5")
                
                os.makedirs(f"batch_generation_{image_size}x{image_size}", exist_ok=True)
                
                for i in range(num_batches):
                    output_path = f"batch_generation_{image_size}x{image_size}/batch_{i+1}.png"
                    generate_high_res_samples(model, num_samples=batch_size, save_path=output_path)
                    print(f"✅ 批次 {i+1}/{num_batches} 完成")
                
                print("🎉 批量生成完成！")
        else:
            print("❌ 无效选择")
            
    except (ValueError, KeyboardInterrupt):
        print("\n👋 退出程序") 