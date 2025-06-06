import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from ddpm_model import DDPMModel
import time
from PIL import Image

def load_true_high_res_model(model_path, device='auto'):
    """加载真正的高分辨率模型（适配train_true_high_res.py训练的模型）"""
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"📂 加载真正的高分辨率模型: {model_path}")
    print(f"🖥️  设备: {device}")
    
    # 从路径推断配置（适配train_true_high_res.py的命名规则）
    filename = os.path.basename(model_path)
    directory = os.path.dirname(model_path)
    
    # 解析实验名称
    config_detected = False
    
    # 检查目录结构：checkpoints/{dataset}_native_{size}x{size}/
    if 'checkpoints' in directory:
        parts = directory.split('/')
        for part in parts:
            if 'native' in part or 'upsampled' in part:
                exp_name = part
                print(f"🔍 检测到实验名称: {exp_name}")
                
                # 解析数据集和尺寸
                if 'cifar10_native_32x32' in exp_name:
                    image_size = 32
                    dataset = 'cifar10'
                    config = {'dim': 64, 'dim_mults': (1, 2, 4, 8)}
                    config_detected = True
                elif 'stl10_native_96x96' in exp_name:
                    image_size = 96
                    dataset = 'stl10'
                    config = {'dim': 160, 'dim_mults': (1, 2, 4, 8)}
                    config_detected = True
                elif 'celeba_native_64x64' in exp_name:
                    image_size = 64
                    dataset = 'celeba'
                    config = {'dim': 128, 'dim_mults': (1, 2, 4, 8)}
                    config_detected = True
                elif 'native_64x64' in exp_name:
                    image_size = 64
                    dataset = 'unknown'
                    config = {'dim': 128, 'dim_mults': (1, 2, 4, 8)}
                    config_detected = True
                elif 'native_96x96' in exp_name:
                    image_size = 96
                    dataset = 'unknown'
                    config = {'dim': 160, 'dim_mults': (1, 2, 4, 8)}
                    config_detected = True
                elif 'upsampled' in exp_name:
                    # 解析上采样模型
                    size_parts = [p for p in exp_name.split('_') if 'x' in p and p.replace('x', '').replace('x', '').isdigit()]
                    if size_parts:
                        size_str = size_parts[0]
                        image_size = int(size_str.split('x')[0])
                        dataset = exp_name.split('_')[0]
                        
                        if image_size <= 32:
                            config = {'dim': 64, 'dim_mults': (1, 2, 4, 8)}
                        elif image_size <= 64:
                            config = {'dim': 128, 'dim_mults': (1, 2, 4, 8)}
                        elif image_size <= 96:
                            config = {'dim': 160, 'dim_mults': (1, 2, 4, 8)}
                        else:
                            config = {'dim': 192, 'dim_mults': (1, 1, 2, 2, 4, 4)}
                        config_detected = True
                break
    
    # 如果无法从路径推断，尝试从文件名推断
    if not config_detected:
        print("⚠️ 无法从路径推断配置，尝试从文件名推断...")
        
        if 'stl10' in filename.lower():
            image_size = 96
            dataset = 'stl10'
            config = {'dim': 160, 'dim_mults': (1, 2, 4, 8)}
        elif 'celeba' in filename.lower():
            image_size = 64  
            dataset = 'celeba'
            config = {'dim': 128, 'dim_mults': (1, 2, 4, 8)}
        elif 'cifar10' in filename.lower():
            image_size = 32
            dataset = 'cifar10'
            config = {'dim': 64, 'dim_mults': (1, 2, 4, 8)}
        else:
            # 默认配置
            image_size = 64
            dataset = 'unknown'
            config = {'dim': 128, 'dim_mults': (1, 2, 4, 8)}
            print("⚠️ 无法推断配置，使用默认64x64配置")
    
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
    print(f"   - 数据集: {dataset.upper()}")
    print(f"   - 图像尺寸: {image_size}x{image_size}")
    print(f"   - 参数数量: {total_params:,}")
    print(f"   - 模型维度: {config['dim']}")
    print(f"   - 层级倍数: {config['dim_mults']}")
    
    return model, image_size, dataset

def generate_true_high_res_samples(
    model, 
    num_samples=16, 
    save_path=None, 
    sampling_steps=None,
    show_progress=True,
    dataset_name='unknown'
):
    """生成真正的高分辨率样本"""
    
    print(f"🎨 开始生成 {num_samples} 个真正的高分辨率样本")
    
    device = model.device
    
    # 根据模型和数据集自动调整采样步数
    if sampling_steps is None:
        if dataset_name == 'cifar10':
            sampling_steps = 250  # CIFAR-10可以用较少步数
        elif dataset_name == 'stl10':
            sampling_steps = 500  # STL-10需要更多步数
        elif dataset_name == 'celeba':
            sampling_steps = 400  # CelebA中等步数
        else:
            # 根据分辨率决定
            if model.image_size <= 32:
                sampling_steps = 250
            elif model.image_size <= 64:
                sampling_steps = 400
            else:
                sampling_steps = 500
    
    print(f"⚙️  采样步数: {sampling_steps}")
    print(f"📊 数据集类型: {dataset_name.upper()}")
    
    start_time = time.time()
    
    with torch.no_grad():
        # 使用模型的diffusion对象进行采样
        samples = model.diffusion.sample(batch_size=num_samples)
    
    generation_time = time.time() - start_time
    print(f"⏱️  生成用时: {generation_time:.2f}秒")
    print(f"🚀 生成速度: {num_samples/generation_time:.1f} 样本/秒")
    
    # 保存样本
    if save_path:
        save_true_high_quality_samples(samples, save_path, model.image_size, dataset_name)
        print(f"💾 真正的高分辨率样本已保存到: {save_path}")
    
    return samples

def save_true_high_quality_samples(samples, path, image_size, dataset_name='unknown'):
    """保存真正的高质量样本"""
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
    
    # 根据分辨率和数据集调整显示大小
    if image_size >= 96:
        fig_size = 16
    elif image_size >= 64:
        fig_size = 14
    else:
        fig_size = 12
    
    plt.figure(figsize=(fig_size, fig_size))
    plt.imshow(grid_np)
    plt.axis('off')
    
    # 生成标题
    title = f'Generated Samples - {dataset_name.upper()} ({image_size}x{image_size})'
    if dataset_name == 'stl10':
        title += ' [原生96x96]'
    elif dataset_name == 'celeba':
        title += ' [原生64x64]'
    elif dataset_name == 'cifar10':
        title += ' [原生32x32]'
    else:
        title += ' [真正的高分辨率]'
    
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    
    # 高质量保存
    dpi = 300 if image_size >= 64 else 200
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

def interpolate_true_high_res(model, num_steps=10, save_path=None, dataset_name='unknown'):
    """真正的高分辨率插值"""
    print(f"🔄 生成真正的高分辨率插值序列 ({num_steps} 步)")
    
    device = model.device
    
    with torch.no_grad():
        # 生成两个随机样本
        sample1 = model.sample(batch_size=1)
        sample2 = model.sample(batch_size=1)
        
        # 使用模型的插值方法
        interpolated = model.interpolate(sample1, sample2, num_steps=num_steps)
    
    if save_path:
        save_true_interpolation_grid(interpolated, save_path, model.image_size, dataset_name)
        print(f"💾 真正的高分辨率插值序列已保存到: {save_path}")
    
    return interpolated

def save_true_interpolation_grid(samples, path, image_size, dataset_name='unknown'):
    """保存真正的高分辨率插值网格"""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    
    samples = (samples + 1) / 2
    samples = torch.clamp(samples, 0, 1)
    
    # 单行网格显示插值过程
    grid = torchvision.utils.make_grid(samples, nrow=len(samples), padding=2)
    grid_np = grid.permute(1, 2, 0).cpu().numpy()
    
    plt.figure(figsize=(len(samples) * 3, 3))
    plt.imshow(grid_np)
    plt.axis('off')
    
    title = f'Interpolation - {dataset_name.upper()} ({image_size}x{image_size})'
    plt.title(title, fontsize=14, fontweight='bold')
    
    plt.savefig(path, bbox_inches='tight', dpi=200, facecolor='white')
    plt.close()

def find_true_high_res_models():
    """查找train_true_high_res.py训练的模型"""
    models = []
    
    # 扫描checkpoints目录
    if os.path.exists('checkpoints'):
        for exp_dir in os.listdir('checkpoints'):
            exp_path = os.path.join('checkpoints', exp_dir)
            if os.path.isdir(exp_path) and ('native' in exp_dir or 'upsampled' in exp_dir):
                # 查找模型文件
                for file in os.listdir(exp_path):
                    if file.endswith('.pt'):
                        model_path = os.path.join(exp_path, file)
                        models.append({
                            'path': model_path,
                            'experiment': exp_dir,
                            'filename': file
                        })
    
    return models

def compare_true_high_res_models(model_paths, num_samples=4):
    """比较不同的真正高分辨率模型"""
    print("📊 比较不同的真正高分辨率模型")
    
    results = {}
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            try:
                model, image_size, dataset = load_true_high_res_model(model_path)
                samples = generate_true_high_res_samples(
                    model, 
                    num_samples=num_samples, 
                    show_progress=False,
                    dataset_name=dataset
                )
                results[f"{dataset.upper()}_{image_size}x{image_size}"] = {
                    'samples': samples,
                    'dataset': dataset,
                    'size': image_size
                }
                print(f"✅ {dataset.upper()} {image_size}x{image_size} 模型测试完成")
            except Exception as e:
                print(f"❌ {model_path} 加载失败: {e}")
        else:
            print(f"⚠️ 模型文件不存在: {model_path}")
    
    if results:
        # 保存比较图
        save_true_comparison_grid(results, "true_high_res_comparison.png")
        print("💾 真正的高分辨率比较图已保存")
    
    return results

def save_true_comparison_grid(results, path):
    """保存真正的高分辨率比较网格"""
    if not results:
        return
    
    n_models = len(results)
    n_samples = 4
    
    fig, axes = plt.subplots(n_models, n_samples, figsize=(16, 4 * n_models))
    if n_models == 1:
        axes = axes.reshape(1, -1)
    
    for row, (model_name, data) in enumerate(results.items()):
        samples = data['samples']
        dataset = data['dataset']
        size = data['size']
        
        samples = (samples + 1) / 2
        samples = torch.clamp(samples, 0, 1)
        
        for col in range(min(n_samples, len(samples))):
            sample_np = samples[col].permute(1, 2, 0).cpu().numpy()
            axes[row, col].imshow(sample_np)
            axes[row, col].axis('off')
            if col == 0:
                label = f"{dataset.upper()}\n{size}x{size}"
                if dataset == 'stl10':
                    label += "\n[原生96x96]"
                elif dataset == 'celeba':
                    label += "\n[原生64x64]"
                elif dataset == 'cifar10':
                    label += "\n[原生32x32]"
                    
                axes[row, col].set_ylabel(label, fontsize=12, fontweight='bold')
    
    plt.suptitle('真正的高分辨率模型比较', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight', dpi=200, facecolor='white')
    plt.close()

def main():
    """主函数 - 真正的高分辨率推理"""
    import argparse
    
    parser = argparse.ArgumentParser(description='真正的高分辨率DDPM推理')
    parser.add_argument('--model', type=str, required=True, help='模型路径')
    parser.add_argument('--samples', type=int, default=16, help='生成样本数量')
    parser.add_argument('--steps', type=int, default=None, help='采样步数')
    parser.add_argument('--output', type=str, default='true_high_res_samples.png', help='输出路径')
    parser.add_argument('--interpolate', action='store_true', help='生成插值序列')
    parser.add_argument('--compare', nargs='+', help='比较多个模型')
    
    args = parser.parse_args()
    
    if args.compare:
        # 比较模式
        compare_true_high_res_models(args.compare)
    else:
        # 单模型生成
        model, image_size, dataset = load_true_high_res_model(args.model)
        
        # 生成样本
        samples = generate_true_high_res_samples(
            model, 
            num_samples=args.samples,
            save_path=args.output,
            sampling_steps=args.steps,
            dataset_name=dataset
        )
        
        # 插值（如果请求）
        if args.interpolate:
            interpolate_path = f"interpolation_{dataset}_{image_size}x{image_size}.png"
            interpolate_true_high_res(model, save_path=interpolate_path, dataset_name=dataset)

if __name__ == "__main__":
    # 如果直接运行，使用交互模式
    print("🎨 真正的高分辨率DDPM推理工具")
    print("=" * 60)
    
    # 查找train_true_high_res.py训练的模型
    models = find_true_high_res_models()
    
    if not models:
        print("❌ 未找到由train_true_high_res.py训练的模型")
        print("💡 请先运行以下命令之一：")
        print("   python train_true_high_res.py stl10     # STL-10 (原生96x96)")
        print("   python train_true_high_res.py celeba    # CelebA (原生64x64)")
        print("   python train_true_high_res.py cifar10   # CIFAR-10 (原生32x32)")
        exit(1)
    
    print("📂 发现的真正高分辨率模型:")
    for i, model_info in enumerate(models):
        exp_name = model_info['experiment']
        filename = model_info['filename']
        print(f"   {i+1}. {exp_name}/{filename}")
    
    try:
        choice = int(input(f"\n选择模型 (输入编号 1-{len(models)}): ")) - 1
        if 0 <= choice < len(models):
            model_path = models[choice]['path']
            
            model, image_size, dataset = load_true_high_res_model(model_path)
            
            print("\n🎯 生成选项:")
            print("1. 生成样本")
            print("2. 生成插值序列") 
            print("3. 批量生成")
            print("4. 质量评估")
            
            option = input("选择操作 (1-4): ")
            
            if option == "1":
                num_samples = int(input("样本数量 (默认16): ") or "16")
                output_path = f"generated_{dataset}_{image_size}x{image_size}_samples.png"
                generate_true_high_res_samples(
                    model, 
                    num_samples=num_samples, 
                    save_path=output_path,
                    dataset_name=dataset
                )
                
            elif option == "2":
                interpolate_true_high_res(
                    model, 
                    save_path=f"interpolation_{dataset}_{image_size}x{image_size}.png",
                    dataset_name=dataset
                )
                
            elif option == "3":
                batch_size = int(input("每批样本数 (默认16): ") or "16")
                num_batches = int(input("批次数量 (默认5): ") or "5")
                
                batch_dir = f"batch_generation_{dataset}_{image_size}x{image_size}"
                os.makedirs(batch_dir, exist_ok=True)
                
                for i in range(num_batches):
                    output_path = f"{batch_dir}/batch_{i+1}.png"
                    generate_true_high_res_samples(
                        model, 
                        num_samples=batch_size, 
                        save_path=output_path,
                        dataset_name=dataset
                    )
                    print(f"✅ 批次 {i+1}/{num_batches} 完成")
                
                print("🎉 批量生成完成！")
                
            elif option == "4":
                print("🔍 质量评估模式")
                # 生成多个样本进行质量评估
                test_samples = [4, 9, 16, 25]
                
                for n in test_samples:
                    output_path = f"quality_test_{dataset}_{image_size}x{image_size}_{n}samples.png"
                    start_time = time.time()
                    generate_true_high_res_samples(
                        model, 
                        num_samples=n, 
                        save_path=output_path,
                        dataset_name=dataset
                    )
                    gen_time = time.time() - start_time
                    print(f"✅ {n}样本生成完成，用时{gen_time:.2f}秒")
                
                print("📊 质量评估完成，请查看生成的图片")
        else:
            print("❌ 无效选择")
            
    except (ValueError, KeyboardInterrupt):
        print("\n👋 退出程序") 