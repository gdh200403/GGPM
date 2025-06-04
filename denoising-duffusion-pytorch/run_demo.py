#!/usr/bin/env python3
"""
DDPM模型运行脚本
提供交互式菜单，引导用户完成训练和推理流程
"""

import os
import sys
import subprocess

def print_banner():
    """打印项目横幅"""
    banner = """
    ██████╗ ██████╗ ██████╗ ███╗   ███╗
    ██╔══██╗██╔══██╗██╔══██╗████╗ ████║
    ██║  ██║██║  ██║██████╔╝██╔████╔██║
    ██║  ██║██║  ██║██╔═══╝ ██║╚██╔╝██║
    ██████╔╝██████╔╝██║     ██║ ╚═╝ ██║
    ╚═════╝ ╚═════╝ ╚═╝     ╚═╝     ╚═╝
    
    经典DDPM扩散模型 - 基于CIFAR-10数据集
    """
    print(banner)

def check_dependencies():
    """检查依赖是否已安装"""
    try:
        import torch
        import torchvision
        from denoising_diffusion_pytorch import Unet, GaussianDiffusion
        import matplotlib.pyplot as plt
        import numpy as np
        from tqdm import tqdm
        print("✅ 所有依赖已安装")
        return True
    except ImportError as e:
        print(f"❌ 缺少依赖: {e}")
        print("请运行: pip install -r requirements.txt")
        return False

def run_demo():
    """运行Demo测试"""
    print("\n🚀 开始运行Demo测试...")
    try:
        result = subprocess.run([sys.executable, "demo.py"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Demo测试成功完成!")
            print(result.stdout)
            return True
        else:
            print("❌ Demo测试失败:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ 运行Demo时出错: {e}")
        return False

def run_training():
    """运行训练"""
    print("\n🏋️ 开始训练模型...")
    
    # 获取用户配置选择
    print("请选择训练配置:")
    print("1. Tiny (快速测试, <4GB GPU)")
    print("2. Small (个人GPU, 4-8GB)")
    print("3. Medium (服务器GPU, 8-16GB)")
    print("4. Large (高端GPU, >16GB)")
    print("5. 自动选择")
    
    choice = input("请输入选择 (1-5): ").strip()
    
    config_map = {
        '1': 'TinyConfig',
        '2': 'SmallConfig', 
        '3': 'MediumConfig',
        '4': 'LargeConfig',
        '5': 'auto'
    }
    
    if choice in config_map:
        print(f"使用配置: {config_map[choice]}")
        
        try:
            result = subprocess.run([sys.executable, "train.py"], 
                                  capture_output=False, text=True)
            if result.returncode == 0:
                print("✅ 训练完成!")
                return True
            else:
                print("❌ 训练过程中出现错误")
                return False
        except KeyboardInterrupt:
            print("\n⏹️ 训练被用户中断")
            return False
        except Exception as e:
            print(f"❌ 训练时出错: {e}")
            return False
    else:
        print("无效选择")
        return False

def run_inference():
    """运行推理"""
    print("\n🎨 开始生成样本...")
    
    # 检查是否有训练好的模型
    checkpoint_paths = [
        'checkpoints/ddpm_final.pt',
        'checkpoints/ddmp_epoch_50.pt',
        'demo_model.pt'
    ]
    
    model_exists = any(os.path.exists(path) for path in checkpoint_paths)
    
    if not model_exists:
        print("❌ 未找到训练好的模型文件")
        print("请先运行训练或Demo测试")
        return False
    
    try:
        result = subprocess.run([sys.executable, "inference.py"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ 推理完成!")
            print("生成的样本保存在 inference_results/ 目录")
            return True
        else:
            print("❌ 推理失败:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ 推理时出错: {e}")
        return False

def show_results():
    """展示结果"""
    print("\n📊 查看结果文件:")
    
    # 检查各种结果文件
    result_files = [
        ("Demo数据样本", "demo_data_samples.png"),
        ("Demo训练损失", "demo_training_loss.png"), 
        ("Demo生成样本", "demo_generated_samples.png"),
        ("训练样本", "samples/"),
        ("推理结果", "inference_results/"),
        ("性能基准", "demo_benchmark.png")
    ]
    
    for name, path in result_files:
        if os.path.exists(path):
            print(f"✅ {name}: {path}")
        else:
            print(f"❌ {name}: 未找到")

def main_menu():
    """主菜单"""
    while True:
        print("\n" + "="*50)
        print("DDPM扩散模型 - 主菜单")
        print("="*50)
        print("1. 🔧 检查环境依赖")
        print("2. 🚀 运行Demo测试 (推荐首次运行)")
        print("3. 🏋️ 开始训练模型")
        print("4. 🎨 运行推理生成")
        print("5. 📊 查看结果文件")
        print("6. 📖 查看帮助信息")
        print("7. 🚪 退出程序")
        print("="*50)
        
        choice = input("请输入选择 (1-7): ").strip()
        
        if choice == '1':
            check_dependencies()
            
        elif choice == '2':
            if check_dependencies():
                run_demo()
            
        elif choice == '3':
            if check_dependencies():
                run_training()
            
        elif choice == '4':
            if check_dependencies():
                run_inference()
            
        elif choice == '5':
            show_results()
            
        elif choice == '6':
            show_help()
            
        elif choice == '7':
            print("👋 再见!")
            break
            
        else:
            print("❌ 无效选择，请输入1-7")

def show_help():
    """显示帮助信息"""
    help_text = """
    📖 DDPM模型使用帮助
    
    🚀 快速开始流程:
    1. 首次使用请先运行"检查环境依赖"
    2. 运行"Demo测试"验证所有组件正常工作
    3. 根据GPU内存选择合适配置进行训练
    4. 训练完成后运行推理生成样本
    
    📁 重要文件说明:
    - requirements.txt: 依赖包列表
    - ddpm_model.py: 模型核心代码
    - train.py: 训练脚本
    - inference.py: 推理脚本
    - demo.py: 快速测试脚本
    - config.py: 配置管理
    
    🔧 故障排除:
    - 如果GPU内存不足，选择更小的配置
    - 如果Demo测试失败，检查依赖安装
    - 训练中断可以从检查点继续
    
    📊 结果查看:
    - samples/: 训练过程中的样本
    - inference_results/: 推理生成的结果
    - checkpoints/: 模型检查点文件
    
    更多详细信息请查看 README.md 文件
    """
    print(help_text)

if __name__ == "__main__":
    print_banner()
    main_menu() 