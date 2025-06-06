#!/usr/bin/env python3
"""
DDPM项目安装脚本
自动检查和安装所需依赖
"""

import subprocess
import sys
import os

def install_dependencies():
    """安装项目依赖"""
    print("🔧 安装项目依赖...")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✅ 依赖安装成功!")
        return True
    except subprocess.CalledProcessError:
        print("❌ 依赖安装失败!")
        return False

def create_directories():
    """创建必要的目录"""
    print("📁 创建项目目录...")
    
    directories = [
        'checkpoints',
        'samples', 
        'inference_results',
        'data'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ 创建目录: {directory}")

def check_cuda():
    """检查CUDA是否可用"""
    print("🔍 检查CUDA环境...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            print(f"✅ CUDA可用!")
            print(f"   GPU数量: {gpu_count}")
            print(f"   GPU名称: {gpu_name}")
            print(f"   GPU内存: {gpu_memory:.1f}GB")
        else:
            print("⚠️  CUDA不可用，将使用CPU训练")
    except ImportError:
        print("❌ PyTorch未安装，无法检查CUDA")

def main():
    """主安装流程"""
    print("🚀 DDPM项目安装器")
    print("=" * 40)
    
    # 1. 安装依赖
    if not install_dependencies():
        print("❌ 安装失败，请手动安装依赖")
        return
    
    # 2. 创建目录
    create_directories()
    
    # 3. 检查CUDA
    check_cuda()
    
    print("\n" + "=" * 40)
    print("🎉 安装完成!")
    print("\n接下来您可以:")
    print("1. 运行 python run_demo.py 启动交互式界面")
    print("2. 运行 python demo.py 进行快速测试")
    print("3. 运行 python train.py 开始训练")
    print("\n更多信息请查看 README.md")

if __name__ == "__main__":
    main() 