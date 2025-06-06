"""
DDPM模型配置文件
包含模型架构、训练参数和数据处理的所有配置
"""

import torch

class DDPMConfig:
    """DDPM模型配置类"""
    
    # 模型架构参数
    IMAGE_SIZE = 32
    CHANNELS = 3
    DIM = 64
    DIM_MULTS = (1, 2, 4, 8)
    TIMESTEPS = 1000
    
    # Trainer训练参数
    TRAIN_NUM_STEPS = 100000  # 总训练步数
    BATCH_SIZE = 32
    LEARNING_RATE = 2e-5
    GRADIENT_ACCUMULATE_EVERY = 2  # 梯度累积步数
    EMA_DECAY = 0.995  # 指数移动平均衰减
    AMP = True  # 混合精度训练
    
    # 数据参数
    DATA_ROOT = './data'
    
    # Trainer保存和采样参数
    RESULTS_FOLDER = 'results'  # Trainer结果文件夹
    SAVE_AND_SAMPLE_EVERY = 1000  # 每多少步保存和采样
    NUM_SAMPLES = 16  # 采样数量
    
    # 设备配置
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 不同规模的配置预设
class TinyConfig(DDPMConfig):
    """小规模配置 - 用于快速测试"""
    DIM = 32
    DIM_MULTS = (1, 2, 4)
    TIMESTEPS = 100
    BATCH_SIZE = 16
    TRAIN_NUM_STEPS = 10000
    SAVE_AND_SAMPLE_EVERY = 500

class SmallConfig(DDPMConfig):
    """小规模配置 - 适合个人GPU"""
    DIM = 64
    DIM_MULTS = (1, 2, 4)
    TIMESTEPS = 500
    BATCH_SIZE = 32
    TRAIN_NUM_STEPS = 50000
    SAVE_AND_SAMPLE_EVERY = 1000

class MediumConfig(DDPMConfig):
    """中等规模配置 - 适合服务器GPU"""
    DIM = 128
    DIM_MULTS = (1, 2, 4, 8)
    TIMESTEPS = 1000
    BATCH_SIZE = 64
    TRAIN_NUM_STEPS = 200000
    SAVE_AND_SAMPLE_EVERY = 2000

class LargeConfig(DDPMConfig):
    """大规模配置 - 适合高端GPU"""
    DIM = 256
    DIM_MULTS = (1, 2, 4, 8)
    TIMESTEPS = 1000
    BATCH_SIZE = 128
    TRAIN_NUM_STEPS = 700000
    SAVE_AND_SAMPLE_EVERY = 5000

# 根据GPU内存自动选择配置
def get_auto_config():
    """根据可用GPU内存自动选择配置"""
    if not torch.cuda.is_available():
        print("未检测到CUDA，使用CPU配置")
        return TinyConfig()
    
    # 获取GPU内存信息
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
    
    if gpu_memory < 4:
        print(f"GPU内存: {gpu_memory:.1f}GB - 使用Tiny配置")
        return TinyConfig()
    elif gpu_memory < 8:
        print(f"GPU内存: {gpu_memory:.1f}GB - 使用Small配置")
        return SmallConfig()
    elif gpu_memory < 16:
        print(f"GPU内存: {gpu_memory:.1f}GB - 使用Medium配置")
        return MediumConfig()
    else:
        print(f"GPU内存: {gpu_memory:.1f}GB - 使用Large配置")
        return LargeConfig()

def print_config(config):
    """打印配置信息"""
    print("=" * 50)
    print("模型配置:")
    print(f"  图像尺寸: {config.IMAGE_SIZE}x{config.IMAGE_SIZE}")
    print(f"  通道数: {config.CHANNELS}")
    print(f"  模型维度: {config.DIM}")
    print(f"  维度倍数: {config.DIM_MULTS}")
    print(f"  时间步数: {config.TIMESTEPS}")
    
    print("\nTrainer训练配置:")
    print(f"  总训练步数: {config.TRAIN_NUM_STEPS}")
    print(f"  批次大小: {config.BATCH_SIZE}")
    print(f"  学习率: {config.LEARNING_RATE}")
    print(f"  梯度累积步数: {config.GRADIENT_ACCUMULATE_EVERY}")
    print(f"  EMA衰减: {config.EMA_DECAY}")
    print(f"  混合精度: {config.AMP}")
    print(f"  设备: {config.DEVICE}")
    
    print("\n数据配置:")
    print(f"  数据根目录: {config.DATA_ROOT}")
    
    print("\n保存配置:")
    print(f"  结果文件夹: {config.RESULTS_FOLDER}")
    print(f"  保存采样间隔: {config.SAVE_AND_SAMPLE_EVERY} steps")
    print(f"  采样数量: {config.NUM_SAMPLES}")
    print("=" * 50)

if __name__ == "__main__":
    # 演示配置使用
    config = get_auto_config()
    print_config(config) 