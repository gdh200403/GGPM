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
    SAMPLING_TIMESTEPS = 250
    LOSS_TYPE = 'l1'  # 'l1' 或 'l2'
    RESNET_BLOCK_GROUPS = 8
    
    # 训练参数
    EPOCHS = 100
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 0.0
    ADAM_BETAS = (0.9, 0.99)
    GRADIENT_CLIP = 1.0
    
    # 数据参数
    DATA_ROOT = './data'
    NUM_WORKERS = 2
    PIN_MEMORY = True
    
    # 保存和日志参数
    SAVE_INTERVAL = 20  # 每多少个epoch保存一次
    CHECKPOINT_DIR = 'checkpoints'
    SAMPLE_DIR = 'samples'
    LOG_INTERVAL = 100  # 每多少个batch打印一次
    
    # 推理参数
    INFERENCE_BATCH_SIZE = 16
    NUM_SAMPLES_TO_GENERATE = 64
    
    # 设备配置
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Demo测试参数
    DEMO_EPOCHS = 5
    DEMO_BATCH_SIZE = 8
    DEMO_DIM = 32
    DEMO_DIM_MULTS = (1, 2, 4)
    DEMO_TIMESTEPS = 100

# 不同规模的配置预设
class TinyConfig(DDPMConfig):
    """小规模配置 - 用于快速测试"""
    DIM = 32
    DIM_MULTS = (1, 2, 4)
    TIMESTEPS = 100
    BATCH_SIZE = 16
    EPOCHS = 20

class SmallConfig(DDPMConfig):
    """小规模配置 - 适合个人GPU"""
    DIM = 64
    DIM_MULTS = (1, 2, 4)
    TIMESTEPS = 500
    BATCH_SIZE = 32
    EPOCHS = 50

class MediumConfig(DDPMConfig):
    """中等规模配置 - 适合服务器GPU"""
    DIM = 128
    DIM_MULTS = (1, 2, 4, 8)
    TIMESTEPS = 1000
    BATCH_SIZE = 64
    EPOCHS = 100

class LargeConfig(DDPMConfig):
    """大规模配置 - 适合高端GPU"""
    DIM = 256
    DIM_MULTS = (1, 2, 4, 8)
    TIMESTEPS = 1000
    BATCH_SIZE = 128
    EPOCHS = 200

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

# 数据增强配置
class AugmentationConfig:
    """数据增强配置"""
    HORIZONTAL_FLIP = True
    VERTICAL_FLIP = False
    ROTATION = 0  # 度数
    COLOR_JITTER = {
        'brightness': 0.1,
        'contrast': 0.1,
        'saturation': 0.1,
        'hue': 0.05
    }
    NORMALIZE_MEAN = (0.5, 0.5, 0.5)
    NORMALIZE_STD = (0.5, 0.5, 0.5)

# 调度器配置
class SchedulerConfig:
    """学习率调度器配置"""
    USE_SCHEDULER = True
    SCHEDULER_TYPE = 'cosine'  # 'step', 'cosine', 'exponential'
    
    # Step调度器参数
    STEP_SIZE = 30
    GAMMA = 0.1
    
    # Cosine调度器参数
    T_MAX = 100
    ETA_MIN = 1e-6
    
    # Exponential调度器参数
    EXPONENTIAL_GAMMA = 0.95

def print_config(config):
    """打印配置信息"""
    print("=" * 50)
    print("模型配置:")
    print(f"  图像尺寸: {config.IMAGE_SIZE}x{config.IMAGE_SIZE}")
    print(f"  通道数: {config.CHANNELS}")
    print(f"  模型维度: {config.DIM}")
    print(f"  维度倍数: {config.DIM_MULTS}")
    print(f"  时间步数: {config.TIMESTEPS}")
    print(f"  采样步数: {config.SAMPLING_TIMESTEPS}")
    
    print("\n训练配置:")
    print(f"  训练轮数: {config.EPOCHS}")
    print(f"  批次大小: {config.BATCH_SIZE}")
    print(f"  学习率: {config.LEARNING_RATE}")
    print(f"  设备: {config.DEVICE}")
    
    print("\n数据配置:")
    print(f"  数据根目录: {config.DATA_ROOT}")
    print(f"  工作进程数: {config.NUM_WORKERS}")
    
    print("\n保存配置:")
    print(f"  检查点目录: {config.CHECKPOINT_DIR}")
    print(f"  样本目录: {config.SAMPLE_DIR}")
    print(f"  保存间隔: {config.SAVE_INTERVAL} epochs")
    print("=" * 50)

if __name__ == "__main__":
    # 演示配置使用
    config = get_auto_config()
    print_config(config) 