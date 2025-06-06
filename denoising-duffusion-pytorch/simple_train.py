"""
简单的DDPM训练示例 - 使用Trainer类
"""

from ddpm_model import DDPMModel
from config import TinyConfig
import os

def simple_train():
    """简单训练示例"""
    
    # 使用Tiny配置进行快速测试
    config = TinyConfig()
    
    print("🚀 开始简单训练示例")
    print(f"配置: {config.__class__.__name__}")
    
    # 创建模型
    model = DDPMModel(
        image_size=config.IMAGE_SIZE,
        channels=config.CHANNELS, 
        dim=config.DIM,
        dim_mults=config.DIM_MULTS,
        timesteps=config.TIMESTEPS,
        device=config.DEVICE
    )
    
    # 准备数据路径（这里使用一个示例路径）
    # 在实际使用中，这应该是包含图像文件的文件夹
    data_path = os.path.join(config.DATA_ROOT, 'cifar10_images')
    
    if not os.path.exists(data_path):
        print("⚠️  数据路径不存在，请先运行 python train.py 准备数据")
        print("或者将 data_path 设置为您的图像文件夹路径")
        return
    
    # 创建Trainer
    trainer = model.create_trainer(data_path, config)
    
    print("开始训练...")
    
    # 开始训练
    trainer.train()
    
    print("✅ 训练完成！")
    
    # 生成一些样本
    print("生成样本...")
    samples = model.sample(batch_size=4)
    print(f"生成样本形状: {samples.shape}")

if __name__ == "__main__":
    simple_train() 