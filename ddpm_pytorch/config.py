"""
DDPM 配置文件 - 简洁版
"""
import torch

class Config:
    # 数据相关
    image_size = 32
    channels = 3
    
    # 模型相关  
    dim = 64
    dim_mults = (1, 2, 4, 8)
    
    # DDPM相关
    timesteps = 1000
    beta_schedule = 'linear'
    beta_start = 0.0001
    beta_end = 0.02
    
    # 训练相关
    batch_size = 1024
    learning_rate = 8e-5
    epochs = 50
    
    # 设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 路径
    data_path = './data'
    save_path = './checkpoints'
    samples_path = './samples' 