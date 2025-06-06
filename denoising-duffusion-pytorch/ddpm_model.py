import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

class DDPMModel:
    """使用Trainer的DDPM模型封装类"""
    
    def __init__(self, 
                 image_size=32,
                 channels=3,
                 dim=64,
                 dim_mults=(1, 2, 4, 8),
                 timesteps=1000,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        
        self.device = device
        self.image_size = image_size
        self.channels = channels
        self.timesteps = timesteps
        
        # 初始化U-Net模型
        self.unet = Unet(
            dim=dim,
            channels=channels,
            dim_mults=dim_mults
        ).to(device)
        
        # 初始化高斯扩散过程
        self.diffusion = GaussianDiffusion(
            self.unet,
            image_size=image_size,
            timesteps=timesteps
        ).to(device)
        
        print(f"DDPM模型初始化完成:")
        print(f"- 图像尺寸: {image_size}x{image_size}")
        print(f"- 通道数: {channels}")
        print(f"- 时间步数: {timesteps}")
        print(f"- 设备: {device}")
        print(f"- 模型参数数量: {self.count_parameters():,}")
    
    def count_parameters(self):
        """计算模型参数数量"""
        return sum(p.numel() for p in self.unet.parameters() if p.requires_grad)
    
    def create_trainer(self, data_path, config):
        """创建Trainer实例"""
        trainer = Trainer(
            self.diffusion,
            data_path,
            train_batch_size=config.BATCH_SIZE,
            train_lr=config.LEARNING_RATE,
            train_num_steps=config.TRAIN_NUM_STEPS,
            gradient_accumulate_every=config.GRADIENT_ACCUMULATE_EVERY,
            ema_decay=config.EMA_DECAY,
            amp=config.AMP,
            results_folder=config.RESULTS_FOLDER,
            save_and_sample_every=config.SAVE_AND_SAMPLE_EVERY,
            num_samples=config.NUM_SAMPLES
        )
        return trainer
    
    def sample(self, batch_size=16):
        """生成样本"""
        with torch.no_grad():
            samples = self.diffusion.sample(batch_size=batch_size)
        return samples 