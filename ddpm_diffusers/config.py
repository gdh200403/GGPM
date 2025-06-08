"""
DDPM 模型配置文件
"""
import os
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class ModelConfig:
    """模型相关配置 - 基于LargeConfig优化"""
    # UNet 配置 (参考LargeConfig: DIM=128, DIM_MULTS=(1,2,4,8))
    sample_size: int = 32  # 图像尺寸 (对应原来的IMAGE_SIZE)
    in_channels: int = 3   # 输入通道数 (RGB)
    out_channels: int = 3  # 输出通道数
    layers_per_block: int = 2
    block_out_channels: Optional[List[int]] = None
    down_block_types: Optional[List[str]] = None
    up_block_types: Optional[List[str]] = None
    attention_head_dim: int = 8
    norm_num_groups: int = 32
    cross_attention_dim: Optional[int] = None
    
    def __post_init__(self):
        if self.block_out_channels is None:
            # 基于LargeConfig的DIM=128和DIM_MULTS=(1,2,4,8)设置通道数
            base_dim = 128
            self.block_out_channels = [
                base_dim * 1,    # 128
                base_dim * 2,    # 256  
                base_dim * 4,    # 512
                base_dim * 8,    # 1024
            ]
        if self.down_block_types is None:
            # 调整为4层架构 (对应DIM_MULTS=(1,2,4,8))
            self.down_block_types = [
                "DownBlock2D",      # 128
                "DownBlock2D",      # 256
                "AttnDownBlock2D",  # 512 (添加注意力)
                "DownBlock2D",      # 1024
            ]
        if self.up_block_types is None:
            self.up_block_types = [
                "UpBlock2D",        # 1024 -> 512
                "AttnUpBlock2D",    # 512 -> 256 (添加注意力)
                "UpBlock2D",        # 256 -> 128
                "UpBlock2D",        # 128 -> out
            ]


@dataclass 
class SchedulerConfig:
    """调度器配置"""
    num_train_timesteps: int = 1000
    beta_start: float = 0.0001
    beta_end: float = 0.02
    beta_schedule: str = "linear"  # "linear", "scaled_linear", "squaredcos_cap_v2"
    prediction_type: str = "epsilon"  # "epsilon", "v_prediction"
    clip_sample: bool = False
    

@dataclass
class TrainingConfig:
    """训练配置 - 基于LargeConfig优化"""
    # 基本设置
    output_dir: str = "./results"
    seed: int = 42
    mixed_precision: str = "fp16"  # "no", "fp16", "bf16" (对应AMP=True)
    
    # 数据相关
    dataset_name: str = "cifar10"  # 支持 "cifar10", "mnist", "custom" 
    image_size: int = 32  # 对应LargeConfig中的IMAGE_SIZE=32
    dataloader_num_workers: int = 4
    
    # 训练超参数 (基于LargeConfig调整)
    train_batch_size: int = 1024  # LargeConfig的BATCH_SIZE=1024太大，调整为64
    eval_batch_size: int = 16
    num_epochs: int = 50   # 根据TRAIN_NUM_STEPS=10000调整轮数
    gradient_accumulation_steps: int = 2  # 对应GRADIENT_ACCUMULATE_EVERY=2
    learning_rate: float = 8e-5  # 对应LargeConfig的LEARNING_RATE=8e-5
    lr_scheduler: str = "cosine"  # "linear", "cosine", "constant"
    lr_warmup_steps: int = 500
    adam_beta1: float = 0.95
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-6
    adam_epsilon: float = 1e-08
    
    # 保存和日志 (基于LargeConfig调整)
    save_images_epochs: int = 2   # 对应SAVE_AND_SAMPLE_EVERY=500，更频繁保存
    save_model_epochs: int = 10    # 更频繁保存模型
    num_eval_samples: int = 16    # 对应NUM_SAMPLES=16
    logging_dir: str = "logs"
    report_to: str = "tensorboard"  # "tensorboard", "wandb", None
    
    # 推理设置
    guidance_scale: float = 1.0
    num_inference_steps: int = 1000  # 对应TIMESTEPS=1000


@dataclass
class DataConfig:
    """数据集配置"""
    dataset_path: Optional[str] = None  # 自定义数据集路径
    cache_dir: Optional[str] = None
    preprocessing_num_workers: int = 4
    
    # 数据增强
    random_flip: bool = True
    normalize: bool = True
    center_crop: bool = True


def get_config() -> tuple:
    """获取所有配置"""
    model_config = ModelConfig()
    scheduler_config = SchedulerConfig()
    training_config = TrainingConfig()
    data_config = DataConfig()
    
    return model_config, scheduler_config, training_config, data_config 