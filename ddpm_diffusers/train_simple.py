"""
简化版 DDPM 训练脚本
用于快速测试和小规模实验
"""
import os
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from diffusers import UNet2DModel, DDPMScheduler, DDPMPipeline
from torchvision.utils import save_image

from config import get_config
from dataset import create_dataloader


def simple_train():
    """简化版训练函数"""
    print("=== DDPM 简化训练 ===")
    
    # 加载配置
    model_config, scheduler_config, training_config, data_config = get_config()
    
    # 优化配置以充分利用GPU
    training_config.num_epochs = 5
    training_config.train_batch_size = 1024  # 增大批次大小充分利用GPU
    training_config.save_images_epochs = 1
    training_config.save_model_epochs = 5
    training_config.num_eval_samples = 16
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建输出目录
    os.makedirs(training_config.output_dir, exist_ok=True)
    
    # 创建数据加载器
    print("创建数据加载器...")
    train_dataloader, num_channels = create_dataloader(
        dataset_name=training_config.dataset_name,
        image_size=training_config.image_size,
        batch_size=training_config.train_batch_size,
        num_workers=training_config.dataloader_num_workers,  # 使用配置的线程数
        normalize=data_config.normalize,
        center_crop=data_config.center_crop,
        random_flip=data_config.random_flip
    )
    
    # 更新模型配置
    model_config.sample_size = training_config.image_size
    model_config.in_channels = num_channels
    model_config.out_channels = num_channels
    
    # 创建模型（使用完整的LargeConfig架构）
    print("创建模型...")
    model = UNet2DModel(
        sample_size=model_config.sample_size,
        in_channels=model_config.in_channels,
        out_channels=model_config.out_channels,
        layers_per_block=model_config.layers_per_block,
        block_out_channels=tuple(model_config.block_out_channels),  # 转换为tuple
        down_block_types=tuple(model_config.down_block_types),      # 转换为tuple
        up_block_types=tuple(model_config.up_block_types),          # 转换为tuple
        attention_head_dim=model_config.attention_head_dim,
        norm_num_groups=model_config.norm_num_groups,
    ).to(device)
    
    # 创建噪声调度器
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=scheduler_config.num_train_timesteps,
        beta_start=scheduler_config.beta_start,
        beta_end=scheduler_config.beta_end,
        beta_schedule=scheduler_config.beta_schedule,
        prediction_type=scheduler_config.prediction_type,
    )
    
    # 创建优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        betas=(training_config.adam_beta1, training_config.adam_beta2),
        weight_decay=training_config.adam_weight_decay,
    )
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数总数: {total_params:,}")
    
    # 保存训练数据样本
    sample_batch = next(iter(train_dataloader))
    sample_images = sample_batch[0] if isinstance(sample_batch, (list, tuple)) else sample_batch
    sample_path = os.path.join(training_config.output_dir, "training_samples.png")
    save_image((sample_images[:4] + 1) / 2, sample_path, nrow=2, normalize=True)
    print(f"训练数据样本已保存: {sample_path}")
    
    # 开始训练
    print(f"开始训练 {training_config.num_epochs} 轮...")
    
    for epoch in range(training_config.num_epochs):
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")
        
        for step, batch in enumerate(progress_bar):
            clean_images = batch[0] if isinstance(batch, (list, tuple)) else batch
            clean_images = clean_images.to(device)
            
            # 生成噪声
            noise = torch.randn_like(clean_images)
            batch_size = clean_images.shape[0]
            
            # 随机采样时间步
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, 
                (batch_size,), device=device
            ).long()
            
            # 添加噪声
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
            
            # 前向传播
            noise_pred = model(noisy_images, timesteps).sample
            
            # 计算损失
            loss = F.mse_loss(noise_pred, noise)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{training_config.num_epochs}, 平均损失: {avg_loss:.4f}")
        
        # 生成样本图像
        if (epoch + 1) % training_config.save_images_epochs == 0:
            print("生成样本图像...")
            model.eval()
            
            with torch.no_grad():
                # 创建随机噪声
                sample_size = model.config.sample_size
                noise = torch.randn(training_config.num_eval_samples, 3, sample_size, sample_size).to(device)
                
                # 设置推理步数（简化版使用较少步数）
                noise_scheduler.set_timesteps(100)
                
                # 去噪过程
                for t in tqdm(noise_scheduler.timesteps, desc="生成中"):
                    noise_pred = model(noise, timestep=t).sample
                    noise = noise_scheduler.step(noise_pred, t, noise).prev_sample
                
                # 保存生成的图像
                generated_images = (noise + 1) / 2
                sample_path = os.path.join(training_config.output_dir, f"generated_epoch_{epoch+1}.png")
                save_image(generated_images, sample_path, nrow=2, normalize=True)
                print(f"生成样本已保存: {sample_path}")
    
    # 保存最终模型
    print("保存最终模型...")
    pipeline = DDPMPipeline(unet=model, scheduler=noise_scheduler)
    final_save_dir = os.path.join(training_config.output_dir, "simple_model")
    pipeline.save_pretrained(final_save_dir)
    print(f"模型已保存到: {final_save_dir}")
    
    # 生成最终样本
    print("生成最终样本...")
    model.eval()
    with torch.no_grad():
        noise = torch.randn(8, 3, model.config.sample_size, model.config.sample_size).to(device)
        noise_scheduler.set_timesteps(200)
        
        for t in tqdm(noise_scheduler.timesteps, desc="最终生成"):
            noise_pred = model(noise, timestep=t).sample
            noise = noise_scheduler.step(noise_pred, t, noise).prev_sample
        
        final_images = (noise + 1) / 2
        final_path = os.path.join(training_config.output_dir, "final_samples.png")
        save_image(final_images, final_path, nrow=4, normalize=True)
        print(f"最终样本已保存: {final_path}")
    
    print("简化训练完成！")
    print(f"结果保存在: {training_config.output_dir}")


if __name__ == "__main__":
    simple_train() 