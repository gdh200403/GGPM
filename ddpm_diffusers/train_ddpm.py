"""
DDPM 模型训练脚本
使用 Hugging Face Diffusers 库
"""
import os
import math
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from diffusers import UNet2DModel, DDPMScheduler, DDPMPipeline
from diffusers.optimization import get_scheduler
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

from config import get_config
from dataset import create_dataloader, visualize_batch

logger = get_logger(__name__)


def make_grid_from_tensor(images, nrow=4, normalize=True):
    """从张量创建网格图像"""
    if normalize:
        # 从 [-1, 1] 标准化到 [0, 1]  
        images = (images + 1) / 2
        images = torch.clamp(images, 0, 1)
    
    grid = make_grid(images, nrow=nrow, padding=2, normalize=False)
    return grid


def save_images(images, save_path, nrow=4):
    """保存图像网格"""
    grid = make_grid_from_tensor(images, nrow=nrow)
    # 转换为 PIL 图像
    grid_np = grid.permute(1, 2, 0).cpu().numpy()
    grid_np = (grid_np * 255).astype(np.uint8)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    img = Image.fromarray(grid_np)
    img.save(save_path)
    print(f"图像已保存到: {save_path}")


def evaluate(model, scheduler, eval_batch_size, device, save_path=None):
    """评估模型并生成样本图像"""
    model.eval()
    
    # 创建随机噪声
    sample_size = model.config.sample_size
    noise = torch.randn(eval_batch_size, 3, sample_size, sample_size).to(device)
    
    # 设置调度器时间步
    scheduler.set_timesteps(num_inference_steps=1000)
    
    # 去噪过程
    with torch.no_grad():
        for t in tqdm(scheduler.timesteps, desc="生成样本"):
            # 模型预测
            noise_pred = model(noise, timestep=t).sample
            # 调度器步骤
            noise = scheduler.step(noise_pred, t, noise).prev_sample
    
    # 生成的图像
    images = noise
    
    if save_path:
        save_images(images, save_path)
    
    return images


def train_one_epoch(model, noise_scheduler, optimizer, train_dataloader, 
                   accelerator, epoch, global_step, tb_writer=None):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch}")
    
    for step, batch in enumerate(progress_bar):
        clean_images = batch[0] if isinstance(batch, (list, tuple)) else batch
        
        # 采样噪声
        noise = torch.randn_like(clean_images)
        batch_size = clean_images.shape[0]
        
        # 随机采样时间步
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, 
                                 (batch_size,), device=clean_images.device).long()
        
        # 根据噪声调度器添加噪声
        noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
        
        # 预测噪声
        with accelerator.accumulate(model):
            noise_pred = model(noisy_images, timesteps).sample
            
            # 计算损失
            loss = F.mse_loss(noise_pred, noise)
            
            # 反向传播
            accelerator.backward(loss)
            
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            optimizer.zero_grad()
        
        # 更新进度条
        if accelerator.sync_gradients:
            global_step += 1
            
        total_loss += loss.detach().item()
        
        # 更新进度条描述
        avg_loss = total_loss / (step + 1)
        progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
        
        # 记录到tensorboard
        if accelerator.is_main_process and tb_writer is not None:
            tb_writer.add_scalar('train/loss', loss.item(), global_step)
            
            # 每100步记录学习率
            if global_step % 100 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                tb_writer.add_scalar('train/learning_rate', current_lr, global_step)
    
    return global_step, total_loss / len(train_dataloader)


def main():
    """主训练函数"""
    # 加载配置
    model_config, scheduler_config, training_config, data_config = get_config()
    
    # 设置加速器
    accelerator_project_config = ProjectConfiguration(
        project_dir=training_config.output_dir,
        logging_dir=os.path.join(training_config.output_dir, training_config.logging_dir),
    )
    
    accelerator = Accelerator(
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        mixed_precision=training_config.mixed_precision,
        log_with=training_config.report_to,
        project_config=accelerator_project_config,
    )
    
    # 设置日志
    if accelerator.is_main_process:
        os.makedirs(training_config.output_dir, exist_ok=True)
        
    logger.info(accelerator.state, main_process_only=False)
    
    # 设置随机种子
    if training_config.seed is not None:
        torch.manual_seed(training_config.seed)
        np.random.seed(training_config.seed)
    
    # 创建数据加载器
    train_dataloader, num_channels = create_dataloader(
        dataset_name=training_config.dataset_name,
        image_size=training_config.image_size,
        batch_size=training_config.train_batch_size,
        num_workers=training_config.dataloader_num_workers,
        dataset_path=data_config.dataset_path,
        cache_dir=data_config.cache_dir,
        normalize=data_config.normalize,
        center_crop=data_config.center_crop,
        random_flip=data_config.random_flip
    )
    
    # 更新模型配置
    model_config.sample_size = training_config.image_size
    model_config.in_channels = num_channels
    model_config.out_channels = num_channels
    
    # 创建模型
    model = UNet2DModel(
        sample_size=model_config.sample_size,
        in_channels=model_config.in_channels,
        out_channels=model_config.out_channels,
        layers_per_block=model_config.layers_per_block,
        block_out_channels=model_config.block_out_channels,
        down_block_types=model_config.down_block_types,
        up_block_types=model_config.up_block_types,
        attention_head_dim=model_config.attention_head_dim,
        norm_num_groups=model_config.norm_num_groups,
    )
    
    # 创建噪声调度器
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=scheduler_config.num_train_timesteps,
        beta_start=scheduler_config.beta_start,
        beta_end=scheduler_config.beta_end,
        beta_schedule=scheduler_config.beta_schedule,
        prediction_type=scheduler_config.prediction_type,
        clip_sample=scheduler_config.clip_sample,
    )
    
    # 创建优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        betas=(training_config.adam_beta1, training_config.adam_beta2),
        weight_decay=training_config.adam_weight_decay,
        eps=training_config.adam_epsilon,
    )
    
    # 创建学习率调度器
    lr_scheduler = get_scheduler(
        training_config.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=training_config.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * training_config.num_epochs),
    )
    
    # 使用accelerator准备模型和优化器
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    
    # 初始化tensorboard
    tb_writer = None
    if accelerator.is_main_process and training_config.report_to == "tensorboard":
        tb_writer = SummaryWriter(log_dir=os.path.join(training_config.output_dir, training_config.logging_dir))
    
    # 打印模型信息
    if accelerator.is_main_process:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"模型参数总数: {total_params:,}")
        logger.info(f"可训练参数: {trainable_params:,}")
        logger.info(f"数据集: {training_config.dataset_name}")
        logger.info(f"批次大小: {training_config.train_batch_size}")
        logger.info(f"训练轮数: {training_config.num_epochs}")
        logger.info(f"学习率: {training_config.learning_rate}")
        
        # 可视化一个批次的训练数据
        sample_batch = next(iter(train_dataloader))
        sample_images = sample_batch[0] if isinstance(sample_batch, (list, tuple)) else sample_batch
        sample_path = os.path.join(training_config.output_dir, "sample_training_data.png")
        save_images(sample_images[:16], sample_path, nrow=4)
    
    # 开始训练
    logger.info("开始训练...")
    global_step = 0
    
    for epoch in range(training_config.num_epochs):
        # 训练一个epoch
        global_step, avg_loss = train_one_epoch(
            model, noise_scheduler, optimizer, train_dataloader,
            accelerator, epoch, global_step, tb_writer
        )
        
        # 更新学习率
        lr_scheduler.step()
        
        if accelerator.is_main_process:
            logger.info(f"Epoch {epoch+1}/{training_config.num_epochs}, 平均损失: {avg_loss:.4f}")
            
            # 记录epoch损失到tensorboard
            if tb_writer is not None:
                tb_writer.add_scalar('epoch/loss', avg_loss, epoch)
            
            # 评估并保存样本图像
            if (epoch + 1) % training_config.save_images_epochs == 0:
                logger.info("生成样本图像...")
                sample_images = evaluate(
                    accelerator.unwrap_model(model),
                    noise_scheduler,
                    training_config.num_eval_samples,
                    accelerator.device,
                    save_path=os.path.join(training_config.output_dir, f"samples_epoch_{epoch+1}.png")
                )
                
                # 记录图像到tensorboard
                if tb_writer is not None:
                    grid = make_grid_from_tensor(sample_images[:16], nrow=4)
                    tb_writer.add_image('generated_samples', grid, epoch)
            
            # 保存模型
            if (epoch + 1) % training_config.save_model_epochs == 0:
                logger.info("保存模型...")
                pipeline = DDPMPipeline(
                    unet=accelerator.unwrap_model(model),
                    scheduler=noise_scheduler,
                )
                
                save_dir = os.path.join(training_config.output_dir, f"checkpoint-epoch-{epoch+1}")
                pipeline.save_pretrained(save_dir)
                logger.info(f"模型已保存到: {save_dir}")
    
    # 训练完成，保存最终模型
    if accelerator.is_main_process:
        logger.info("训练完成！保存最终模型...")
        
        final_pipeline = DDPMPipeline(
            unet=accelerator.unwrap_model(model),
            scheduler=noise_scheduler,
        )
        
        final_save_dir = os.path.join(training_config.output_dir, "final_model")
        final_pipeline.save_pretrained(final_save_dir)
        logger.info(f"最终模型已保存到: {final_save_dir}")
        
        # 生成最终样本
        logger.info("生成最终样本...")
        final_samples = evaluate(
            accelerator.unwrap_model(model),
            noise_scheduler,
            training_config.num_eval_samples,
            accelerator.device,
            save_path=os.path.join(training_config.output_dir, "final_samples.png")
        )
        
        if tb_writer is not None:
            tb_writer.close()
    
    logger.info("训练完成！")


if __name__ == "__main__":
    main() 