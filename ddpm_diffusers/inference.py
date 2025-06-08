"""
DDPM 模型推理脚本
使用训练好的模型生成图像
"""
import os
import argparse
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from diffusers import DDPMPipeline
from torchvision.utils import make_grid


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
    if torch.is_tensor(images):
        grid = make_grid_from_tensor(images, nrow=nrow)
        # 转换为 PIL 图像
        grid_np = grid.permute(1, 2, 0).cpu().numpy()
        grid_np = (grid_np * 255).astype(np.uint8)
    else:
        # 如果已经是numpy数组
        grid_np = images
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    img = Image.fromarray(grid_np)
    img.save(save_path)
    print(f"图像已保存到: {save_path}")


def generate_images(model_path, num_images=16, num_inference_steps=1000, 
                   guidance_scale=1.0, seed=None, device=None):
    """生成图像
    
    Args:
        model_path: 模型路径
        num_images: 生成图像数量
        num_inference_steps: 推理步数
        guidance_scale: 引导尺度
        seed: 随机种子
        device: 设备
    
    Returns:
        生成的图像张量
    """
    
    # 设置设备
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"使用设备: {device}")
    print(f"加载模型: {model_path}")
    
    # 加载模型
    try:
        pipeline = DDPMPipeline.from_pretrained(model_path, torch_dtype=torch.float16 if device == "cuda" else torch.float32)
        pipeline = pipeline.to(device)
    except Exception as e:
        print(f"加载模型失败: {e}")
        return None
    
    # 设置随机种子
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        print(f"设置随机种子: {seed}")
    
    print(f"开始生成 {num_images} 张图像...")
    print(f"推理步数: {num_inference_steps}")
    print(f"引导尺度: {guidance_scale}")
    
    # 生成图像
    with torch.no_grad():
        # 计算批次
        batch_size = min(8, num_images)  # 避免显存不够
        num_batches = (num_images + batch_size - 1) // batch_size
        
        all_images = []
        
        for i in range(num_batches):
            current_batch_size = min(batch_size, num_images - i * batch_size)
            
            print(f"生成批次 {i+1}/{num_batches} ({current_batch_size} 张图像)...")
            
            # 生成一个批次
            result = pipeline(
                batch_size=current_batch_size,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=torch.Generator(device=device).manual_seed(seed + i if seed else None)
            )
            
            batch_images = result.images
            
            # 转换为张量
            batch_tensors = []
            for img in batch_images:
                # PIL图像转换为张量
                img_array = np.array(img)
                img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0
                # 转换到 [-1, 1] 范围
                img_tensor = img_tensor * 2.0 - 1.0
                batch_tensors.append(img_tensor)
            
            batch_tensor = torch.stack(batch_tensors)
            all_images.append(batch_tensor)
        
        # 合并所有批次
        generated_images = torch.cat(all_images, dim=0)
    
    print(f"成功生成 {generated_images.shape[0]} 张图像")
    print(f"图像尺寸: {generated_images.shape}")
    
    return generated_images


def interpolate_latents(model_path, start_seed=0, end_seed=1000, num_steps=8, 
                       num_inference_steps=1000, device=None):
    """在潜在空间中插值生成图像序列
    
    Args:
        model_path: 模型路径
        start_seed: 起始随机种子
        end_seed: 结束随机种子
        num_steps: 插值步数
        num_inference_steps: 推理步数
        device: 设备
    
    Returns:
        插值图像序列
    """
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"加载模型进行潜在空间插值...")
    
    # 加载模型
    pipeline = DDPMPipeline.from_pretrained(model_path, torch_dtype=torch.float16 if device == "cuda" else torch.float32)
    pipeline = pipeline.to(device)
    
    # 生成起始和结束噪声
    sample_size = pipeline.unet.config.sample_size
    
    torch.manual_seed(start_seed)
    start_noise = torch.randn(1, 3, sample_size, sample_size, device=device)
    
    torch.manual_seed(end_seed)
    end_noise = torch.randn(1, 3, sample_size, sample_size, device=device)
    
    # 插值
    interpolated_images = []
    
    for i in range(num_steps):
        alpha = i / (num_steps - 1)
        interpolated_noise = (1 - alpha) * start_noise + alpha * end_noise
        
        print(f"生成插值图像 {i+1}/{num_steps} (alpha={alpha:.2f})...")
        
        # 手动去噪过程
        pipeline.scheduler.set_timesteps(num_inference_steps)
        noise = interpolated_noise.clone()
        
        with torch.no_grad():
            for t in tqdm(pipeline.scheduler.timesteps, desc=f"插值 {i+1}"):
                noise_pred = pipeline.unet(noise, timestep=t).sample
                noise = pipeline.scheduler.step(noise_pred, t, noise).prev_sample
        
        # 转换为标准化图像
        image = (noise + 1) / 2
        image = torch.clamp(image, 0, 1)
        
        interpolated_images.append(image.squeeze(0))
    
    return torch.stack(interpolated_images)


def create_gif(images, save_path, duration=0.5):
    """创建GIF动画
    
    Args:
        images: 图像张量列表
        save_path: 保存路径
        duration: 每帧持续时间（秒）
    """
    
    pil_images = []
    
    for img in images:
        # 转换为PIL图像
        if torch.is_tensor(img):
            img_np = img.permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8)
            pil_images.append(Image.fromarray(img_np))
        else:
            pil_images.append(img)
    
    # 保存GIF
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    pil_images[0].save(
        save_path,
        save_all=True,
        append_images=pil_images[1:],
        duration=int(duration * 1000),  # 毫秒
        loop=0
    )
    
    print(f"GIF动画已保存到: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="DDPM 模型推理")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--output_dir", type=str, default="./inference_results", help="输出目录")
    parser.add_argument("--num_images", type=int, default=16, help="生成图像数量")
    parser.add_argument("--num_inference_steps", type=int, default=1000, help="推理步数")
    parser.add_argument("--guidance_scale", type=float, default=1.0, help="引导尺度")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    parser.add_argument("--device", type=str, default=None, help="设备 (cuda/cpu)")
    parser.add_argument("--grid_size", type=int, default=4, help="网格大小")
    parser.add_argument("--interpolate", action="store_true", help="是否进行潜在空间插值")
    parser.add_argument("--interp_steps", type=int, default=8, help="插值步数")
    parser.add_argument("--start_seed", type=int, default=0, help="插值起始种子")
    parser.add_argument("--end_seed", type=int, default=1000, help="插值结束种子")
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.interpolate:
        print("执行潜在空间插值...")
        interpolated_images = interpolate_latents(
            model_path=args.model_path,
            start_seed=args.start_seed,
            end_seed=args.end_seed,
            num_steps=args.interp_steps,
            num_inference_steps=args.num_inference_steps,
            device=args.device
        )
        
        # 保存插值图像网格
        save_path = os.path.join(args.output_dir, "interpolation_grid.png")
        save_images(interpolated_images, save_path, nrow=args.interp_steps)
        
        # 创建GIF动画
        gif_path = os.path.join(args.output_dir, "interpolation.gif")
        create_gif(interpolated_images, gif_path, duration=0.5)
        
    else:
        print("执行图像生成...")
        generated_images = generate_images(
            model_path=args.model_path,
            num_images=args.num_images,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            device=args.device
        )
        
        if generated_images is not None:
            # 保存图像网格
            save_path = os.path.join(args.output_dir, "generated_images.png")
            save_images(generated_images, save_path, nrow=args.grid_size)
            
            # 保存单独的图像
            single_images_dir = os.path.join(args.output_dir, "single_images")
            os.makedirs(single_images_dir, exist_ok=True)
            
            for i, img in enumerate(generated_images):
                single_path = os.path.join(single_images_dir, f"image_{i:04d}.png")
                save_images(img.unsqueeze(0), single_path, nrow=1)
            
            print(f"推理完成！结果保存在: {args.output_dir}")


if __name__ == "__main__":
    main() 