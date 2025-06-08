import torch
from diffusers.pipelines.ddpm.pipeline_ddpm import DDPMPipeline
import matplotlib.pyplot as plt

def generate_cifar10_images(num_images=16, seed=None):
    """
    使用预训练的 DDPM-CIFAR10 模型生成图片，并拼接成一个大图。

    参数:
    num_images (int): 希望生成的图片数量（建议为平方数，如16）。
    seed (int, optional): 用于可复现结果的随机种子。
    """
    print("正在从 Hugging Face Hub 加载预训练模型...")
    pipeline = DDPMPipeline.from_pretrained("google/ddpm-cifar10-32")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline.to(device)
    print(f"当前使用的设备: {device}")

    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)

    print(f"开始推理，生成 {num_images} 张图片...")
    result = pipeline(batch_size=num_images, generator=generator)
    if isinstance(result, dict) and "images" in result:
        images = result["images"]
    elif isinstance(result, (tuple, list)):
        images = result[0]
    else:
        raise ValueError("无法从 pipeline 的输出中获取 images。")
    print("推理完成！")

    # 拼接成一个大图（4x4网格）
    grid_size = int(num_images ** 0.5)
    assert grid_size * grid_size == num_images, "num_images 必须为完全平方数"
    img_w, img_h = images[0].size
    from PIL import Image
    grid_img = Image.new('RGB', (img_w * grid_size, img_h * grid_size))
    for idx, img in enumerate(images):
        row = idx // grid_size
        col = idx % grid_size
        grid_img.paste(img, (col * img_w, row * img_h))
    grid_img.save("ddpm_cifar10_grid.png")
    print("拼接大图已保存到: ddpm_cifar10_grid.png")

    # 可视化大图
    plt.figure(figsize=(8, 8))
    plt.imshow(grid_img)
    plt.axis("off")
    plt.title(f"DDPM 生成的 {num_images} 张 CIFAR-10 拼接图")
    plt.show()

if __name__ == "__main__":
    NUM_IMAGES_TO_GENERATE = 16  # 生成16张图片
    RANDOM_SEED = 42
    generate_cifar10_images(num_images=NUM_IMAGES_TO_GENERATE, seed=RANDOM_SEED)