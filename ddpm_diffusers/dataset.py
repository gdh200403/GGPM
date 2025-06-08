"""
数据集加载器
支持 CIFAR-10、MNIST 和自定义数据集
"""
import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from PIL import Image
import numpy as np
from typing import Optional, Tuple, List


class CustomImageDataset(Dataset):
    """自定义图像数据集"""
    
    def __init__(self, data_dir: str, transform=None, extensions: List[str] = None):
        self.data_dir = data_dir
        self.transform = transform
        
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        # 获取所有图像文件
        self.image_paths = []
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if any(file.lower().endswith(ext) for ext in extensions):
                    self.image_paths.append(os.path.join(root, file))
        
        print(f"找到 {len(self.image_paths)} 张图像")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image


def get_transforms(image_size: int, normalize: bool = True, 
                  center_crop: bool = True, random_flip: bool = True) -> transforms.Compose:
    """获取数据预处理变换"""
    transform_list = []
    
    # 调整尺寸
    if center_crop:
        transform_list.append(transforms.Resize(image_size))
        transform_list.append(transforms.CenterCrop(image_size))
    else:
        transform_list.append(transforms.Resize((image_size, image_size)))
    
    # 随机水平翻转
    if random_flip:
        transform_list.append(transforms.RandomHorizontalFlip())
    
    # 转换为张量
    transform_list.append(transforms.ToTensor())
    
    # 标准化到 [-1, 1]
    if normalize:
        transform_list.append(transforms.Normalize([0.5], [0.5]))
    
    return transforms.Compose(transform_list)


def create_dataloader(dataset_name: str, 
                     image_size: int = 64,
                     batch_size: int = 16,
                     shuffle: bool = True,
                     num_workers: int = 4,
                     dataset_path: Optional[str] = None,
                     cache_dir: Optional[str] = None,
                     normalize: bool = True,
                     center_crop: bool = True,
                     random_flip: bool = True) -> Tuple[DataLoader, int]:
    """创建数据加载器
    
    Args:
        dataset_name: 数据集名称 ("cifar10", "mnist", "custom")
        image_size: 图像尺寸
        batch_size: 批次大小
        shuffle: 是否打乱数据
        num_workers: 数据加载线程数
        dataset_path: 自定义数据集路径
        cache_dir: 缓存目录
        normalize: 是否标准化到[-1,1]
        center_crop: 是否中心裁剪
        random_flip: 是否随机翻转
    
    Returns:
        (dataloader, num_channels)
    """
    
    # 获取数据变换
    transform = get_transforms(image_size, normalize, center_crop, random_flip)
    
    if dataset_name.lower() == "cifar10":
        dataset = datasets.CIFAR10(
            root=cache_dir or "./data",
            train=True,
            download=True,
            transform=transform
        )
        num_channels = 3
        
    elif dataset_name.lower() == "mnist":
        # MNIST 需要特殊处理，转换为3通道
        mnist_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # 转换为3通道
            transforms.Normalize([0.5], [0.5]) if normalize else transforms.Lambda(lambda x: x)
        ])
        
        dataset = datasets.MNIST(
            root=cache_dir or "./data",
            train=True,
            download=True,
            transform=mnist_transform
        )
        num_channels = 3
        
    elif dataset_name.lower() == "custom":
        if dataset_path is None:
            raise ValueError("使用自定义数据集时必须指定 dataset_path")
        
        if not os.path.exists(dataset_path):
            raise ValueError(f"数据集路径不存在: {dataset_path}")
        
        dataset = CustomImageDataset(dataset_path, transform=transform)
        num_channels = 3
        
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # 确保批次大小一致
    )
    
    print(f"数据集: {dataset_name}")
    print(f"样本数量: {len(dataset)}")
    print(f"批次大小: {batch_size}")
    print(f"批次数量: {len(dataloader)}")
    print(f"图像尺寸: {image_size}x{image_size}")
    print(f"通道数: {num_channels}")
    
    return dataloader, num_channels


def visualize_batch(dataloader: DataLoader, num_samples: int = 8, save_path: Optional[str] = None):
    """可视化一个批次的数据"""
    import matplotlib.pyplot as plt
    
    # 获取一个批次
    batch = next(iter(dataloader))
    if isinstance(batch, (list, tuple)):
        images = batch[0]
    else:
        images = batch
    
    # 限制显示数量
    num_samples = min(num_samples, images.shape[0])
    
    # 反标准化 (从 [-1,1] 到 [0,1])
    images = (images + 1) / 2
    images = torch.clamp(images, 0, 1)
    
    # 创建子图
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.flatten()
    
    for i in range(num_samples):
        img = images[i].permute(1, 2, 0).cpu().numpy()
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(f'Sample {i+1}')
    
    # 隐藏多余的子图
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"样本图像已保存到: {save_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    # 测试数据加载器
    print("测试 CIFAR-10 数据加载器...")
    dataloader, num_channels = create_dataloader(
        dataset_name="cifar10",
        image_size=64,
        batch_size=16
    )
    
    # 可视化样本
    visualize_batch(dataloader, num_samples=8, save_path="sample_images.png")
    
    print("数据加载器测试完成！") 