import torch
import os
from ddpm_model import DDPMModel
from config import get_auto_config, print_config

def prepare_cifar10_data(config):
    """准备CIFAR-10数据路径"""
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    
    # 创建数据目录
    data_dir = os.path.join(config.DATA_ROOT, 'cifar10_images')
    os.makedirs(data_dir, exist_ok=True)
    
    # 如果数据已存在，直接返回路径
    if len(os.listdir(data_dir)) > 1000:  # 假设已有足够图像
        print(f"使用现有CIFAR-10图像数据: {data_dir}")
        return data_dir
    
    print("正在准备CIFAR-10图像数据...")
    
    # 下载CIFAR-10数据集
    transform = transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.ToTensor(),
    ])
    
    dataset = torchvision.datasets.CIFAR10(
        root=config.DATA_ROOT,
        train=True,
        download=True,
        transform=transform
    )
    
    # 创建数据加载器来迭代数据
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # 将数据保存为图像文件
    for idx, (image_tensor, _) in enumerate(dataloader):
        if idx >= 10000:  # 限制数量以节省空间
            break
        
        # 转换为PIL图像
        image = transforms.ToPILImage()(image_tensor.squeeze(0))
        image_path = os.path.join(data_dir, f'image_{idx:05d}.png')
        image.save(image_path)
        
        if idx % 1000 == 0:
            print(f"已处理 {idx} 张图像...")
    
    print(f"CIFAR-10图像数据准备完成: {data_dir}")
    return data_dir

def train_with_trainer(config):
    """使用Trainer进行训练"""
    
    # 准备数据
    data_path = prepare_cifar10_data(config)
    
    # 创建模型
    model = DDPMModel(
        image_size=config.IMAGE_SIZE,
        channels=config.CHANNELS,
        dim=config.DIM,
        dim_mults=config.DIM_MULTS,
        timesteps=config.TIMESTEPS,
        device=config.DEVICE
    )
    
    # 创建Trainer
    trainer = model.create_trainer(data_path, config)
    
    print("开始训练...")
    print_config(config)
    
    # 开始训练
    trainer.train()
    
    print("训练完成！")
    return trainer

def test_setup():
    """测试训练设置"""
    print("🔍 验证训练设置...")
    
    try:
        config = get_auto_config()
        
        # 创建一个小模型进行测试
        model = DDPMModel(
            image_size=32,
            channels=3,
            dim=32,
            dim_mults=(1, 2),
            timesteps=100,
            device=config.DEVICE
        )
        print("✅ 模型创建正常")
        
        # 测试采样
        samples = model.sample(batch_size=4)
        print(f"✅ 采样测试正常 - 样本形状: {samples.shape}")
        
        print("🎉 训练设置验证通过！")
        return True
        
    except Exception as e:
        print(f"❌ 训练设置验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import sys
    
    # 获取配置
    config = get_auto_config()
    
    # 检查命令行参数
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        # 只运行验证测试
        if test_setup():
            print("✅ 训练脚本准备就绪，可以开始训练")
        else:
            print("❌ 请修复问题后再开始训练")
        sys.exit()
    
    # 运行验证测试
    print("开始训练前验证...")
    if not test_setup():
        print("❌ 验证失败，请检查环境设置")
        sys.exit(1)
    
    # 开始训练
    try:
        trainer = train_with_trainer(config)
        print("🎉 训练成功完成！")
    except Exception as e:
        print(f"❌ 训练过程中出错: {e}")
        import traceback
        traceback.print_exc()