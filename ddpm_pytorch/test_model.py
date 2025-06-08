"""
测试模型是否工作正常
"""
import torch
from model import UNet
from ddpm import DDPM
from config import Config

def test_model():
    print("测试模型...")
    print(f"使用设备: {Config.device}")
    
    # 创建模型
    model = UNet(
        in_channels=Config.channels,
        dim=Config.dim,
        dim_mults=Config.dim_mults
    ).to(Config.device)
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建DDPM
    ddpm = DDPM(
        model=model,
        timesteps=Config.timesteps,
        device=Config.device
    )
    
    # 测试前向传播
    batch_size = 4
    x = torch.randn(batch_size, Config.channels, Config.image_size, Config.image_size).to(Config.device)
    t = torch.randint(0, Config.timesteps, (batch_size,)).to(Config.device)
    
    print("测试模型前向传播...")
    with torch.no_grad():
        output = model(x, t)
        print(f"输入形状: {x.shape}")
        print(f"输出形状: {output.shape}")
        print("✅ 模型前向传播测试通过！")
    
    # 测试训练损失
    print("测试训练损失计算...")
    loss = ddpm.training_loss(x)
    print(f"训练损失: {loss.item():.4f}")
    print("✅ 训练损失计算测试通过！")
    
    # 测试采样
    print("测试采样生成...")
    with torch.no_grad():
        samples = ddpm.sample((2, Config.channels, Config.image_size, Config.image_size))
        print(f"生成样本形状: {samples.shape}")
        print("✅ 采样生成测试通过！")
    
    print("🎉 所有测试通过！模型工作正常。")

if __name__ == '__main__':
    test_model() 