# PyTorch 原生 DDPM 实现

这是一个使用 PyTorch 原生实现的经典 DDPM (Denoising Diffusion Probabilistic Models) 框架，代码简洁易懂，包含最核心的功能。

## 项目结构

```
ddpm_pytorch/
├── config.py      # 配置文件
├── model.py       # UNet模型实现  
├── ddpm.py        # DDPM算法核心
├── train.py       # 训练脚本
├── inference.py   # 推理脚本
├── requirements.txt # 依赖文件
└── README.md      # 说明文档
```

## 核心特性

- ✅ **经典DDPM算法**：完整实现论文中的扩散模型
- ✅ **简洁UNet架构**：包含残差块、注意力机制、时间嵌入
- ✅ **前向扩散过程**：q(x_t | x_0) 噪声添加过程
- ✅ **反向去噪过程**：p(x_{t-1} | x_t) 采样生成过程  
- ✅ **训练和推理**：完整的训练循环和多种推理模式
- ✅ **可视化功能**：生成过程可视化、插值生成

## 安装依赖

```bash
cd ddpm_pytorch
pip install -r requirements.txt
```

## 快速开始

### 1. 训练模型

```bash
python train.py
```

训练过程会：
- 自动下载 CIFAR-10 数据集
- 每1000步生成样本图像
- 每10轮保存模型检查点
- 显示训练进度和损失

### 2. 推理生成

使用训练好的模型生成图像：

```bash
# 生成16个样本
python inference.py --checkpoint ./checkpoints/final_model.pth --num_samples 16

# 生成过程可视化
python inference.py --checkpoint ./checkpoints/final_model.pth --mode process

# 插值生成
python inference.py --checkpoint ./checkpoints/final_model.pth --mode interpolate

# 生成所有类型
python inference.py --checkpoint ./checkpoints/final_model.pth --mode all
```

## 配置说明

在 `config.py` 中可以调整所有参数：

```python
class Config:
    # 数据相关
    image_size = 32        # 图像尺寸
    channels = 3           # 通道数
    
    # 模型相关  
    dim = 64               # 基础维度
    dim_mults = (1, 2, 4, 8)  # 维度倍数
    
    # DDPM相关
    timesteps = 1000       # 时间步数
    beta_schedule = 'linear'  # 噪声调度
    beta_start = 0.0001    # 起始噪声
    beta_end = 0.02        # 结束噪声
    
    # 训练相关
    batch_size = 32        # 批次大小
    learning_rate = 2e-4   # 学习率
    epochs = 100           # 训练轮数
```

## 核心算法

### 1. 前向扩散过程

```python
def q_sample(self, x_start, t, noise=None):
    """q(x_t | x_0) - 从原始图像加噪声"""
    if noise is None:
        noise = torch.randn_like(x_start)
    
    sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t]
    sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
    
    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
```

### 2. 反向去噪过程

```python
def p_sample(self, x_t, t):
    """p(x_{t-1} | x_t) - 从噪声图像去噪"""
    model_mean, model_variance = self.p_mean_variance(x_t, t)
    
    noise = torch.randn_like(x_t)
    nonzero_mask = (t != 0).float()
    
    return model_mean + nonzero_mask * torch.sqrt(model_variance) * noise
```

### 3. 训练损失

```python
def training_loss(self, x_start):
    """计算训练损失 - 预测噪声的MSE"""
    t = torch.randint(0, self.timesteps, (batch_size,), device=device)
    noise = torch.randn_like(x_start)
    x_noisy = self.q_sample(x_start, t, noise)
    predicted_noise = self.model(x_noisy, t)
    loss = F.mse_loss(predicted_noise, noise)
    return loss
```

## UNet 架构

模型采用经典的 UNet 架构：

- **编码器**：下采样路径，提取多尺度特征
- **解码器**：上采样路径，重建图像分辨率
- **跳跃连接**：保留细节信息
- **残差块**：包含时间嵌入的残差连接
- **注意力机制**：在高层特征上应用自注意力
- **时间嵌入**：正弦余弦位置编码

## 推理模式

### 标准采样

从纯噪声逐步去噪生成图像：

```bash
python inference.py --checkpoint model.pth --mode sample --num_samples 16
```

### 生成过程可视化

展示从噪声到图像的完整去噪过程：

```bash
python inference.py --checkpoint model.pth --mode process
```

### 插值生成

在噪声空间中插值生成平滑过渡的图像序列：

```bash
python inference.py --checkpoint model.pth --mode interpolate
```

## 训练监控

训练过程中会输出：
- 实时损失值
- 训练进度条
- 每1000步的生成样本
- 每10轮的模型检查点

生成的文件：
```
checkpoints/          # 模型检查点
├── checkpoint_epoch_10.pth
├── checkpoint_epoch_20.pth
└── final_model.pth

samples/              # 训练过程中的样本
├── sample_step_1000.png
├── sample_step_2000.png
└── ...
```

## 性能优化建议

1. **增大批次大小**：如果显存充足，增大 `batch_size`
2. **调整模型尺寸**：增大 `dim` 或 `dim_mults` 提升模型容量
3. **使用混合精度**：添加 `torch.cuda.amp` 加速训练
4. **数据并行**：使用 `DataParallel` 或 `DistributedDataParallel`

## 算法细节

### 噪声调度

支持线性和余弦两种调度方式：

- **线性调度**：β 从 0.0001 线性增长到 0.02
- **余弦调度**：更平滑的噪声添加过程

### 时间步数

默认使用1000个时间步，可以调整：
- 更多时间步：生成质量更好，但推理更慢
- 更少时间步：推理更快，但质量可能下降

## 常见问题

**Q: 训练很慢怎么办？**
A: 可以减少时间步数、使用更小的模型、增大批次大小

**Q: 生成质量不好？**
A: 增加训练轮数、使用更大的模型、调整学习率

**Q: 显存不足？**
A: 减小批次大小、使用更小的图像尺寸、减少模型维度

**Q: 如何使用自己的数据集？**
A: 修改 `train.py` 中的数据加载部分，替换为自己的数据集

## 许可证

MIT License

## 参考

- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672) 