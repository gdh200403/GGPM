# DDPM - 经典扩散模型实现

基于 `denoising_diffusion_pytorch` 的经典DDPM (Denoising Diffusion Probabilistic Models) 实现，支持CIFAR-10数据集的训练和推理。

## 🚀 项目特性

- ✅ **经典DDPM架构**: 使用U-Net作为去噪网络
- ✅ **完整训练流程**: 支持从头开始训练和断点续训
- ✅ **多样化推理**: 样本生成、插值、渐进式生成
- ✅ **自动配置**: 根据GPU内存自动选择合适的模型配置
- ✅ **可视化支持**: 训练过程和生成结果的完整可视化
- ✅ **Demo测试**: 快速验证代码正确性的测试脚本

## 📦 安装环境

### 1. 克隆项目
```bash
git clone <项目地址>
cd DDPM
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 验证安装
```bash
python demo.py
```

## 🎯 快速开始

### Step 1: 运行Demo测试
```bash
python demo.py
```
这将执行一个快速的端到端测试，验证所有组件是否正常工作。

### Step 2: 完整训练
```bash
python train.py
```

### Step 3: 生成样本
```bash
python inference.py
```

## 📁 项目结构

```
├── ddpm_model.py      # DDPM模型核心实现
├── train.py           # 训练脚本
├── inference.py       # 推理脚本
├── demo.py           # Demo测试脚本
├── config.py         # 配置管理
├── requirements.txt  # 依赖列表
├── README.md        # 项目说明
├── checkpoints/     # 模型检查点
├── samples/         # 训练过程样本
├── inference_results/ # 推理结果
└── data/            # CIFAR-10数据集
```

## ⚙️ 配置说明

项目提供了多种预设配置，会根据您的GPU内存自动选择：

- **Tiny配置** (< 4GB GPU): 快速测试
- **Small配置** (4-8GB GPU): 个人GPU训练
- **Medium配置** (8-16GB GPU): 服务器GPU训练
- **Large配置** (> 16GB GPU): 高端GPU训练

您也可以在 `config.py` 中自定义配置参数。

## 🔧 使用方法

### 训练模型
```python
from train import train_ddpm

# 使用默认参数训练
model, losses = train_ddpm(epochs=50, batch_size=16)

# 自定义参数训练
model, losses = train_ddmp(
    epochs=100,
    batch_size=32,
    learning_rate=1e-4,
    save_interval=10
)
```

### 生成样本
```python
from ddpm_model import DDPMModel

# 加载训练好的模型
model = DDPMModel()
model.load_model('checkpoints/ddpm_final.pt')

# 生成样本
samples = model.sample(batch_size=16)
```

### 高级功能
```python
# 插值生成
interpolated = model.interpolate(image1, image2, num_steps=10)

# 渐进式生成过程
progressive_samples = model.sample(
    batch_size=1, 
    return_all_timesteps=True
)
```

## 📊 结果展示

训练过程中会自动生成：
- 训练损失曲线
- 每个epoch的生成样本
- 模型检查点

推理阶段会生成：
- 基础样本生成
- 插值序列
- 渐进式生成过程
- 与真实数据的对比

## 🎮 Demo测试详情

`demo.py` 包含以下测试：

1. **数据加载测试**: 验证CIFAR-10数据集加载
2. **模型初始化测试**: 验证模型架构正确性
3. **快速训练测试**: 5个epoch的训练验证
4. **推理测试**: 样本生成功能验证
5. **模型保存/加载测试**: 检查点功能验证
6. **性能基准测试**: 不同批次大小的性能对比

## 🔍 故障排除

### 常见问题

1. **GPU内存不足**
   ```
   CUDA out of memory
   ```
   解决方案: 减小批次大小或使用更小的模型配置

2. **依赖包版本冲突**
   ```
   ImportError: cannot import name...
   ```
   解决方案: 使用虚拟环境并严格按照requirements.txt安装

3. **CIFAR-10下载失败**
   ```
   Failed to download CIFAR-10
   ```
   解决方案: 检查网络连接或手动下载数据集

### 调试技巧

- 运行 `python demo.py` 进行快速诊断
- 检查 `samples/` 目录中的生成样本质量
- 观察训练损失曲线是否下降

## 📈 性能优化

### 训练优化
- 使用更大的批次大小（如果GPU内存允许）
- 启用混合精度训练
- 使用数据并行（多GPU）

### 推理优化
- 减少采样步数 (`sampling_timesteps`)
- 使用DDIM采样（更快的采样方法）
- 批量生成样本

## 🤝 贡献指南

欢迎提交问题和改进建议！

## 📄 许可证

MIT License

## 🙏 致谢

- [denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch)
- [DDPM论文](https://arxiv.org/abs/2006.11239)
- CIFAR-10数据集

---

如有问题，请提交Issue或查看Demo测试结果进行调试。