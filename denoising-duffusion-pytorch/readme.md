# DDPM - 经典扩散模型完整实现

基于 `denoising_diffusion_pytorch` 的经典DDPM (Denoising Diffusion Probabilistic Models) 完整实现，支持多分辨率、多数据集的训练和推理，包含多种优化策略和完整的可视化功能。

## 🚀 项目特性

### 核心功能
- ✅ **经典DDPM架构**: 使用U-Net作为去噪网络
- ✅ **多分辨率支持**: CIFAR-10(32x32)、CelebA(64x64)、STL-10(96x96)
- ✅ **多种训练模式**: 基础、优化、高分辨率训练
- ✅ **完整推理功能**: 样本生成、插值、渐进式生成、批量生成
- ✅ **自动配置**: 根据GPU内存和数据集自动选择最优配置

### 训练优化
- ⚡ **混合精度训练**: 30-50% 速度提升，节省显存
- ⚡ **梯度累积**: 支持更大有效批次大小
- ⚡ **模型编译**: PyTorch 2.0+ 编译加速
- ⚡ **优化数据加载**: 多进程、预取、持久化worker
- ⚡ **学习率调度**: Cosine退火调度器
- ⚡ **智能检查点**: 最佳模型自动保存

### 可视化和分析
- 📊 **训练监控**: 实时损失曲线、样本生成
- 📊 **推理可视化**: 多种展示格式和对比分析
- 📊 **进度展示**: 渐进式去噪过程可视化
- 📊 **性能基准**: 不同配置的性能对比

## 📦 环境安装

### 1. 克隆项目
```bash
git clone <项目地址>
cd DDPM
```

### 2. 自动安装
```bash
python setup.py
```

### 3. 手动安装
```bash
pip install -r requirements.txt
```

### 4. 验证安装
```bash
python demo.py
```

## 🎯 快速开始

### Step 1: Demo测试（推荐）
```bash
python demo.py
```
运行完整的端到端测试，包括：
- 数据加载验证
- 模型正确性测试
- 快速训练验证
- 推理功能测试
- 数学性质验证
- 性能基准测试

### Step 2: 选择训练方式

#### 基础训练（入门推荐）
```bash
python train.py
```

#### 优化训练（推荐使用）
```bash
python train_optimized.py
```
- 1.7-2.2x 训练加速
- 混合精度 + 模型编译
- 智能批次大小调整

#### 高分辨率训练
```bash
# STL-10 (96x96原生分辨率)
python train_true_high_res.py --dataset stl10

# CelebA (64x64人脸数据)
python train_true_high_res.py --dataset celeba

# CIFAR-10上采样到64x64
python train_true_high_res.py --dataset cifar10 --target_size 64 --use_upsampling
```

### Step 3: 生成样本

#### 基础推理
```bash
python inference.py
```

#### 高分辨率推理
```bash
python inference_high_res.py
```

## 📁 项目结构

```
denoising-duffusion-pytorch/
├── 核心模块
│   ├── ddpm_model.py          # DDPM模型核心实现
│   ├── config.py              # 配置管理系统
│   └── requirements.txt       # 项目依赖
├── 训练脚本
│   ├── train.py              # 基础训练脚本
│   ├── train_optimized.py    # 优化训练脚本 (推荐)
│   ├── train_true_high_res.py # 高分辨率训练
│   └── train_stl10_optimized.py # STL-10专用优化训练
├── 推理脚本
│   ├── inference.py          # 基础推理脚本
│   └── inference_high_res.py # 高分辨率推理
├── 工具脚本
│   ├── demo.py              # 完整Demo测试
│   └── setup.py             # 自动安装脚本
├── 输出目录
│   ├── checkpoints/         # 模型检查点
│   ├── samples/            # 训练过程样本
│   ├── inference_results/  # 推理结果
│   ├── demo_results/       # Demo测试结果
│   └── data/              # 数据集存储
└── readme.md              # 项目文档
```

## ⚙️ 配置系统

项目提供智能配置系统，会根据GPU内存自动选择最优配置：

### 自动配置等级
- **Tiny配置** (< 4GB GPU): 快速测试，dim=32
- **Small配置** (4-8GB GPU): 个人GPU训练，dim=64  
- **Medium配置** (8-16GB GPU): 服务器GPU训练，dim=128
- **Large配置** (> 16GB GPU): 高端GPU训练，dim=256

### 数据集特定配置
- **CIFAR-10**: 32x32, dim=64, dim_mults=(1,2,4,8)
- **CelebA**: 64x64, dim=128, dim_mults=(1,2,4,8)
- **STL-10**: 96x96, dim=160, dim_mults=(1,2,4,8)

### 自定义配置
```python
from config import DDPMConfig

class MyConfig(DDPMConfig):
    IMAGE_SIZE = 64
    DIM = 128
    BATCH_SIZE = 32
    EPOCHS = 100
```

## 🔧 详细使用方法

### 训练模式对比

| 脚本 | 特点 | 适用场景 | 预期加速 |
|------|------|----------|----------|
| `train.py` | 基础训练，稳定可靠 | 初学者、调试 | 1.0x |
| `train_optimized.py` | 全面优化，推荐使用 | 日常训练 | 1.7-2.2x |
| `train_true_high_res.py` | 原生高分辨率 | 高质量图像 | 1.5-2.0x |

### GPU性能预估

#### RTX 3060 (12GB)
- **CIFAR-10**: 20-28小时 (50 epochs, 优化版)
- **STL-10**: 35-50小时 (50 epochs, 原生96x96)
- **推荐配置**: batch_size=16-24, mixed_precision=True

#### RTX 3090 (24GB)  
- **CIFAR-10**: 6-10小时 (50 epochs, 优化版)
- **STL-10**: 12-20小时 (50 epochs, 原生96x96)
- **推荐配置**: batch_size=32-48, mixed_precision=True

### 推理功能详解

#### 基础生成
```python
from ddpm_model import DDPMModel

model = DDPMModel()
model.load_model('checkpoints/ddpm_final.pt')
samples = model.sample(batch_size=16)
```

#### 插值生成
```python
# 在两个图像间插值
interpolated = model.interpolate(image1, image2, num_steps=10)
```

#### 渐进式生成
```python
# 展示完整去噪过程
progressive = model.sample(batch_size=1, return_all_timesteps=True)
```

#### 批量生成
```python
# 生成大量样本
from inference import batch_generate
all_samples = batch_generate(model, total_samples=1000, batch_size=32)
```

## 📊 训练效果监控

### 损失值解读
- **> 0.5**: 主要是噪声
- **0.3-0.5**: 开始出现轮廓
- **0.15-0.3**: 图像可识别
- **0.08-0.15**: 图像清晰
- **< 0.08**: 高质量图像

### 检查点策略
- 每10个epoch自动保存
- 最佳模型自动保存
- 支持断点续训
- 包含完整模型状态

## 🔍 Demo测试详情

`demo.py` 包含全面的功能验证：

### 基础测试
1. **数据加载测试**: CIFAR-10数据集下载和加载
2. **模型初始化测试**: 模型架构正确性验证
3. **训练流程测试**: 快速3-epoch训练验证

### 高级测试
4. **数学正确性验证**: 扩散过程数学性质检查
5. **梯度稳定性测试**: 训练稳定性验证
6. **采样一致性测试**: 生成质量一致性检查
7. **性能基准测试**: 不同批次大小性能对比

### 模型验证
8. **保存/加载测试**: 检查点功能完整性
9. **推理功能测试**: 生成、插值功能验证
10. **可视化测试**: 图像保存和显示功能

## 🎨 数据集支持

### CIFAR-10 (32x32)
- **优点**: 快速训练，适合实验
- **用途**: 算法验证、快速原型
- **训练时间**: 6-28小时 (取决于GPU)

### STL-10 (96x96)
- **优点**: 原生高分辨率，图像质量好
- **用途**: 高质量图像生成
- **训练时间**: 12-50小时

### CelebA (64x64)
- **优点**: 人脸数据，结构化强
- **用途**: 人脸生成、风格迁移
- **注意**: 需手动下载数据集

## 🚀 性能优化技巧

### 训练加速
```python
# 使用优化训练脚本
python train_optimized.py

# 关键优化参数
- use_amp=True          # 混合精度训练
- compile_model=True    # 模型编译
- num_workers=8         # 数据加载并行
- gradient_accumulation=4  # 梯度累积
```

### 显存优化
- 使用混合精度训练节省50%显存
- 梯度累积实现更大有效批次
- 自动批次大小调整避免OOM

### 推理优化
- 减少采样步数: `sampling_timesteps=100`
- 批量生成提高效率
- 使用编译模型加速推理

## 🔧 常见问题解决

### 安装问题
```bash
# CUDA版本不匹配
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 依赖冲突
pip install --upgrade denoising-diffusion-pytorch

# 虚拟环境推荐
conda create -n ddpm python=3.9
conda activate ddpm
```

### 训练问题
```bash
# GPU内存不足
RuntimeError: CUDA out of memory
# 解决：减小batch_size或使用混合精度

# 数据加载错误
# 解决：检查数据目录权限，重新下载数据集

# 模型加载失败
# 解决：检查模型路径，确认模型完整性
```

### 生成质量问题
- 训练不足：增加训练epoch数
- 学习率过高：降低学习率
- 模型容量不足：增加模型维度

## 📈 扩展开发

### 添加新数据集
```python
def get_custom_dataloader(dataset_path, transform):
    dataset = CustomDataset(dataset_path, transform=transform)
    return DataLoader(dataset, batch_size=32, shuffle=True)
```

### 自定义模型架构
```python
class CustomDDPM(DDPMModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 自定义修改
```

### 新的采样策略
```python
def custom_sample(self, batch_size, custom_schedule):
    # 实现自定义采样
    pass
```

## 📚 参考资料

- **DDPM论文**: [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- **U-Net论文**: [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- **原始实现**: [denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch)

## 🤝 贡献和支持

### 贡献指南
1. Fork项目
2. 创建功能分支
3. 提交更改
4. 发起Pull Request

### 问题报告
- 使用GitHub Issues报告bug
- 提供完整的错误信息和环境配置
- 包含复现步骤

### 功能请求
- 详细描述需求
- 说明使用场景
- 提供设计思路

## 📄 许可证

MIT License - 详见LICENSE文件

## 🙏 致谢

- [Lucidrains](https://github.com/lucidrains) - denoising-diffusion-pytorch作者
- DDPM论文作者团队
- PyTorch开发团队
- 数据集提供方：CIFAR、STL、CelebA

---

## 💡 快速上手建议

1. **新手**: 运行 `python demo.py` → `python train.py`
2. **进阶**: 使用 `python train_optimized.py` 获得最佳性能
3. **高分辨率**: 尝试 `python train_true_high_res.py --dataset stl10`
4. **问题调试**: 查看Demo测试结果，检查每个模块状态

**开始您的DDPM之旅吧！** 🎨✨