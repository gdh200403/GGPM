# 高分辨率DDPM训练指南

本指南介绍如何使用高分辨率版本的DDPM模型进行训练和推理。

## 🚀 快速开始

### 1. 训练高分辨率模型

#### 基础用法 - 64x64分辨率
```bash
python train_high_res.py --dataset=cifar10 --size=64
```

#### 高分辨率训练 - 128x128
```bash
python train_high_res.py --dataset=cifar10 --size=128
```

#### 超高分辨率 - 256x256 (需要高端GPU)
```bash
python train_high_res.py --dataset=cifar10 --size=256
```

### 2. 支持的数据集

- **CIFAR-10**: `--dataset=cifar10` (32x32原始，可缩放到任意分辨率)
- **CIFAR-100**: `--dataset=cifar100` (100个类别，更具挑战性)
- **STL-10**: `--dataset=imagenet` (96x96原始，作为ImageNet替代)
- **CelebA**: `--dataset=celeba` (需要手动下载)

### 3. 命令行选项

```bash
python train_high_res.py \
  --dataset=cifar10 \          # 数据集选择
  --size=64 \                  # 目标分辨率
  --no-amp \                   # 关闭混合精度(如果有问题)
  --no-compile                 # 关闭模型编译(兼容性)
```

## 📊 分辨率与性能对照表

| 分辨率 | 推荐GPU | 批次大小 | 训练时间(50epochs) | 模型大小 |
|--------|---------|----------|-------------------|----------|
| 32x32  | GTX 1060+ | 32-64    | 8-15小时          | ~35MB    |
| 64x64  | RTX 3060+ | 16-32    | 15-25小时         | ~145MB   |
| 128x128| RTX 3080+ | 8-16     | 30-50小时         | ~380MB   |
| 256x256| RTX 3090+ | 4-8      | 60-100小时        | ~890MB   |

## 🎨 推理和生成

### 基础推理
```bash
python inference_high_res.py --model=checkpoints/cifar10_64x64/ddpm_best.pt --samples=16
```

### 高质量采样
```bash
python inference_high_res.py \
  --model=checkpoints/cifar10_128x128/ddpm_best.pt \
  --samples=9 \
  --steps=500 \
  --output=high_quality_samples.png
```

### 生成插值序列
```bash
python inference_high_res.py \
  --model=checkpoints/cifar10_64x64/ddpm_best.pt \
  --interpolate
```

### 比较不同分辨率
```bash
python inference_high_res.py \
  --compare \
  checkpoints/cifar10_32x32/ddpm_best.pt \
  checkpoints/cifar10_64x64/ddpm_best.pt \
  checkpoints/cifar10_128x128/ddpm_best.pt
```

## ⚙️ 自动优化特性

### 智能批次大小调整
系统会根据您的GPU自动调整批次大小：
- **RTX 3060 (12GB)**: 64x64→批次16, 128x128→批次8
- **RTX 3090 (24GB)**: 64x64→批次32, 128x128→批次16
- **其他GPU**: 保守估计，避免内存溢出

### 梯度累积
当批次大小较小时，自动使用梯度累积保持训练稳定性：
```
有效批次大小 = 实际批次大小 × 梯度累积步数
目标: 保持有效批次大小在16-32之间
```

### 混合精度训练
自动启用FP16混合精度训练，提升速度30-50%：
- 自动处理梯度缩放
- 兼容所有现代GPU
- 可通过`--no-amp`关闭

## 📁 文件组织

训练后的文件会按实验名称组织：
```
checkpoints/
├── cifar10_64x64/
│   ├── ddpm_best.pt         # 最佳模型
│   ├── ddpm_final.pt        # 最终模型
│   └── ddpm_epoch_10.pt     # 定期检查点
├── samples/
│   └── cifar10_64x64/
│       ├── epoch_10_samples.png
│       └── loss_curve_epoch_10.png
└── training_progress/
    └── cifar10_64x64/
        └── quick_epoch_5.png    # 快速采样监控
```

## 🔧 故障排除

### 1. 内存不足错误
```bash
# 降低批次大小
python train_high_res.py --dataset=cifar10 --size=64
# 系统会自动调整，或手动设置更小值

# 关闭混合精度
python train_high_res.py --dataset=cifar10 --size=64 --no-amp
```

### 2. CelebA数据集下载失败
CelebA需要手动下载，或使用STL-10作为替代：
```bash
python train_high_res.py --dataset=imagenet --size=96
```

### 3. 模型编译错误
```bash
python train_high_res.py --dataset=cifar10 --size=64 --no-compile
```

### 4. 训练速度慢
- 确保使用GPU: `torch.cuda.is_available()`
- 检查CUDA版本兼容性
- 考虑使用较小分辨率先验证

## 📈 训练监控

### 损失值解读
- **> 0.5**: 模型还在学习基础噪声去除
- **0.3-0.5**: 开始形成图像轮廓  
- **0.15-0.3**: 可识别的对象结构
- **0.08-0.15**: 清晰的图像细节
- **< 0.08**: 高质量生成

### 检查点策略
- **最佳模型**: 基于损失自动保存
- **定期检查点**: 每10个epoch保存
- **快速采样**: 每5个epoch生成监控样本

## 🎯 性能优化建议

### 1. GPU优化
- 使用RTX 30/40系列享受最佳性能
- 确保足够的VRAM余量
- 监控GPU利用率和内存使用

### 2. 数据加载优化
- 增加`num_workers`到CPU核心数
- 使用SSD存储数据集
- 启用`pin_memory`加速GPU传输

### 3. 训练策略
- 从低分辨率开始，逐步提高
- 使用预训练模型fine-tuning
- 适当调整学习率和调度器

## 🔄 从32x32升级到高分辨率

如果您已经有32x32的训练经验，升级很简单：

```bash
# 原来的训练
python train.py  # 32x32 CIFAR-10

# 升级到64x64
python train_high_res.py --dataset=cifar10 --size=64

# 升级到128x128 (需要更强GPU)
python train_high_res.py --dataset=cifar10 --size=128
```

所有优化都会自动应用，无需手动调整参数！ 