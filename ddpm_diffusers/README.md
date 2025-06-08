# DDPM 扩散模型 - 训练与推理

这个项目使用 Hugging Face Diffusers 库实现了一个完整的 DDPM (Denoising Diffusion Probabilistic Models) 框架，支持训练和推理。

## 功能特性

- ✅ 支持多种数据集 (CIFAR-10, MNIST, 自定义数据集)
- ✅ 使用 Accelerate 库支持多GPU训练和混合精度
- ✅ TensorBoard 可视化训练过程
- ✅ 自动保存检查点和生成的样本
- ✅ 完整的推理脚本，支持批量生成
- ✅ 潜在空间插值和GIF动画生成
- ✅ 灵活的配置系统

## 项目结构

```
ddpm_diffusers/
├── config.py          # 配置文件
├── dataset.py         # 数据加载器
├── train_ddpm.py      # 训练脚本
├── inference.py       # 推理脚本
├── requirements.txt   # 依赖包
└── README.md         # 说明文档
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 快速开始

### 1. 训练模型

使用 CIFAR-10 数据集训练：

```bash
cd ddpm_diffusers
python train_ddpm.py
```

训练过程中会：
- 自动下载 CIFAR-10 数据集
- 保存训练样本可视化
- 定期生成样本图像
- 保存模型检查点
- 记录训练日志到 TensorBoard

### 2. 推理生成图像

使用训练好的模型生成图像：

```bash
python inference.py --model_path ./results/final_model --num_images 16 --output_dir ./inference_results
```

### 3. 潜在空间插值

生成插值动画：

```bash
python inference.py --model_path ./results/final_model --interpolate --interp_steps 10 --output_dir ./interpolation_results
```

## 详细使用说明

### 配置系统

在 `config.py` 中可以修改所有训练和模型参数：

```python
# 主要配置选项
training_config.dataset_name = "cifar10"  # "cifar10", "mnist", "custom"
training_config.image_size = 64           # 图像尺寸
training_config.train_batch_size = 16     # 批次大小
training_config.num_epochs = 100          # 训练轮数
training_config.learning_rate = 1e-4      # 学习率
```

### 使用自定义数据集

1. 修改配置：

```python
training_config.dataset_name = "custom"
data_config.dataset_path = "/path/to/your/images"
```

2. 数据集目录结构：

```
your_dataset/
├── image1.jpg
├── image2.png
├── subfolder/
│   ├── image3.jpg
│   └── image4.png
└── ...
```

### 训练脚本参数

`train_ddpm.py` 支持以下主要功能：

- **多GPU训练**: 使用 Accelerate 自动检测并使用多GPU
- **混合精度**: 默认开启 FP16 混合精度训练
- **自动保存**: 定期保存模型检查点和样本图像
- **TensorBoard日志**: 记录损失、学习率、生成样本等

### 推理脚本参数

```bash
python inference.py \
    --model_path ./results/final_model \      # 模型路径
    --num_images 32 \                         # 生成图像数量
    --num_inference_steps 1000 \              # 推理步数(越多质量越好)
    --seed 42 \                               # 随机种子
    --grid_size 8 \                           # 网格大小
    --output_dir ./outputs                    # 输出目录
```

**插值参数：**

```bash
python inference.py \
    --model_path ./results/final_model \
    --interpolate \                           # 开启插值模式
    --interp_steps 16 \                       # 插值步数
    --start_seed 0 \                          # 起始种子
    --end_seed 1000                           # 结束种子
```

## 训练监控

### TensorBoard

启动 TensorBoard 查看训练过程：

```bash
tensorboard --logdir ./results/logs
```

可以查看：
- 训练损失曲线
- 学习率变化
- 生成的样本图像
- 模型架构图

### 输出文件

训练过程中会生成以下文件：

```
results/
├── logs/                           # TensorBoard日志
├── sample_training_data.png        # 训练数据样本
├── samples_epoch_10.png           # 每10轮生成的样本
├── samples_epoch_20.png
├── ...
├── checkpoint-epoch-10/           # 模型检查点
├── checkpoint-epoch-20/
├── ...
├── final_model/                   # 最终模型
└── final_samples.png             # 最终生成样本
```

## 模型架构

### UNet2D 配置

```python
UNet2DModel(
    sample_size=64,                           # 图像尺寸
    in_channels=3,                           # 输入通道数
    out_channels=3,                          # 输出通道数
    layers_per_block=2,                      # 每块的层数
    block_out_channels=[128, 128, 256, 256, 512, 512],  # 通道数配置
    down_block_types=[...],                  # 下采样块类型
    up_block_types=[...],                    # 上采样块类型
    attention_head_dim=8,                    # 注意力头维度
)
```

### 噪声调度器

```python
DDPMScheduler(
    num_train_timesteps=1000,                # 训练时间步数
    beta_start=0.0001,                       # β起始值
    beta_end=0.02,                           # β结束值
    beta_schedule="linear",                  # β调度方式
    prediction_type="epsilon",               # 预测类型
)
```

## 性能优化

### 训练加速

1. **多GPU训练**: 自动检测并使用所有可用GPU
2. **混合精度**: 使用 FP16 减少显存占用，加速训练
3. **梯度累积**: 支持梯度累积模拟更大批次
4. **数据加载**: 多线程数据加载，避免IO瓶颈

### 推理优化

1. **批量生成**: 自动分批生成，避免显存溢出
2. **FP16推理**: GPU推理时自动使用半精度
3. **缓存优化**: 合理的批次大小设置

## 常见问题

### Q: 显存不足怎么办？

A: 可以：
- 减小 `train_batch_size`
- 启用梯度累积 `gradient_accumulation_steps > 1`
- 使用更小的图像尺寸
- 修改模型架构，减少通道数

### Q: 训练速度慢怎么办？

A: 可以：
- 确保启用混合精度训练
- 增加数据加载线程数 `dataloader_num_workers`
- 使用SSD存储数据集
- 检查GPU利用率

### Q: 生成质量不好怎么办？

A: 可以：
- 增加训练轮数
- 调整学习率
- 增加推理步数 `num_inference_steps`
- 使用更复杂的模型架构

### Q: 如何微调预训练模型？

A: 可以：
1. 加载预训练模型权重
2. 使用较小的学习率
3. 冻结部分层进行微调

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！

## 致谢

本项目基于以下优秀的开源项目：
- [Hugging Face Diffusers](https://github.com/huggingface/diffusers)
- [Hugging Face Accelerate](https://github.com/huggingface/accelerate)
- [PyTorch](https://pytorch.org/) 