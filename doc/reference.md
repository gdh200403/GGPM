# 参考文献和代码库

## 官方实现

- 论文：[2006.11239](https://arxiv.org/pdf/2006.11239)
  代码库：[hojonathanho/diffusion: Denoising Diffusion Probabilistic Models](https://github.com/hojonathanho/diffusion)

## 博客

1. 篇幅最长、数学推导、原理分析，无代码：[What are Diffusion Models? | Lil'Log](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/#:~:text=Diffusion models are inspired by,data samples from the noise)

2. huggingface文章，讲解代码，讲解详尽：[《注解扩散模型》 --- The Annotated Diffusion Model](https://huggingface.co/blog/annotated-diffusion)，讲解的就是代码库2

3. 简洁讲解和实现：[《机器学习中的扩散模型导论》 --- Introduction to Diffusion Models for Machine Learning](https://www.assemblyai.com/blog/diffusion-models-for-machine-learning-introduction)，调用代码库2简单实现。

4. 个人博客：[笔记｜扩散模型（一）：DDPM 理论与实现 | 極東晝寢愛好家](https://littlenyima.github.io/posts/13-denoising-diffusion-probabilistic-models/)，讲解细致，有代码库，手把手教实现，基于diffuser搭建网络。
   对应代码库：[LittleNyima/code-snippets](https://github.com/LittleNyima/code-snippets/tree/master)

## 代码库

1. CIFAR10训练，1.9k stars，作者声称“最简单的 DDPM 实现，用 CIFAR-10 数据集进行了训练”：[zoubohao/DenoisingDiffusionProbabilityModel-ddpm-: This may be the simplest implement of DDPM. You can directly run Main.py to train the UNet on CIFAR-10 dataset and see the amazing process of denoising.](https://github.com/zoubohao/DenoisingDiffusionProbabilityModel-ddpm-/tree/main)

2. star数最高，9.4k，厉害实现，Phil Wang 基于原始 TensorFlow 实现的 PyTorch 实现：[lucidrains/denoising-diffusion-pytorch: Implementation of Denoising Diffusion Probabilistic Model in Pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch?tab=readme-ov-file)

