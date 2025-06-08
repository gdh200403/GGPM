"""
DDPM算法核心实现
"""
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


class DDPM:
    """DDPM算法实现"""
    
    def __init__(self, model, timesteps=1000, beta_start=0.0001, beta_end=0.02, 
                 beta_schedule='linear', device='cuda'):
        self.model = model
        self.timesteps = timesteps
        self.device = device
        
        # 创建beta调度
        if beta_schedule == 'linear':
            self.betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        elif beta_schedule == 'cosine':
            s = 0.008
            steps = timesteps + 1
            x = torch.linspace(0, timesteps, steps, device=device)
            alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clip(betas, 0.0001, 0.9999)
        
        # 预计算常数
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # 用于采样的常数
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        
        # 用于q(x_{t-1} | x_t, x_0)的后验方差
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
    
    def q_sample(self, x_start, t, noise=None):
        """前向扩散过程：q(x_t | x_0)"""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def predict_start_from_noise(self, x_t, t, noise):
        """从噪声预测原始图像"""
        sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod[t])[:, None, None, None]
        sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod[t] - 1)[:, None, None, None]
        
        return sqrt_recip_alphas_cumprod * x_t - sqrt_recipm1_alphas_cumprod * noise
    
    def q_posterior_mean_variance(self, x_start, x_t, t):
        """计算后验分布q(x_{t-1} | x_t, x_0)的均值和方差"""
        posterior_mean = (
            torch.sqrt(self.alphas_cumprod_prev[t])[:, None, None, None] * self.betas[t][:, None, None, None] * x_start / (1.0 - self.alphas_cumprod[t])[:, None, None, None]
            + torch.sqrt(self.alphas[t])[:, None, None, None] * (1.0 - self.alphas_cumprod_prev[t])[:, None, None, None] * x_t / (1.0 - self.alphas_cumprod[t])[:, None, None, None]
        )
        posterior_variance = self.posterior_variance[t][:, None, None, None]
        
        return posterior_mean, posterior_variance
    
    def p_mean_variance(self, x_t, t):
        """计算p(x_{t-1} | x_t)的均值和方差"""
        # 预测噪声
        predicted_noise = self.model(x_t, t)
        
        # 预测原始图像
        x_start = self.predict_start_from_noise(x_t, t, predicted_noise)
        
        # 计算后验均值和方差
        model_mean, posterior_variance = self.q_posterior_mean_variance(x_start, x_t, t)
        
        return model_mean, posterior_variance
    
    def p_sample(self, x_t, t):
        """从p(x_{t-1} | x_t)采样"""
        model_mean, model_variance = self.p_mean_variance(x_t, t)
        
        noise = torch.randn_like(x_t)
        # 当t=0时不添加噪声
        nonzero_mask = (t != 0).float()[:, None, None, None]
        
        return model_mean + nonzero_mask * torch.sqrt(model_variance) * noise
    
    def sample(self, shape, return_all_timesteps=False):
        """从纯噪声生成样本"""
        device = next(self.model.parameters()).device
        b = shape[0]
        
        # 从纯噪声开始
        x = torch.randn(shape, device=device)
        
        if return_all_timesteps:
            imgs = [x]
        
        # 逐步去噪
        for i in tqdm(reversed(range(0, self.timesteps)), desc='采样中', total=self.timesteps):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, t)
            if return_all_timesteps:
                imgs.append(x)
        
        if return_all_timesteps:
            return torch.stack(imgs, dim=1)
        return x
    
    def training_loss(self, x_start):
        """计算训练损失"""
        batch_size = x_start.shape[0]
        device = x_start.device
        
        # 随机采样时间步
        t = torch.randint(0, self.timesteps, (batch_size,), device=device).long()
        
        # 采样噪声
        noise = torch.randn_like(x_start)
        
        # 前向扩散
        x_noisy = self.q_sample(x_start, t, noise)
        
        # 预测噪声
        predicted_noise = self.model(x_noisy, t)
        
        # 计算MSE损失
        loss = F.mse_loss(predicted_noise, noise)
        
        return loss 