"""
简洁的UNet模型实现
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def timestep_embedding(timesteps, dim):
    """时间步嵌入"""
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
    emb = timesteps[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    return emb


class ResBlock(nn.Module):
    """残差块"""
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        
        self.block1 = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )
        
        self.block2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x, time_emb):
        h = self.block1(x)
        h += self.time_mlp(time_emb)[:, :, None, None]
        h = self.block2(h)
        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    """简化的注意力块"""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
    
    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).view(B, 3, C, H * W)
        q, k, v = qkv.unbind(1)
        
        # 简化的注意力计算
        attn = torch.softmax(q.transpose(-2, -1) @ k / math.sqrt(C), dim=-1)
        h = (v @ attn.transpose(-2, -1)).view(B, C, H, W)
        
        return x + self.proj(h)


class UNet(nn.Module):
    """简洁的UNet模型"""
    def __init__(self, in_channels=3, dim=64, dim_mults=(1, 2, 4, 8)):
        super().__init__()
        
        # 时间嵌入
        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            nn.Linear(dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # 初始卷积
        self.init_conv = nn.Conv2d(in_channels, dim, 7, padding=3)
        
        # 下采样路径
        self.downs = nn.ModuleList()
        dims = [dim] + [dim * m for m in dim_mults]
        
        for i in range(len(dim_mults)):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            
            self.downs.append(nn.ModuleList([
                ResBlock(in_dim, out_dim, time_dim),
                ResBlock(out_dim, out_dim, time_dim),
                AttentionBlock(out_dim) if i >= 2 else nn.Identity(),
                nn.Conv2d(out_dim, out_dim, 4, 2, 1) if i < len(dim_mults) - 1 else nn.Identity()
            ]))
        
        # 中间层
        mid_dim = dims[-1]
        self.mid_block1 = ResBlock(mid_dim, mid_dim, time_dim)
        self.mid_attn = AttentionBlock(mid_dim)
        self.mid_block2 = ResBlock(mid_dim, mid_dim, time_dim)
        
        # 上采样路径
        self.ups = nn.ModuleList()
        
        for i in reversed(range(len(dim_mults))):
            in_dim = dims[i + 1]
            out_dim = dims[i]
            
            self.ups.append(nn.ModuleList([
                ResBlock(in_dim * 2, in_dim, time_dim),
                ResBlock(in_dim, out_dim, time_dim),
                AttentionBlock(out_dim) if i >= 2 else nn.Identity(),
                nn.ConvTranspose2d(out_dim, out_dim, 4, 2, 1) if i > 0 else nn.Identity()
            ]))
        
        # 输出层
        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, dim),
            nn.SiLU(),
            nn.Conv2d(dim, in_channels, 3, padding=1)
        )
    
    def forward(self, x, timesteps):
        # 时间嵌入
        t = timestep_embedding(timesteps, self.time_mlp[0].in_features)
        t = self.time_mlp(t)
        
        # 初始特征
        x = self.init_conv(x)
        
        # 存储跳跃连接
        skips = []
        
        # 下采样
        for resblock1, resblock2, attn, downsample in self.downs:
            x = resblock1(x, t)
            x = resblock2(x, t)
            x = attn(x)
            skips.append(x)
            x = downsample(x)
        
        # 中间层
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)
        
        # 上采样
        for resblock1, resblock2, attn, upsample in self.ups:
            x = torch.cat([x, skips.pop()], dim=1)
            x = resblock1(x, t)
            x = resblock2(x, t) 
            x = attn(x)
            x = upsample(x)
        
        return self.final_conv(x) 