import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.hcan_vae import MultiHeadSelfAttentionConv
from convolutions import DepthwiseConv1d, PointwiseConv1d

class FusedSwishLayerNorm(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.norm(x)
        x = x.permute(0, 2, 1)
        return F.silu(x)
    
class SqueezeExcitation(nn.Module):
    """Squeeze and Excitation block for channel attention"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)  

class SResidualCell(nn.Module):
    """ Conv1x1 -> Self-Attention(Swish) -> Self-Attention(Tanh) -> Element-wise -> BatchNorm -> SE
    """
    def __init__(self, in_channels, num_heads=2, reduction=16):
        super().__init__()
        
        self.conv1x1 = nn.Conv1d(in_channels, in_channels,kernel_size=1)
        self.depthwise_conv = DepthwiseConv1d(in_channels, kernel_size=3, padding=1)

        
        self.self_attn = MultiHeadSelfAttentionConv(in_channels, num_heads)
        
        self.batch_norm = nn.BatchNorm1d(in_channels)
        
        self.se = SqueezeExcitation(in_channels, reduction)
        
        
    def forward(self, x):
        residual = x
        
        # Initial convolutions
        x = self.conv1x1(x)
        x = self.depthwise_conv(x)
        
        # Single attention with different activations
        attn_out = self.self_attn(x)
        att_swish = F.silu(attn_out)
        att_tanh = torch.tanh(attn_out)
        
        # Element-wise multiplication
        x = att_swish * att_tanh
        
        # Batch normalization
        x = self.batch_norm(x)
        
        # Squeeze and Excitation
        x = self.se(x)
                
        # Residual connection
        return residual + x

