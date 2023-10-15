from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
from torch import nn, Tensor


from timm.models.layers import trunc_normal_
import math
from math import log
from torch.nn.init import calculate_gain
from torch.nn.parameter import Parameter
from ops_dcnv3.modules.dcnv3 import DCNv3



class DWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x: Tensor, H, W) -> Tensor:
        B, _, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        return x.flatten(2).transpose(1, 2)


class ConvModule(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, 1, bias=False)
        self.bn = nn.BatchNorm2d(c2)        # use SyncBN in original
        self.activate = nn.ReLU(True)

    def forward(self, x: Tensor) -> Tensor:
        return self.activate(self.bn(self.conv(x)))


class PAM_Module(nn.Module):
    """ Position attention module"""

    # Ref from SAGAN
    def __init__(self, dim, num_heads=8, sr_ratio=8, qkv_bias=False):
        super(PAM_Module, self).__init__()
        self.sr_ratio = sr_ratio
        self.chanel_in = dim
        self.num_heads = num_heads
        self.kv = nn.Linear(dim, 2 * dim, bias=qkv_bias)   # k，v channel all are cut to small dims;
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, sr_ratio, sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        B, N, C = x.shape
        q = x.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        if self.sr_ratio > 1:
            x = x.permute(0, 2, 1).reshape(B, C, H, W)
            x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
            x = self.norm(x)
        k, v = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()

        return q, k, v


class CrossPAM(nn.Module):
    """ Position attention for two modality"""
    def __init__(self, in_dim, num_heads=8, sr_ratio=8):
        super(CrossPAM, self).__init__()
        self.channel_in = in_dim

        self.norm1 = nn.LayerNorm(in_dim)
        self.extra_norm1 = nn.LayerNorm(in_dim)

        self.pam = PAM_Module(in_dim, num_heads=8, sr_ratio=sr_ratio)
        self.extra_pam = PAM_Module(in_dim, num_heads=8, sr_ratio=sr_ratio)

        self.proj = nn.Linear(in_dim, in_dim)
        self.extra_proj = nn.Linear(in_dim, in_dim)

        self.scale = (in_dim//num_heads) ** -0.5

        self.drop_path = nn.Identity()

        self.norm2 = nn.LayerNorm(in_dim)
        self.extra_norm2 = nn.LayerNorm(in_dim)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x1, x2):
        B, C, H, W = x1.shape

        x1 = x1.flatten(2).transpose(1, 2)  # B C H W -> B N C
        x2 = x2.flatten(2).transpose(1, 2)  # B C H W -> B N C
        x1 = self.norm1(x1)
        x2 = self.extra_norm1(x2)

        q1, k1, v1 = self.pam(x1, H, W)   # size Cx(H*W)
        q2, k2, v2 = self.extra_pam(x2, H, W)

        attn1 = (q1 @ k2.transpose(-2, -1)) * self.scale  # (B, head, N, C/head) * (B, head, C/head, N) = N*N
        attn1 = attn1.softmax(dim=-1)
        attn2 = (q2 @ k1.transpose(-2, -1)) * self.scale  # (B, head, N, C/head) * (B, head, C/head, N) = N*N
        attn2 = attn2.softmax(dim=-1)

        x1 = (attn2 @ v1).transpose(1, 2).reshape(B, H*W, C)
        x1 = self.proj(x1)
        x2 = (attn1 @ v2).transpose(1, 2).reshape(B, H*W, C)
        x2 = self.proj(x2)


        x1 = x1.permute(0, 2, 1).reshape(B, C, H, W).contiguous()  # B N C -> B C N -> B C H W
        x2 = x2.permute(0, 2, 1).reshape(B, C, H, W).contiguous()  # B N C -> B C N -> B C H W

        return x1, x2


# Stage 1
# External spatial attention for CARM

# CARM Modal
class ExternalAttentionRectifyModule(nn.Module):
    def __init__(self, dim, num_heads=8, sr_ratio=8, reduction=1, lambda_c=.5, lambda_s=.5, norm_layer=nn.BatchNorm2d):
        super(ExternalAttentionRectifyModule, self).__init__()
        # self.external_weights = MultiHeadAttention(dim=dim, norm_layer=norm_layer)
        self.external_weights = CrossPAM(in_dim=dim, num_heads=8, sr_ratio=8)
        self.norm = norm_layer(dim)
        # self.channel_embd = ConvModule(2*dim, dim)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x1, x2):
        attn_x1, attn_x2 = self.external_weights(x1, x2)
        out_x1 = x1 + attn_x1
        out_x2 = x2 + attn_x2
        return out_x1, out_x2


#         stage 2
#         FFMs
class AttentionModule(nn.Module):
    '''
           Arguments:
               kernel (int): The largest kernel size of the LKA's Conv.
    '''
    def __init__(self, dim, kernel=7):
        super().__init__()
        if kernel == 5:
            k1, p1 = 3, 1
            k2, p2, d2 = 5, 2, 1
        elif kernel == 7:
            k1, p1 = 5, 2
            k2, p2, d2 = 7, 9, 3
        else:
            k1, p1 = 5, 2
            k2, p2, d2 = 7, 9, 3
        self.conv0 = nn.Conv2d(dim, dim, k1, padding=p1, groups=dim)
        self.conv_spatial = nn.Conv2d(
            dim, dim, k2, stride=1, padding=p2, groups=dim, dilation=d2)
        self.conv1 = nn.Conv2d(dim, dim, 1)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.softmax(attn)
        attn = self.conv1(attn)
        return u * attn


# It's a good attention mechanism  for rectification of two different modalities
class BidirectionalAttention(nn.Module):
    def __init__(self, dim, num_heads=8, kernel=7, qkv_bias=False, qk_scale=None):
        super(BidirectionalAttention, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.proj_1 = nn.Conv2d(dim, dim, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = AttentionModule(dim, kernel=kernel)
        self.proj_2 = nn.Conv2d(dim, dim, 1)

        self.proj_1_e = nn.Conv2d(dim, dim, 1)
        self.activation_e = nn.GELU()
        self.spatial_gating_unit_e = AttentionModule(dim, kernel=kernel)
        self.proj_2_e = nn.Conv2d(dim, dim, 1)

    @torch.jit.script
    def combine_add(x, attn):
        return x + attn

    def forward(self, x1, x2):
        B, C, H, W = x1.shape
        residual1 = x1.clone()
        residual2 = x2.clone()
        # x = torch.cat([x1, x2], dim=1)     # concatenation in channel dims
        x1 = self.proj_1(x1)
        x1 = self.activation(x1)
        x1 = self.spatial_gating_unit(x1)
        x1 = self.proj_2(x1)
        # attn1, attn2 = torch.split(x, [C, C], dim=1)  # split two attention values

        x2 = self.proj_1(x2)
        x2 = self.activation(x2)
        x2 = self.spatial_gating_unit(x2)
        x2 = self.proj_2(x2)

        attn1 = x1 * residual1
        attn2 = x2 * residual2
        x1 = self.combine_add(residual1, attn2)
        x2 = self.combine_add(residual2, attn1)
        return x1, x2


# Stage 2 Fusion Module
# -------*****--------- #


class FeatureEmbed(nn.Module):
    def __init__(self, in_channels, out_channels, ratio=2, act_layer=nn.GELU, drop=0.1, norm_layer=nn.BatchNorm2d):
        super(FeatureEmbed, self).__init__()
        """
           MLP Feature-Embed Block: 
        """
        out_features = out_channels
        hidden_features = in_channels * ratio
        self.fc1 = nn.Linear(in_channels, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        # self.residual = nn.Conv2d(in_channels, out_features, 1)

        self.apply(self._init_weights)
        self.norm1 = nn.LayerNorm(in_channels)
        self.norm = norm_layer(out_channels)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # B C H W -> B N C
        x = self.norm1(x)
        # x_residual = x
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        # x_residual = x_residual.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        x = self.norm(x.permute(0, 2, 1).reshape(B, C // 2, H, W).contiguous())  # B N C -> B C N -> B C H W

        # x = self.residual(x_residual) + x

        return x


class CrossFusionModule(nn.Module):
    def __init__(self, dim, ratio=2, num_heads=None, kernel=7, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.cross = BidirectionalAttention(dim=dim, num_heads=num_heads, kernel=kernel)
        self.channel_emb = FeatureEmbed(in_channels=dim * 2, out_channels=dim, ratio=ratio)
        # self.spatial_emb = FeatureEmbed(in_channels=dim * 2, out_channels=dim, reduction=reduction)

    def forward(self, x1, x2):
        x1, x2 = self.cross(x1, x2)
        x = torch.cat((x1, x2), dim=1)
        merge = self.channel_emb(x)
        # merge_aux = self.spatial_emb(x)

        return merge

