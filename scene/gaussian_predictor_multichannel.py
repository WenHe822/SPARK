import torch
import torch.nn as nn
import torchvision 

import numpy as np
import torch
from torch.nn.functional import silu

from einops import rearrange, repeat

from utils.general_utils import matrix_to_quaternion, quaternion_raw_multiply
from utils.graphics_utils import fov2focal
import torch.nn.functional as F
# U-Net implementation from EDM
# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Model architectures and preconditioning schemes used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

#----------------------------------------------------------------------------
# Unified routine for initializing weights and biases.

def weight_init(shape, mode, fan_in, fan_out):
    if mode == 'xavier_uniform': return np.sqrt(6 / (fan_in + fan_out)) * (torch.rand(*shape) * 2 - 1)
    if mode == 'xavier_normal':  return np.sqrt(2 / (fan_in + fan_out)) * torch.randn(*shape)
    if mode == 'kaiming_uniform': return np.sqrt(3 / fan_in) * (torch.rand(*shape) * 2 - 1)
    if mode == 'kaiming_normal':  return np.sqrt(1 / fan_in) * torch.randn(*shape)
    raise ValueError(f'Invalid init mode "{mode}"')

#----------------------------------------------------------------------------
# Fully-connected layer.

class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, init_mode='kaiming_normal', init_weight=1, init_bias=0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        init_kwargs = dict(mode=init_mode, fan_in=in_features, fan_out=out_features)
        self.weight = torch.nn.Parameter(weight_init([out_features, in_features], **init_kwargs) * init_weight)
        self.bias = torch.nn.Parameter(weight_init([out_features], **init_kwargs) * init_bias) if bias else None

    def forward(self, x):
        x = x @ self.weight.to(x.dtype).t()
        if self.bias is not None:
            x = x.add_(self.bias.to(x.dtype))
        return x

#----------------------------------------------------------------------------
# Convolutional layer with optional up/downsampling.

class Conv2d(torch.nn.Module):
    def __init__(self,
        in_channels, out_channels, kernel, bias=True, up=False, down=False,
        resample_filter=[1,1], fused_resample=False, init_mode='kaiming_normal', init_weight=1, init_bias=0,
    ):
        assert not (up and down)
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up
        self.down = down
        self.fused_resample = fused_resample
        init_kwargs = dict(mode=init_mode, fan_in=in_channels*kernel*kernel, fan_out=out_channels*kernel*kernel)
        self.weight = torch.nn.Parameter(weight_init([out_channels, in_channels, kernel, kernel], **init_kwargs) * init_weight) if kernel else None
        self.bias = torch.nn.Parameter(weight_init([out_channels], **init_kwargs) * init_bias) if kernel and bias else None
        f = torch.as_tensor(resample_filter, dtype=torch.float32)
        f = f.ger(f).unsqueeze(0).unsqueeze(1) / f.sum().square()
        self.register_buffer('resample_filter', f if up or down else None)

    def forward(self, x, N_views_xa=1):
        w = self.weight.to(x.dtype) if self.weight is not None else None
        b = self.bias.to(x.dtype) if self.bias is not None else None
        f = self.resample_filter.to(x.dtype) if self.resample_filter is not None else None
        w_pad = w.shape[-1] // 2 if w is not None else 0
        f_pad = (f.shape[-1] - 1) // 2 if f is not None else 0

        if self.fused_resample and self.up and w is not None:
            x = torch.nn.functional.conv_transpose2d(x, f.mul(4).tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=max(f_pad - w_pad, 0))
            x = torch.nn.functional.conv2d(x, w, padding=max(w_pad - f_pad, 0))
        elif self.fused_resample and self.down and w is not None:
            x = torch.nn.functional.conv2d(x, w, padding=w_pad+f_pad)
            x = torch.nn.functional.conv2d(x, f.tile([self.out_channels, 1, 1, 1]), groups=self.out_channels, stride=2)
        else:
            if self.up:
                x = torch.nn.functional.conv_transpose2d(x, f.mul(4).tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=f_pad)
            if self.down:
                x = torch.nn.functional.conv2d(x, f.tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=f_pad)
            if w is not None:
                x = torch.nn.functional.conv2d(x, w, padding=w_pad)
        if b is not None:
            x = x.add_(b.reshape(1, -1, 1, 1))
        return x

#----------------------------------------------------------------------------
# Group normalization.

class GroupNorm(torch.nn.Module):
    def __init__(self, num_channels, num_groups=32, min_channels_per_group=4, eps=1e-5):
        super().__init__()
        self.num_groups = min(num_groups, num_channels // min_channels_per_group)
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(num_channels))
        self.bias = torch.nn.Parameter(torch.zeros(num_channels))

    def forward(self, x, N_views_xa=1):
        x = torch.nn.functional.group_norm(x, num_groups=self.num_groups, weight=self.weight.to(x.dtype), bias=self.bias.to(x.dtype), eps=self.eps)
        return x.to(memory_format=torch.channels_last)

#----------------------------------------------------------------------------
# Attention weight computation, i.e., softmax(Q^T * K).
# Performs all computation using FP32, but uses the original datatype for
# inputs/outputs/gradients to conserve memory.

class AttentionOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k):
        w = torch.einsum('ncq,nck->nqk', q.to(torch.float32), (k / np.sqrt(k.shape[1])).to(torch.float32)).softmax(dim=2).to(q.dtype)
        ctx.save_for_backward(q, k, w)
        return w

    @staticmethod
    def backward(ctx, dw):
        q, k, w = ctx.saved_tensors
        db = torch._softmax_backward_data(grad_output=dw.to(torch.float32), output=w.to(torch.float32), dim=2, input_dtype=torch.float32)
        dq = torch.einsum('nck,nqk->ncq', k.to(torch.float32), db).to(q.dtype) / np.sqrt(k.shape[1])
        dk = torch.einsum('ncq,nqk->nck', q.to(torch.float32), db).to(k.dtype) / np.sqrt(k.shape[1])
        return dq, dk
 
#----------------------------------------------------------------------------
# Timestep embedding used in the DDPM++ and ADM architectures.

# class PositionalEmbedding(torch.nn.Module):
#     def __init__(self, num_channels, max_positions=10000, endpoint=False):
#         super().__init__()
#         self.num_channels = num_channels
#         self.max_positions = max_positions
#         self.endpoint = endpoint

#     def forward(self, x):
#         b, c = x.shape
#         x = rearrange(x, 'b c -> (b c)')
#         freqs = torch.arange(start=0, end=self.num_channels//2, dtype=torch.float32, device=x.device)
#         freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
#         freqs = (1 / self.max_positions) ** freqs
#         x = x.ger(freqs.to(x.dtype))
#         x = torch.cat([x.cos(), x.sin()], dim=1)
#         x = rearrange(x, '(b c) emb_ch -> b (c emb_ch)', b=b)
#         return x
class PositionalEmbedding(torch.nn.Module):
    """位置嵌入层。

    该类实现了位置嵌入，用于将时间步嵌入到模型中。

    Args:
        num_channels (int): 嵌入的通道数。
        max_positions (int): 最大位置数，默认为 10000。
        endpoint (bool): 是否包含端点，默认为 False。
    """
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        """前向传播。

        计算输入 x 的位置嵌入。

        Args:
            x (torch.Tensor): 输入张量，形状为 (batch_size, num_channels)。

        Returns:
            torch.Tensor: 嵌入后的输出张量，形状为 (batch_size, num_channels * 2)。
        """
        b, c = x.shape
        x = rearrange(x, 'b c -> (b c)')
        freqs = torch.arange(start=0, end=self.num_channels//2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        x = rearrange(x, '(b c) emb_ch -> b (c emb_ch)', b=b)
        return x
#----------------------------------------------------------------------------
# Timestep embedding used in the NCSN++ architecture.

class FourierEmbedding(torch.nn.Module):
    def __init__(self, num_channels, scale=16):
        super().__init__()
        self.register_buffer('freqs', torch.randn(num_channels // 2) * scale)

    def forward(self, x):
        b, c = x.shape
        x = rearrange(x, 'b c -> (b c)')
        x = x.ger((2 * np.pi * self.freqs).to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        x = rearrange(x, '(b c) emb_ch -> b (c emb_ch)', b=b)
        return x  

class CrossAttentionBlock(torch.nn.Module):
    def __init__(self, num_channels, num_heads = 1, eps=1e-5):
        super().__init__()

        self.num_heads = 1
        init_attn = dict(init_mode='xavier_uniform', init_weight=np.sqrt(0.2))
        init_zero = dict(init_mode='xavier_uniform', init_weight=1e-5)

        self.norm = GroupNorm(num_channels=num_channels, eps=eps)

        self.q_proj = Conv2d(in_channels=num_channels, out_channels=num_channels, kernel=1, **init_attn)
        self.kv_proj = Conv2d(in_channels=num_channels, out_channels=num_channels*2, kernel=1, **init_attn)

        self.out_proj = Conv2d(in_channels=num_channels, out_channels=num_channels, kernel=3, **init_zero)

    def forward(self, q, kv):
        q_proj = self.q_proj(self.norm(q)).reshape(q.shape[0] * self.num_heads, q.shape[1] // self.num_heads, -1)
        k_proj, v_proj = self.kv_proj(self.norm(kv)).reshape(kv.shape[0] * self.num_heads, 
                                                   kv.shape[1] // self.num_heads, 2, -1).unbind(2)
        w = AttentionOp.apply(q_proj, k_proj)
        a = torch.einsum('nqk,nck->ncq', w, v_proj)
        x = self.out_proj(a.reshape(*q.shape)).add_(q)

        return x

#----------------------------------------------------------------------------
# Unified U-Net block with optional up/downsampling and self-attention.
# Represents the union of all features employed by the DDPM++, NCSN++, and
# ADM architectures.

class UNetBlock(torch.nn.Module):
    def __init__(self,
        in_channels, out_channels, emb_channels, up=False, down=False, attention=False,
        num_heads=None, channels_per_head=64, dropout=0, skip_scale=1, eps=1e-5,
        resample_filter=[1,1], resample_proj=False, adaptive_scale=True,
        init=dict(), init_zero=dict(init_weight=0), init_attn=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if emb_channels is not None:
            self.affine = Linear(in_features=emb_channels, out_features=out_channels*(2 if adaptive_scale else 1), **init)
        self.num_heads = 0 if not attention else num_heads if num_heads is not None else out_channels // channels_per_head
        self.dropout = dropout
        self.skip_scale = skip_scale
        self.adaptive_scale = adaptive_scale

        self.norm0 = GroupNorm(num_channels=in_channels, eps=eps)
        self.conv0 = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel=3, up=up, down=down, resample_filter=resample_filter, **init)
        self.norm1 = GroupNorm(num_channels=out_channels, eps=eps)
        self.conv1 = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=3, **init_zero)

        self.skip = None
        if out_channels != in_channels or up or down:
            kernel = 1 if resample_proj or out_channels!= in_channels else 0
            self.skip = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel=kernel, up=up, down=down, resample_filter=resample_filter, **init)

        if self.num_heads:
            self.norm2 = GroupNorm(num_channels=out_channels, eps=eps)
            self.qkv = Conv2d(in_channels=out_channels, out_channels=out_channels*3, kernel=1, **(init_attn if init_attn is not None else init))
            self.proj = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=1, **init_zero)

    def forward(self, x, emb=None, N_views_xa=1):
        orig = x
        x = self.conv0(silu(self.norm0(x)))

        if emb is not None:
            params = self.affine(emb).unsqueeze(2).unsqueeze(3).to(x.dtype)
            if self.adaptive_scale:
                scale, shift = params.chunk(chunks=2, dim=1)
                x = silu(torch.addcmul(shift, self.norm1(x), scale + 1))
            else:
                x = silu(self.norm1(x.add_(params)))

        x = silu(self.norm1(x))

        x = self.conv1(torch.nn.functional.dropout(x, p=self.dropout, training=self.training))
        x = x.add_(self.skip(orig) if self.skip is not None else orig)
        x = x * self.skip_scale

        if self.num_heads:
            if N_views_xa != 1:
                B, C, H, W = x.shape
                # (B, C, H, W) -> (B/N, N, C, H, W) -> (B/N, N, H, W, C)
                x = x.reshape(B // N_views_xa, N_views_xa, *x.shape[1:]).permute(0, 1, 3, 4, 2)
                # (B/N, N, H, W, C) -> (B/N, N*H, W, C) -> (B/N, C, N*H, W)
                x = x.reshape(B // N_views_xa, N_views_xa * x.shape[2], *x.shape[3:]).permute(0, 3, 1, 2)
            q, k, v = self.qkv(self.norm2(x)).reshape(x.shape[0] * self.num_heads, x.shape[1] // self.num_heads, 3, -1).unbind(2)
            w = AttentionOp.apply(q, k)
            a = torch.einsum('nqk,nck->ncq', w, v)
            x = self.proj(a.reshape(*x.shape)).add_(x)
            x = x * self.skip_scale
            if N_views_xa != 1:
                # (B/N, C, N*H, W) -> (B/N, N*H, W, C)
                x = x.permute(0, 2, 3, 1)
                # (B/N, N*H, W, C) -> (B/N, N, H, W, C) -> (B/N, N, C, H, W)
                x = x.reshape(B // N_views_xa, N_views_xa, H, W, C).permute(0, 1, 4, 2, 3)
                # (B/N, N, C, H, W) -> # (B, C, H, W) 
                x = x.reshape(B, C, H, W)
        return x


#----------------------------------------------------------------------------
# Reimplementation of the DDPM++ and NCSN++ architectures from the paper
# "Score-Based Generative Modeling through Stochastic Differential
# Equations". Equivalent to the original implementation by Song et al.,
# available at https://github.com/yang-song/score_sde_pytorch
# taken from EDM repository https://github.com/NVlabs/edm/blob/main/training/networks.py#L372

class SongUNet(nn.Module):
    def __init__(self,
        img_resolution,                     # Image resolution at input/output.
        in_channels,                        # Number of color channels at input.
        out_channels,                       # Number of color channels at output.
        emb_dim_in           = 0,            # Input embedding dim.
        augment_dim         = 0,            # Augmentation label dimensionality, 0 = no augmentation.

        model_channels      = 128,          # Base multiplier for the number of channels.
        channel_mult        = [1,2,2,2],    # Per-resolution multipliers for the number of channels.
        channel_mult_emb    = 4,            # Multiplier for the dimensionality of the embedding vector.
        num_blocks          = 4,            # Number of residual blocks per resolution.
        attn_resolutions    = [16],         # List of resolutions with self-attention.
        dropout             = 0.10,         # Dropout probability of intermediate activations.
        label_dropout       = 0,            # Dropout probability of class labels for classifier-free guidance.

        embedding_type      = 'positional', # Timestep embedding type: 'positional' for DDPM++, 'fourier' for NCSN++.
        channel_mult_noise  = 0,            # Timestep embedding size: 1 for DDPM++, 2 for NCSN++.
        encoder_type        = 'standard',   # Encoder architecture: 'standard' for DDPM++, 'residual' for NCSN++.
        decoder_type        = 'standard',   # Decoder architecture: 'standard' for both DDPM++ and NCSN++.
        resample_filter     = [1,1],        # Resampling filter: [1,1] for DDPM++, [1,3,3,1] for NCSN++.
    ):
        assert embedding_type in ['fourier', 'positional']
        assert encoder_type in ['standard', 'skip', 'residual']
        assert decoder_type in ['standard', 'skip']

        super().__init__()
        self.label_dropout = label_dropout
        self.emb_dim_in = emb_dim_in
        if emb_dim_in > 0:
            emb_channels = model_channels * channel_mult_emb
        else:
            emb_channels = None
        noise_channels = model_channels * channel_mult_noise
        init = dict(init_mode='xavier_uniform')
        init_zero = dict(init_mode='xavier_uniform', init_weight=1e-5)
        init_attn = dict(init_mode='xavier_uniform', init_weight=np.sqrt(0.2))
        block_kwargs = dict(
            emb_channels=emb_channels, num_heads=1, dropout=dropout, skip_scale=np.sqrt(0.5), eps=1e-6,
            resample_filter=resample_filter, resample_proj=True, adaptive_scale=False,
            init=init, init_zero=init_zero, init_attn=init_attn,
        )

        # Mapping.
        # self.map_label = Linear(in_features=label_dim, out_features=noise_channels, **init) if label_dim else None
        # self.map_augment = Linear(in_features=augment_dim, out_features=noise_channels, bias=False, **init) if augment_dim else None
        # self.map_layer0 = Linear(in_features=noise_channels, out_features=emb_channels, **init)
        # self.map_layer1 = Linear(in_features=emb_channels, out_features=emb_channels, **init)
        if emb_dim_in > 0:
            self.map_layer0 = Linear(in_features=emb_dim_in, out_features=emb_channels, **init)
            self.map_layer1 = Linear(in_features=emb_channels, out_features=emb_channels, **init)

        if noise_channels > 0:
            self.noise_map_layer0 = Linear(in_features=noise_channels, out_features=emb_channels, **init)
            self.noise_map_layer1 = Linear(in_features=emb_channels, out_features=emb_channels, **init)

        # Encoder.
        self.enc = torch.nn.ModuleDict()
        cout = in_channels
        caux = in_channels
        for level, mult in enumerate(channel_mult):
            res = img_resolution >> level
            if level == 0:
                cin = cout
                cout = model_channels
                self.enc[f'{res}x{res}_conv'] = Conv2d(in_channels=cin, out_channels=cout, kernel=3, **init)
            else:
                self.enc[f'{res}x{res}_down'] = UNetBlock(in_channels=cout, out_channels=cout, down=True, **block_kwargs)
                if encoder_type == 'skip':
                    self.enc[f'{res}x{res}_aux_down'] = Conv2d(in_channels=caux, out_channels=caux, kernel=0, down=True, resample_filter=resample_filter)
                    self.enc[f'{res}x{res}_aux_skip'] = Conv2d(in_channels=caux, out_channels=cout, kernel=1, **init)
                if encoder_type == 'residual':
                    self.enc[f'{res}x{res}_aux_residual'] = Conv2d(in_channels=caux, out_channels=cout, kernel=3, down=True, resample_filter=resample_filter, fused_resample=True, **init)
                    caux = cout
            for idx in range(num_blocks):
                cin = cout
                cout = model_channels * mult
                attn = (res in attn_resolutions)
                self.enc[f'{res}x{res}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout, attention=attn, **block_kwargs)
        skips = [block.out_channels for name, block in self.enc.items() if 'aux' not in name]

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        for level, mult in reversed(list(enumerate(channel_mult))):
            res = img_resolution >> level
            if level == len(channel_mult) - 1:
                self.dec[f'{res}x{res}_in0'] = UNetBlock(in_channels=cout, out_channels=cout, attention=True, **block_kwargs)
                self.dec[f'{res}x{res}_in1'] = UNetBlock(in_channels=cout, out_channels=cout, **block_kwargs)
            else:
                self.dec[f'{res}x{res}_up'] = UNetBlock(in_channels=cout, out_channels=cout, up=True, **block_kwargs)
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = model_channels * mult
                attn = (idx == num_blocks and res in attn_resolutions)
                self.dec[f'{res}x{res}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout, attention=attn, **block_kwargs)
            if decoder_type == 'skip' or level == 0:
                if decoder_type == 'skip' and level < len(channel_mult) - 1:
                    self.dec[f'{res}x{res}_aux_up'] = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=0, up=True, resample_filter=resample_filter)
                self.dec[f'{res}x{res}_aux_norm'] = GroupNorm(num_channels=cout, eps=1e-6)
                self.dec[f'{res}x{res}_aux_conv'] = Conv2d(in_channels=cout, out_channels=out_channels, kernel=3, init_weight=0.2, **init)# init_zero)

    def forward(self, x, film_camera_emb=None, N_views_xa=1):

        emb = None

        if film_camera_emb is not None:
            if self.emb_dim_in != 1:
                film_camera_emb = film_camera_emb.reshape(
                    film_camera_emb.shape[0], 2, -1).flip(1).reshape(*film_camera_emb.shape) # swap sin/cos
            film_camera_emb = silu(self.map_layer0(film_camera_emb))
            film_camera_emb = silu(self.map_layer1(film_camera_emb))
            emb = film_camera_emb

        # Encoder.
        skips = []
        aux = x
        for name, block in self.enc.items():
            if 'aux_down' in name:
                aux = block(aux, N_views_xa)
            elif 'aux_skip' in name:
                x = skips[-1] = x + block(aux, N_views_xa)
            elif 'aux_residual' in name:
                x = skips[-1] = aux = (x + block(aux, N_views_xa)) / np.sqrt(2)
            else:
                x = block(x, emb=emb, N_views_xa=N_views_xa) if isinstance(block, UNetBlock) \
                    else block(x, N_views_xa=N_views_xa)
                skips.append(x)

        # Decoder.
        aux = None
        tmp = None
        for name, block in self.dec.items():
            if 'aux_up' in name:
                aux = block(aux, N_views_xa)
            elif 'aux_norm' in name:
                tmp = block(x, N_views_xa)
            elif 'aux_conv' in name:
                tmp = block(silu(tmp), N_views_xa)
                aux = tmp if aux is None else tmp + aux
            else:
                if x.shape[1] != block.in_channels:
                    # skip connection is pixel-aligned which is good for
                    # foreground features
                    # but it's not good for gradient flow and background features
                    x = torch.cat([x, skips.pop()], dim=1)
                x = block(x, emb=emb, N_views_xa=N_views_xa)
        return aux

# ================== End of implementation taken from EDM ===============
# NVIDIA copyright does not apply to the code below this line
class SingleImageSongUNetPredictor(nn.Module):
    def __init__(self, cfg, out_channels, bias, scale):
        super(SingleImageSongUNetPredictor, self).__init__()
        self.out_channels = out_channels # Note: out_channels here is already K-multiplied
        self.cfg = cfg
        if cfg.cam_embd.embedding is None:
            in_channels = 1
            emb_dim_in = 0
        else:
            in_channels = 1
            emb_dim_in = 6 * cfg.cam_embd.dimension

        self.encoder = SongUNet(cfg.data.training_resolution,
                                in_channels,
                                sum(out_channels), # Total output channels = sum(base_dims) * K
                                model_channels=cfg.model.base_dim,
                                num_blocks=cfg.model.num_blocks,
                                emb_dim_in=emb_dim_in,
                                channel_mult_noise=0,
                                attn_resolutions=cfg.model.attention_resolutions)

        # The final Conv2d maps features to the required K-multiplied output channels
        self.out = nn.Conv2d(in_channels=sum(out_channels),
                                 out_channels=sum(out_channels),
                                 kernel_size=1)

        # Initialize weights and biases for the output layer
        # Note: bias and scale lists should correspond to the *base* parameter types,
        # the initialization needs to handle the K repetition implicitly.
        num_gaussians_per_pixel = getattr(cfg.model, "num_gaussians", 1)
        base_split_dims = [1, 2, 1, 3, 4] # Base channels for depth, offset, density, scale, rot
        current_channel_idx = 0
        for i, base_dim in enumerate(base_split_dims):
            num_channels_for_param = base_dim * num_gaussians_per_pixel
            b = bias[i] # Get bias for this parameter type
            s = scale[i] # Get scale for this parameter type

            # Apply initialization across all K*base_dim channels for this parameter
            nn.init.xavier_uniform_(
                self.out.weight[current_channel_idx : current_channel_idx + num_channels_for_param, :, :, :],
                s)
            nn.init.constant_(
                self.out.bias[current_channel_idx : current_channel_idx + num_channels_for_param],
                b)
            current_channel_idx += num_channels_for_param

    def forward(self, x, film_camera_emb=None, N_views_xa=1):
        x = self.encoder(x,
                         film_camera_emb=film_camera_emb,
                         N_views_xa=N_views_xa)
        return self.out(x)

def networkCallBack(cfg, name, out_channels, **kwargs):
    if name == "SingleUNet":
        # Pass the K-multiplied out_channels directly
        return SingleImageSongUNetPredictor(cfg, out_channels, **kwargs)
    else:
        raise NotImplementedError



def generate_pixel_coords_centered(H, W, device='cpu'):
    """
    生成形状 (H, W, 2) 的网格坐标，使得:
      pixel_coords[H//2, W//2] ~= (0, 0)
    即图像中心落在 (0,0)。
    """
    # 用 linspace / arange 来构造居中坐标
    # 注意若想更精确，可用 (W-1)/2.0, (H-1)/2.0
    half_w = (W - 1) / 2.0
    half_h = (H - 1) / 2.0

    # u_range: 大小为 W, 范围 roughly [-half_w, +half_w]
    u_range = torch.linspace(-half_w, half_w, W, device=device)
    # v_range: 大小为 H, 范围 roughly [-half_h, +half_h]
    v_range = torch.linspace(-half_h, half_h, H, device=device)

    # meshgrid, indexing='ij' => v in dim0, u in dim1
    vv, uu = torch.meshgrid(v_range, u_range, indexing='ij')
    # pixel_coords[y,x, 0] = u, pixel_coords[y,x, 1] = v
    # 这里 (y,x) 只是数组索引，并不代表真实 x->horizontal, y->vertical
    pixel_coords = torch.stack([uu, vv], dim=-1)  # (H, W, 2)
    return pixel_coords
class GaussianSplatPredictor(nn.Module):
    """
    针对 CBCT 重建的Predictor (修改后支持多高斯)。
    - 输入: (B, N_view, C, H, W) 的投影图像 + (可选)相机/角度信息
    - 输出: 一个 dict，包含高斯参数 (xyz, scaling, rotation, density, offset)
            每个参数的形状类似 (B, N_view * H * W * K, D)
    """
    def __init__(self, cfg):
        """
        Args:
            cfg: 配置字典/对象, 需包含 cfg.model.num_gaussians (默认为1)
        """
        super(GaussianSplatPredictor, self).__init__()
        self.cfg = cfg
        self.num_gaussians = getattr(cfg.model, "num_gaussians", 1) # 获取每个像素的高斯数 K
        print(f"Initializing GaussianSplatPredictor with {self.num_gaussians} Gaussians per pixel.")

        self.emb_dim_in = 6 * cfg.cam_embd.dimension if cfg.cam_embd.embedding is not None else 0
        self.angle_embed_dim = cfg.cam_embd.dimension
        # 使用与第一版代码一致的 PositionalEmbedding
        self.cam_embedding_map = PositionalEmbedding(self.cfg.cam_embd.dimension)
        if self.emb_dim_in > 0:
            # 保持线性层投影
            self.angle_proj = nn.Linear(self.angle_embed_dim, self.emb_dim_in) # 修正：输入维度应为 pos emb 输出维度

        # ------- 拆分通道 + 初始化参数 (已适配多高斯) -------
        split_dims, scale_inits, bias_inits = self.get_splits_and_inits()
        self.network = networkCallBack(
            cfg,
            cfg.model.name,
            out_channels=split_dims, # 传递 K-multiplied 的通道数
            scale=scale_inits,      # 传递 base scale/bias
            bias=bias_inits
        )

        # ------- 各种激活函数 (保持不变) --------
        self.depth_act     = nn.Sigmoid()
        self.density_act   = nn.Softplus(beta=1.0)
        self.scaling_act = lambda x: F.softplus(x, beta=1.0) + 1e-3
        self.rotation_act  = lambda x: nn.functional.normalize(x, dim=1) # 应用在 channel dim (dim=1)
        self.offset_act    = lambda x: torch.tanh(x) * self.cfg.model.xyz_range
    def get_splits_and_inits(self):
        """
        根据 cfg.model.num_gaussians (K) 计算总输出通道数。
        返回 K-multiplied split_dimensions 和 base scale/bias.
        """
        K = self.num_gaussians
        base_split_dims = [1, 2, 1, 3, 4]  # depth(1), offset(2), density(1), scaling(3), rotation(4)
        # 总输出通道 = K * sum(base_split_dims)
        split_dimensions_k_multiplied = [d * K for d in base_split_dims]

        # scale 和 bias 保持 base 值，初始化时会正确应用
        scale_inits = [
            self.cfg.model.depth_scale,
            self.cfg.model.xyz_scale,
            self.cfg.model.density_scale,
            self.cfg.model.scale_scale,
            1.0
        ]
        bias_inits = [
            self.cfg.model.depth_bias,
            self.cfg.model.xyz_bias,
            self.cfg.model.density_bias,
            np.log(self.cfg.model.scale_bias), # log before exp()
            0.0
        ]
        return split_dimensions_k_multiplied, scale_inits, bias_inits

    def flatten_multi_gaussians(self, x, base_channels):
        """
        将 tensor x 从形状 (B_eff, C, H, W) 转换为 (B_eff, H*W*K, base_channels)，其中:
        - C = base_channels * K, K 为每个像素的高斯个数 (self.num_gaussians)
        - B_eff = B * N_view
        """
        B_eff, C, H, W = x.shape
        K = self.num_gaussians
        if C != base_channels * K:
             raise ValueError(f"Input channel dimension {C} does not match base_channels ({base_channels}) * num_gaussians ({K})")

        # Reshape and permute: (B_eff, K, base_C, H, W) -> (B_eff, H, W, K, base_C)
        x = x.view(B_eff, K, base_channels, H, W)
        x = x.permute(0, 3, 4, 1, 2).contiguous()
        # Flatten: (B_eff, H*W*K, base_C)
        x = x.view(B_eff, H * W * K, base_channels)
        return x

    def make_contiguous(self, tensor_dict):
        return {k: v.contiguous() for k, v in tensor_dict.items()}

    def multi_view_union(self, tensor_dict, B, N_view):
        # 注意：现在 N = H * W * K
        for k, v in tensor_dict.items():
            # v: (B*N_view, N, ...) = (B*N_view, H*W*K, D)
            N = v.shape[1] # N = H*W*K
            D = v.shape[2:]
            # Reshape to (B, N_view, N, D...)
            v = v.view(B, N_view, N, *D)
            # Reshape to (B, N_view * N, D...) = (B, N_view*H*W*K, D)
            v = v.view(B, N_view * N, *D)
            tensor_dict[k] = v
        return tensor_dict

    def transform_rotations(self, rotations, source_cv2wT_quat):
        """
        rotations: (B*N_view, N, 4), where N = H*W*K
        source_cv2wT_quat: (B*N_view, 4)
        """
        # 广播 source_cv2wT_quat
        Mq = source_cv2wT_quat.unsqueeze(1).expand_as(rotations) # (B*N_view, N, 4)
        # 假设 quaternion_raw_multiply 支持批处理
        out_quat = quaternion_raw_multiply(Mq, rotations)
        return out_quat

    def pack_camera_params(self, camera_params_dicts, scanner_cfg_list, device='cpu'):
        num_cameras = len(camera_params_dicts)
        num_scanners = len(scanner_cfg_list)

        # 确保列表长度一致
        if num_cameras != num_scanners:
             raise ValueError(f"Mismatch: len(camera_params_dicts)={num_cameras} vs len(scanner_cfg_list)={num_scanners}")
        if num_cameras == 0:
             raise ValueError("Received empty camera_params_dicts or scanner_cfg_list")


        # 从字典中提取 view_to_world
        v2w_list = [cp["view_to_world"].to(device) for cp in camera_params_dicts]

        DSO_list, DSD_list, du_list, dv_list, offU_list, offV_list = [], [], [], [], [], []
        for j in range(num_cameras):
            # 直接使用传入的 scanner_cfg
            scanner = scanner_cfg_list[j]
            if scanner is None:
                 raise ValueError(f"Scanner_cfg at index {j} is None.")

            DSO_list.append(scanner["DSO"])
            if "DSD" not in scanner:
                 raise KeyError(f"Key 'DSD' not found in scanner_cfg at index {j}")
            DSD_list.append(scanner["DSD"])
            du_val, dv_val = scanner["dDetector"]
            offU_val, offV_val = scanner["offDetector"]
            du_list.append(du_val)
            dv_list.append(dv_val)
            offU_list.append(offU_val)
            offV_list.append(offV_val)

        DSO_t   = torch.tensor(DSO_list,   dtype=torch.float32, device=device)
        DSD_t   = torch.tensor(DSD_list,   dtype=torch.float32, device=device)
        du_t    = torch.tensor(du_list,    dtype=torch.float32, device=device)
        dv_t    = torch.tensor(dv_list,    dtype=torch.float32, device=device)
        offU_t  = torch.tensor(offU_list,  dtype=torch.float32, device=device)
        offV_t  = torch.tensor(offV_list,  dtype=torch.float32, device=device)
        v2w_t   = torch.stack(v2w_list, dim=0) # (B*N_view, 4, 4)

        # 验证形状是否为 (num_cameras)
        expected_len = num_cameras
        if not all(t.shape[0] == expected_len for t in [DSO_t, DSD_t, du_t, dv_t, offU_t, offV_t, v2w_t]):
             # 获取实际长度用于错误消息
             actual_shapes = { "DSO": DSO_t.shape, "DSD": DSD_t.shape, "du": du_t.shape, "dv": dv_t.shape,
                              "offU": offU_t.shape, "offV": offV_t.shape, "v2w": v2w_t.shape }
             raise ValueError(f"Shape mismatch in packed camera params. Expected length {expected_len}. Got shapes: {actual_shapes}")


        return DSO_t, DSD_t, du_t, dv_t, offU_t, offV_t, v2w_t

    def backproject_cbct_vec(
        self,
        depth_map,       # (B*N, H, W, K)
        offset_map,      # (B*N, H, W, 2*K)
        DSO_t,           # (B*N,)
        DSD_t,           # (B*N,)
        du_t, dv_t,      # (B*N,)
        offU_t, offV_t,  # (B*N,)
        v2w_t,           # (B*N, 4, 4) - Camera-to-World transform matrix
        pixel_coords,    # (H, W, 2)
        chunk_size=None
    ):
        device = depth_map.device
        BtimesN, H, W, K = depth_map.shape
        _B, _H, _W, K_times_2 = offset_map.shape

        if K_times_2 != 2 * K:
             raise ValueError(f"Offset map last dimension {K_times_2} is not 2 * Depth map last dimension {K}")
        if _B!=BtimesN or _H!=H or _W!=W:
             raise ValueError("Depth map and Offset map shape mismatch (excluding last dim)")

        # DSO  = DSO_t.view(-1, 1, 1) # 不再直接使用 DSO 计算射线方向
        DSD  = DSD_t.view(-1, 1, 1, 1) # (B*N, 1, 1, 1) for broadcasting with (B*N, H, W)
        du   = du_t.view(-1, 1, 1, 1) # (B*N, 1, 1, 1)
        dv   = dv_t.view(-1, 1, 1, 1) # (B*N, 1, 1, 1)
        offU = offU_t.view(-1, 1, 1, 1) # (B*N, 1, 1, 1)
        offV = offV_t.view(-1, 1, 1, 1) # (B*N, 1, 1, 1)

        uv = pixel_coords.to(device).unsqueeze(0) # (1, H, W, 2)
        # Expand later inside chunk processing if needed, or expand here if memory allows
        # uv = uv.expand(BtimesN, -1, -1, -1) # (B*N, H, W, 2)

        def process_chunk(h_start, h_end):
            # Adjust slicing for uv if expanded outside
            uv_chunk = uv[:, h_start:h_end, :, :].expand(BtimesN, -1, -1, -1) # (B*N, chunkH, W, 2)
            depth_chunk   = depth_map[:, h_start:h_end, :, :]     # (B*N, chunkH, W, K)
            offset_chunk  = offset_map[:, h_start:h_end, :, :]    # (B*N, chunkH, W, 2*K)

            chunkH = uv_chunk.shape[1]

            # --- 计算射线方向 (与 K 无关) ---
            # Ensure broadcasting works correctly: (B*N, chunkH, W) shapes needed
            uv_x = uv_chunk[..., 0] - offU[:,:,:chunkH,:] # Correct broadcasting shape? Check dimensions. Let's redo reshape.
            # Reshape geometry params for broadcasting with chunk: (B*N, 1, 1)
            DSD_c  = DSD_t.view(-1, 1, 1)
            du_c   = du_t.view(-1, 1, 1)
            dv_c   = dv_t.view(-1, 1, 1)
            offU_c = offU_t.view(-1, 1, 1)
            offV_c = offV_t.view(-1, 1, 1)

            uv_x = uv_chunk[..., 0] - offU_c # (B*N, chunkH, W)
            uv_y = uv_chunk[..., 1] - offV_c # (B*N, chunkH, W)
            x_d = uv_x * du_c # (B*N, chunkH, W)
            y_d = uv_y * dv_c # (B*N, chunkH, W)

            z_d = DSD_c.expand_as(x_d) # <-- 修改后的代码：使用 DSD (B*N, chunkH, W)
            ray_dir = torch.stack([x_d, y_d, z_d], dim=-1) # (B*N, chunkH, W, 3)
            ray_dir_norm = torch.norm(ray_dir, dim=-1, keepdim=True) + 1e-8 # (B*N, chunkH, W, 1)
            ray_dir_unit = ray_dir / ray_dir_norm # (B*N, chunkH, W, 3)

            # --- 计算 K 个高斯的原始点 (沿射线) ---
            # depth_chunk: (B*N, chunkH, W, K)
            # ray_dir_unit: (B*N, chunkH, W, 3)
            # Expand ray_dir_unit for multiplication with depth
            ray_dir_unit_exp = ray_dir_unit.unsqueeze(3) # (B*N, chunkH, W, 1, 3)
            depth_chunk_exp = depth_chunk.unsqueeze(-1) # (B*N, chunkH, W, K, 1)
            raw_point = depth_chunk_exp * ray_dir_unit_exp # (B*N, chunkH, W, K, 3)

            # --- 计算正交基 p, q (与 K 无关) ---
            # ray_dir_unit is (B*N, chunkH, W, 3)
            up = torch.tensor([0, 1, 0], device=device, dtype=ray_dir_unit.dtype)
            up = up.view(1,1,1,3).expand_as(ray_dir_unit) # (B*N, chunkH, W, 3)
            dot = (ray_dir_unit * up).sum(dim=-1, keepdim=True) # (B*N, chunkH, W, 1)
            mask = (torch.abs(dot) > 0.995).expand_as(up)
            up_alt = torch.tensor([1, 0, 0], device=device, dtype=ray_dir_unit.dtype)
            up_alt = up_alt.view(1,1,1,3).expand_as(up)
            up = torch.where(mask, up_alt, up) # (B*N, chunkH, W, 3)
            p = torch.cross(up, ray_dir_unit, dim=-1) # (B*N, chunkH, W, 3)
            p = p / (torch.norm(p, dim=-1, keepdim=True) + 1e-8) # (B*N, chunkH, W, 3)
            q = torch.cross(ray_dir_unit, p, dim=-1) # (B*N, chunkH, W, 3)
            q = q / (torch.norm(q, dim=-1, keepdim=True) + 1e-8) # (B*N, chunkH, W, 3)


            # --- 处理 K 个偏移量 ---
            # offset_chunk: (B*N, chunkH, W, 2*K)
            offset_chunk_res = offset_chunk.view(BtimesN, chunkH, W, K, 2) # (B*N, chunkH, W, K, 2)
            offset_x = offset_chunk_res[..., 0:1] # (B*N, chunkH, W, K, 1)
            offset_y = offset_chunk_res[..., 1:2] # (B*N, chunkH, W, K, 1)

            # --- 扩展 p, q 以匹配 K ---
            p_exp = p.unsqueeze(3) # (B*N, chunkH, W, 1, 3)
            q_exp = q.unsqueeze(3) # (B*N, chunkH, W, 1, 3)

            # --- 计算最终相机坐标系下的点 (考虑 K 个高斯) ---
            # raw_point: (B*N, chunkH, W, K, 3)
            # offset_x: (B*N, chunkH, W, K, 1)
            # p_exp:    (B*N, chunkH, W, 1, 3)
            pos_cam = raw_point + offset_x * p_exp + offset_y * q_exp # (B*N, chunkH, W, K, 3)

            # --- 变换到世界坐标系 ---
            pos_cam_flat = pos_cam.view(BtimesN, chunkH * W * K, 3)
            ones = torch.ones(BtimesN, chunkH * W * K, 1, dtype=pos_cam.dtype, device=device)
            pos_cam_h = torch.cat([pos_cam_flat, ones], dim=-1) # (B*N, chunkH*W*K, 4)

            # 恢复原始的世界坐标变换计算方式 (假设 v2w_t 是 [[R,0],[t,1]] 或类似形式，且此操作正确)
            # P_world_h = torch.bmm(pos_cam_h, v2w_t.transpose(1, 2)) # <-- 之前的修正
            pos_world_h = torch.bmm(
                v2w_t.transpose(1, 2),     # Shape (B*N, 4, 4)
                pos_cam_h.transpose(1, 2)  # Shape (B*N, 4, chunkH*W*K)
            ).transpose(1, 2)              # Output Shape (B*N, chunkH*W*K, 4) # <-- 恢复原始代码

            # Dehomogenize
            pos_world_chunk = pos_world_h[..., :3] / (pos_world_h[..., 3:] + 1e-8) # (B*N, chunkH*W*K, 3)

            return pos_world_chunk

        # --- 分块处理或一次性处理 ---
        if not chunk_size or chunk_size >= H:
            pos_world_all = process_chunk(0, H)
        else:
            pos_world_list = []
            start = 0
            while start < H:
                end = min(start + chunk_size, H)
                chunk_res = process_chunk(start, end)
                pos_world_list.append(chunk_res)
                start = end
            pos_world_all = torch.cat(pos_world_list, dim=1) # (B*N, H*W*K, 3)

        return pos_world_all

    def get_camera_embeddings(self, view_to_world_tensor):
        # 这个函数应该保持不变
        b, n_view = view_to_world_tensor.shape[:2]
        if self.cfg.cam_embd.embedding == "pose":
            cam_embedding = torch.cat([view_to_world_tensor[:, :, 3, :3],
                                       view_to_world_tensor[:, :, 2, :3]], dim=2) # (B, N_view, 6)
        # elif self.cfg.cam_embd.embedding == "index": # Add if needed
        #     cam_embedding = torch.arange(n_view, ... )
        else:
            raise ValueError(f"Unsupported cam_embd.embedding type: {self.cfg.cam_embd.embedding}")

        # Reshape for embedding map
        cam_embedding = rearrange(cam_embedding, 'b n_view c -> (b n_view) c')
        cam_embedding = self.cam_embedding_map(cam_embedding) # Apply positional embedding
        # Reshape back to include N_view dimension if needed by the network's FiLM layers later
        # film_camera_emb expects (B * N_view, emb_dim)
        # cam_embedding = rearrange(cam_embedding, '(b n_view) c -> b n_view c', b=b, n_view=n_view)
        return cam_embedding # Shape (B*N_view, emb_dim)

    def forward(
        self,
        x,                       # (B, N_view, C, H, W)
        source_cv2wT_quat=None,  # (B, N_view, 4)
        camera_params_dicts=None,# list[dict], len = B*N_view <--- 修改名称和类型提示
        scanner_cfg_list=None,   # list[dict], len = B*N_view <--- 添加参数
        activate_output=True
    ):
        """
        Forward pass supporting K Gaussians per pixel.
        Accepts camera_params as a list of dicts and scanner_cfg as a separate list.
        """
        B, N_views, C_in, H, W = x.shape
        K = self.num_gaussians

        # 输入验证
        if camera_params_dicts is None or not isinstance(camera_params_dicts, list):
             raise ValueError("camera_params_dicts must be a list of dictionaries.")
        if scanner_cfg_list is None or not isinstance(scanner_cfg_list, list):
             raise ValueError("scanner_cfg_list must be provided as a list of dictionaries.")
        if len(camera_params_dicts) != B * N_views:
             raise ValueError(f"Length of camera_params_dicts ({len(camera_params_dicts)}) does not match B*N_views ({B*N_views})")
        if len(scanner_cfg_list) != B * N_views:
             # 注意：overfit_oneproj.py 中 scanner_cfg_list 的长度是 B*N_view，不是 B
             # for i in range(batch_size):
             #     for j in range(cfg.data.input_images): # input_images is N_views
             #         scanner_cfg_list.append(data["scanner_cfg"][i])
             # 所以这里的长度检查应该是 B * N_views
             raise ValueError(f"Length of scanner_cfg_list ({len(scanner_cfg_list)}) does not match B*N_views ({B*N_views})")


        # Reshape input: (B, N_view, C, H, W) -> (B*N_view, C, H, W)
        x_flat = x.reshape(B * N_views, C_in, H, W).contiguous()

        # Generate pixel coordinates (once per size)
        if not hasattr(self, 'pixel_coords') or self.pixel_coords.shape[0] != H or self.pixel_coords.shape[1] != W:
             self.pixel_coords = generate_pixel_coords_centered(H, W, device=x.device)

        # --- Camera Embedding ---
        film_camera_emb = None
        if self.cfg.cam_embd.embedding is not None:
            # Prepare view_to_world tensor from the list of dictionaries
            view_to_world_list = [cp["view_to_world"].to(x.device) for cp in camera_params_dicts]
            view_to_world_tensor = torch.stack(view_to_world_list, dim=0) # (B*N_view, 4, 4)

            # Reshape for get_camera_embeddings if needed (expects B, N_view, ...)
            view_to_world_tensor_reshaped = view_to_world_tensor.view(B, N_views, 4, 4)
            cam_embedding = self.get_camera_embeddings(view_to_world_tensor_reshaped) # (B*N_view, emb_dim)

            if cam_embedding is not None:
                film_camera_emb = cam_embedding # Already shape (B*N_view, emb_dim)

        # --- Network Forward Pass ---
        net_out = self.network(x_flat, film_camera_emb=film_camera_emb, N_views_xa=N_views)
        # net_out shape: (B*N_view, TotalChannels, H, W), where TotalChannels = sum(base_dims)*K

        # --- Split Output Channels ---
        base_split_dims = [1, 2, 1, 3, 4]
        split_dims_k_multiplied = [d * K for d in base_split_dims]
        if net_out.shape[1] != sum(split_dims_k_multiplied):
            raise ValueError(f"Network output channels {net_out.shape[1]} != expected {sum(split_dims_k_multiplied)}")

        depth_raw, offset_raw, density_raw, scaling_raw, rotation_raw = net_out.split(split_dims_k_multiplied, dim=1)
        # Shapes e.g., depth_raw: (B*N, K, H, W), offset_raw: (B*N, 2*K, H, W), ...

        # --- Apply Activations ---
        if activate_output:
            depth_acted    = self.depth_act(depth_raw)    # (B*N, K, H, W) -> (B*N, H, W, K) after permute
            density_acted  = self.density_act(density_raw) # (B*N, K, H, W) -> (B*N, H, W, K)
            scaling_acted  = self.scaling_act(scaling_raw) # (B*N, 3*K, H, W) -> (B*N, H, W, 3K)
            rotation_acted = self.rotation_act(rotation_raw)# (B*N, 4*K, H, W) -> (B*N, H, W, 4K)
            offset_acted   = self.offset_act(offset_raw)  # (B*N, 2*K, H, W) -> (B*N, H, W, 2K)
        else:
            depth_acted, offset_acted, density_acted, scaling_acted, rotation_acted = \
                depth_raw, offset_raw, density_raw, scaling_raw, rotation_raw

        # Apply Z-range scaling if configured
        if getattr(self.cfg.model, "use_zrange", False):
            z_near = getattr(self.cfg.model, "z_near", 0.01)
            z_far  = getattr(self.cfg.model, "z_far", 100)
            if z_far <= z_near:
                raise ValueError(f"cfg.model.z_near={z_near} must be < z_far={z_far}")
            depth_acted = z_near + (z_far - z_near) * depth_acted

        # Apply isotropic scaling if configured
        if self.cfg.model.isotropic:
            scaling_acted_reshaped = scaling_acted.view(B * N_views, K, 3, H, W)
            isotropic_scale = scaling_acted_reshaped[:, :, 0:1, :, :]
            scaling_acted_iso = isotropic_scale.repeat(1, 1, 3, 1, 1)
            scaling_acted = scaling_acted_iso.view(B * N_views, 3 * K, H, W)

        # --- Backprojection ---
        # Pass the dictionary list and the scanner config list
        DSO_t, DSD_t, du_t, dv_t, offU_t, offV_t, v2w_t = self.pack_camera_params(
            camera_params_dicts, scanner_cfg_list, device=x.device
        )

        # Permute outputs for backprojection function
        depth_map   = depth_acted.permute(0, 2, 3, 1).contiguous()
        offset_map  = offset_acted.permute(0, 2, 3, 1).contiguous()
        density_map = density_acted.permute(0, 2, 3, 1).contiguous() # density needs permute too
        scaling_map = scaling_acted.permute(0, 2, 3, 1).contiguous() # scaling needs permute too
        rotation_map = rotation_acted.permute(0, 2, 3, 1).contiguous() # rotation needs permute too


        pos_world = self.backproject_cbct_vec(
            depth_map=depth_map,     # (B*N, H, W, K)
            offset_map=offset_map,   # (B*N, H, W, 2*K)
            DSO_t=DSO_t,             # (B*N,)
            DSD_t=DSD_t,             # (B*N,)
            du_t=du_t, dv_t=dv_t,     # (B*N,)
            offU_t=offU_t, offV_t=offV_t, # (B*N,)
            v2w_t=v2w_t,             # (B*N, 4, 4)
            pixel_coords=self.pixel_coords, # (H, W, 2)
            chunk_size=getattr(self.cfg.model, "backproj_chunk_size", None)
        )
        # pos_world shape: (B*N, H*W*K, 3)

        # --- Flatten Parameters for Output ---
        density_flat  = density_map.reshape(B*N_views, H*W*K, 1)
        scaling_flat  = scaling_map.reshape(B*N_views, H*W*K, 3)
        rotation_flat = rotation_map.reshape(B*N_views, H*W*K, 4)
        offset_flat   = offset_map.reshape(B*N_views, H*W*K, 2)

        # --- Transform Rotations to World Space ---
        if source_cv2wT_quat is not None:
             if source_cv2wT_quat.shape != (B, N_views, 4):
                  raise ValueError(f"Expected source_cv2wT_quat shape {(B, N_views, 4)}, got {source_cv2wT_quat.shape}")
             scq_flat = source_cv2wT_quat.reshape(B*N_views, 4).to(x.device)
             rotation_flat_norm = rotation_flat / (torch.norm(rotation_flat, dim=-1, keepdim=True) + 1e-8)
             rot_world = self.transform_rotations(rotation_flat_norm, scq_flat)
             rot_world = rot_world / (torch.norm(rot_world, dim=-1, keepdim=True) + 1e-8)
        else:
            print("Warning: source_cv2wT_quat not provided. Returning rotations in camera frame.")
            rot_world = rotation_flat / (torch.norm(rotation_flat, dim=-1, keepdim=True) + 1e-8)

        # --- Assemble Output Dictionary ---
        out_dict = {
            "xyz":      pos_world,       # (B*N, H*W*K, 3)
            "rotation": rot_world,       # (B*N, H*W*K, 4)
            "density":  density_flat,    # (B*N, H*W*K, 1)
            "scaling":  scaling_flat,    # (B*N, H*W*K, 3)
            "offset":   offset_flat      # (B*N, H*W*K, 2)
        }

        # Reshape from (B*N_view, N_points, D) to (B, N_view * N_points, D)
        out_dict = self.multi_view_union(out_dict, B, N_views)

        # Ensure contiguous tensors
        out_dict = self.make_contiguous(out_dict)

        return out_dict