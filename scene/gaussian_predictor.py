# import torch
# import torch.nn as nn
# import torchvision 

# import numpy as np
# import torch
# from torch.nn.functional import silu

# from einops import rearrange, repeat

# from utils.general_utils import matrix_to_quaternion, quaternion_raw_multiply
# from utils.graphics_utils import fov2focal

# # U-Net implementation from EDM
# # Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# #
# # This work is licensed under a Creative Commons
# # Attribution-NonCommercial-ShareAlike 4.0 International License.
# # You should have received a copy of the license along with this
# # work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

# """Model architectures and preconditioning schemes used in the paper
# "Elucidating the Design Space of Diffusion-Based Generative Models"."""

# #----------------------------------------------------------------------------
# # Unified routine for initializing weights and biases.

# def weight_init(shape, mode, fan_in, fan_out):
#     if mode == 'xavier_uniform': return np.sqrt(6 / (fan_in + fan_out)) * (torch.rand(*shape) * 2 - 1)
#     if mode == 'xavier_normal':  return np.sqrt(2 / (fan_in + fan_out)) * torch.randn(*shape)
#     if mode == 'kaiming_uniform': return np.sqrt(3 / fan_in) * (torch.rand(*shape) * 2 - 1)
#     if mode == 'kaiming_normal':  return np.sqrt(1 / fan_in) * torch.randn(*shape)
#     raise ValueError(f'Invalid init mode "{mode}"')

# #----------------------------------------------------------------------------
# # Fully-connected layer.

# class Linear(torch.nn.Module):
#     def __init__(self, in_features, out_features, bias=True, init_mode='kaiming_normal', init_weight=1, init_bias=0):
#         super().__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         init_kwargs = dict(mode=init_mode, fan_in=in_features, fan_out=out_features)
#         self.weight = torch.nn.Parameter(weight_init([out_features, in_features], **init_kwargs) * init_weight)
#         self.bias = torch.nn.Parameter(weight_init([out_features], **init_kwargs) * init_bias) if bias else None

#     def forward(self, x):
#         x = x @ self.weight.to(x.dtype).t()
#         if self.bias is not None:
#             x = x.add_(self.bias.to(x.dtype))
#         return x

# #----------------------------------------------------------------------------
# # Convolutional layer with optional up/downsampling.

# class Conv2d(torch.nn.Module):
#     def __init__(self,
#         in_channels, out_channels, kernel, bias=True, up=False, down=False,
#         resample_filter=[1,1], fused_resample=False, init_mode='kaiming_normal', init_weight=1, init_bias=0,
#     ):
#         assert not (up and down)
#         super().__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.up = up
#         self.down = down
#         self.fused_resample = fused_resample
#         init_kwargs = dict(mode=init_mode, fan_in=in_channels*kernel*kernel, fan_out=out_channels*kernel*kernel)
#         self.weight = torch.nn.Parameter(weight_init([out_channels, in_channels, kernel, kernel], **init_kwargs) * init_weight) if kernel else None
#         self.bias = torch.nn.Parameter(weight_init([out_channels], **init_kwargs) * init_bias) if kernel and bias else None
#         f = torch.as_tensor(resample_filter, dtype=torch.float32)
#         f = f.ger(f).unsqueeze(0).unsqueeze(1) / f.sum().square()
#         self.register_buffer('resample_filter', f if up or down else None)

#     def forward(self, x, N_views_xa=1):
#         w = self.weight.to(x.dtype) if self.weight is not None else None
#         b = self.bias.to(x.dtype) if self.bias is not None else None
#         f = self.resample_filter.to(x.dtype) if self.resample_filter is not None else None
#         w_pad = w.shape[-1] // 2 if w is not None else 0
#         f_pad = (f.shape[-1] - 1) // 2 if f is not None else 0

#         if self.fused_resample and self.up and w is not None:
#             x = torch.nn.functional.conv_transpose2d(x, f.mul(4).tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=max(f_pad - w_pad, 0))
#             x = torch.nn.functional.conv2d(x, w, padding=max(w_pad - f_pad, 0))
#         elif self.fused_resample and self.down and w is not None:
#             x = torch.nn.functional.conv2d(x, w, padding=w_pad+f_pad)
#             x = torch.nn.functional.conv2d(x, f.tile([self.out_channels, 1, 1, 1]), groups=self.out_channels, stride=2)
#         else:
#             if self.up:
#                 x = torch.nn.functional.conv_transpose2d(x, f.mul(4).tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=f_pad)
#             if self.down:
#                 x = torch.nn.functional.conv2d(x, f.tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=f_pad)
#             if w is not None:
#                 x = torch.nn.functional.conv2d(x, w, padding=w_pad)
#         if b is not None:
#             x = x.add_(b.reshape(1, -1, 1, 1))
#         return x

# #----------------------------------------------------------------------------
# # Group normalization.

# class GroupNorm(torch.nn.Module):
#     def __init__(self, num_channels, num_groups=32, min_channels_per_group=4, eps=1e-5):
#         super().__init__()
#         self.num_groups = min(num_groups, num_channels // min_channels_per_group)
#         self.eps = eps
#         self.weight = torch.nn.Parameter(torch.ones(num_channels))
#         self.bias = torch.nn.Parameter(torch.zeros(num_channels))

#     def forward(self, x, N_views_xa=1):
#         x = torch.nn.functional.group_norm(x, num_groups=self.num_groups, weight=self.weight.to(x.dtype), bias=self.bias.to(x.dtype), eps=self.eps)
#         return x.to(memory_format=torch.channels_last)

# #----------------------------------------------------------------------------
# # Attention weight computation, i.e., softmax(Q^T * K).
# # Performs all computation using FP32, but uses the original datatype for
# # inputs/outputs/gradients to conserve memory.

# class AttentionOp(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, q, k):
#         w = torch.einsum('ncq,nck->nqk', q.to(torch.float32), (k / np.sqrt(k.shape[1])).to(torch.float32)).softmax(dim=2).to(q.dtype)
#         ctx.save_for_backward(q, k, w)
#         return w

#     @staticmethod
#     def backward(ctx, dw):
#         q, k, w = ctx.saved_tensors
#         db = torch._softmax_backward_data(grad_output=dw.to(torch.float32), output=w.to(torch.float32), dim=2, input_dtype=torch.float32)
#         dq = torch.einsum('nck,nqk->ncq', k.to(torch.float32), db).to(q.dtype) / np.sqrt(k.shape[1])
#         dk = torch.einsum('ncq,nqk->nck', q.to(torch.float32), db).to(k.dtype) / np.sqrt(k.shape[1])
#         return dq, dk
 
# #----------------------------------------------------------------------------
# # Timestep embedding used in the DDPM++ and ADM architectures.

# # class PositionalEmbedding(torch.nn.Module):
# #     def __init__(self, num_channels, max_positions=10000, endpoint=False):
# #         super().__init__()
# #         self.num_channels = num_channels
# #         self.max_positions = max_positions
# #         self.endpoint = endpoint

# #     def forward(self, x):
# #         b, c = x.shape
# #         x = rearrange(x, 'b c -> (b c)')
# #         freqs = torch.arange(start=0, end=self.num_channels//2, dtype=torch.float32, device=x.device)
# #         freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
# #         freqs = (1 / self.max_positions) ** freqs
# #         x = x.ger(freqs.to(x.dtype))
# #         x = torch.cat([x.cos(), x.sin()], dim=1)
# #         x = rearrange(x, '(b c) emb_ch -> b (c emb_ch)', b=b)
# #         return x
# class PositionalEmbedding(nn.Module):
#     """
#     示例性地用来将标量 (如角度) 转换为一个高维向量嵌入，
#     模仿 DDPM++/ADM 的时间步编码方式。
#     """
#     def __init__(self, num_channels, max_positions=10000, endpoint=False):
#         super().__init__()
#         self.num_channels = num_channels
#         self.max_positions = max_positions
#         self.endpoint = endpoint

#     def forward(self, x):
#         """
#         x: (B, 1) 或 (B, C) 的角度/标量张量，这里假设只有 1 通道即角度。
#            如果需要更多几何信息，可在外部先拼接成多通道后传进来。
#         返回: (B, num_channels) 的嵌入。
#         """
#         # x: (B, 1)
#         b, c = x.shape
#         if c != 1:
#             # 如果你想一次性编码多个参数，可改成 c>1
#             pass

#         # 按照 DDPM++ 的思路，先将 x 扩展到 (B*C)
#         x = rearrange(x, 'b c -> (b c)')  # (B, )
        
#         # 生成频率 freqs
#         freqs = torch.arange(
#             start=0, 
#             end=self.num_channels // 2, 
#             dtype=torch.float32, 
#             device=x.device
#         )
#         denom = (self.num_channels // 2 - (1 if self.endpoint else 0))
#         freqs = freqs / denom
#         freqs = (1 / self.max_positions) ** freqs

#         # x.ger(...) 等价于外积： (B,) x (embed_dim,) -> (B, embed_dim)
#         x = x.unsqueeze(-1) * freqs  # 先做广播乘法
#         x_cos = torch.cos(x)
#         x_sin = torch.sin(x)
#         x = torch.cat([x_cos, x_sin], dim=-1)  # (B, num_channels)

#         return x
# #----------------------------------------------------------------------------
# # Timestep embedding used in the NCSN++ architecture.

# class FourierEmbedding(torch.nn.Module):
#     def __init__(self, num_channels, scale=16):
#         super().__init__()
#         self.register_buffer('freqs', torch.randn(num_channels // 2) * scale)

#     def forward(self, x):
#         b, c = x.shape
#         x = rearrange(x, 'b c -> (b c)')
#         x = x.ger((2 * np.pi * self.freqs).to(x.dtype))
#         x = torch.cat([x.cos(), x.sin()], dim=1)
#         x = rearrange(x, '(b c) emb_ch -> b (c emb_ch)', b=b)
#         return x  

# class CrossAttentionBlock(torch.nn.Module):
#     def __init__(self, num_channels, num_heads = 1, eps=1e-5):
#         super().__init__()

#         self.num_heads = 1
#         init_attn = dict(init_mode='xavier_uniform', init_weight=np.sqrt(0.2))
#         init_zero = dict(init_mode='xavier_uniform', init_weight=1e-5)

#         self.norm = GroupNorm(num_channels=num_channels, eps=eps)

#         self.q_proj = Conv2d(in_channels=num_channels, out_channels=num_channels, kernel=1, **init_attn)
#         self.kv_proj = Conv2d(in_channels=num_channels, out_channels=num_channels*2, kernel=1, **init_attn)

#         self.out_proj = Conv2d(in_channels=num_channels, out_channels=num_channels, kernel=3, **init_zero)

#     def forward(self, q, kv):
#         q_proj = self.q_proj(self.norm(q)).reshape(q.shape[0] * self.num_heads, q.shape[1] // self.num_heads, -1)
#         k_proj, v_proj = self.kv_proj(self.norm(kv)).reshape(kv.shape[0] * self.num_heads, 
#                                                    kv.shape[1] // self.num_heads, 2, -1).unbind(2)
#         w = AttentionOp.apply(q_proj, k_proj)
#         a = torch.einsum('nqk,nck->ncq', w, v_proj)
#         x = self.out_proj(a.reshape(*q.shape)).add_(q)

#         return x

# #----------------------------------------------------------------------------
# # Unified U-Net block with optional up/downsampling and self-attention.
# # Represents the union of all features employed by the DDPM++, NCSN++, and
# # ADM architectures.

# class UNetBlock(torch.nn.Module):
#     def __init__(self,
#         in_channels, out_channels, emb_channels, up=False, down=False, attention=False,
#         num_heads=None, channels_per_head=64, dropout=0, skip_scale=1, eps=1e-5,
#         resample_filter=[1,1], resample_proj=False, adaptive_scale=True,
#         init=dict(), init_zero=dict(init_weight=0), init_attn=None,
#     ):
#         super().__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         if emb_channels is not None:
#             self.affine = Linear(in_features=emb_channels, out_features=out_channels*(2 if adaptive_scale else 1), **init)
#         self.num_heads = 0 if not attention else num_heads if num_heads is not None else out_channels // channels_per_head
#         self.dropout = dropout
#         self.skip_scale = skip_scale
#         self.adaptive_scale = adaptive_scale

#         self.norm0 = GroupNorm(num_channels=in_channels, eps=eps)
#         self.conv0 = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel=3, up=up, down=down, resample_filter=resample_filter, **init)
#         self.norm1 = GroupNorm(num_channels=out_channels, eps=eps)
#         self.conv1 = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=3, **init_zero)

#         self.skip = None
#         if out_channels != in_channels or up or down:
#             kernel = 1 if resample_proj or out_channels!= in_channels else 0
#             self.skip = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel=kernel, up=up, down=down, resample_filter=resample_filter, **init)

#         if self.num_heads:
#             self.norm2 = GroupNorm(num_channels=out_channels, eps=eps)
#             self.qkv = Conv2d(in_channels=out_channels, out_channels=out_channels*3, kernel=1, **(init_attn if init_attn is not None else init))
#             self.proj = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=1, **init_zero)

#     def forward(self, x, emb=None, N_views_xa=1):
#         orig = x
#         x = self.conv0(silu(self.norm0(x)))

#         if emb is not None:
#             params = self.affine(emb).unsqueeze(2).unsqueeze(3).to(x.dtype)
#             if self.adaptive_scale:
#                 scale, shift = params.chunk(chunks=2, dim=1)
#                 x = silu(torch.addcmul(shift, self.norm1(x), scale + 1))
#             else:
#                 x = silu(self.norm1(x.add_(params)))

#         x = silu(self.norm1(x))

#         x = self.conv1(torch.nn.functional.dropout(x, p=self.dropout, training=self.training))
#         x = x.add_(self.skip(orig) if self.skip is not None else orig)
#         x = x * self.skip_scale

#         if self.num_heads:
#             if N_views_xa != 1:
#                 B, C, H, W = x.shape
#                 # (B, C, H, W) -> (B/N, N, C, H, W) -> (B/N, N, H, W, C)
#                 x = x.reshape(B // N_views_xa, N_views_xa, *x.shape[1:]).permute(0, 1, 3, 4, 2)
#                 # (B/N, N, H, W, C) -> (B/N, N*H, W, C) -> (B/N, C, N*H, W)
#                 x = x.reshape(B // N_views_xa, N_views_xa * x.shape[2], *x.shape[3:]).permute(0, 3, 1, 2)
#             q, k, v = self.qkv(self.norm2(x)).reshape(x.shape[0] * self.num_heads, x.shape[1] // self.num_heads, 3, -1).unbind(2)
#             w = AttentionOp.apply(q, k)
#             a = torch.einsum('nqk,nck->ncq', w, v)
#             x = self.proj(a.reshape(*x.shape)).add_(x)
#             x = x * self.skip_scale
#             if N_views_xa != 1:
#                 # (B/N, C, N*H, W) -> (B/N, N*H, W, C)
#                 x = x.permute(0, 2, 3, 1)
#                 # (B/N, N*H, W, C) -> (B/N, N, H, W, C) -> (B/N, N, C, H, W)
#                 x = x.reshape(B // N_views_xa, N_views_xa, H, W, C).permute(0, 1, 4, 2, 3)
#                 # (B/N, N, C, H, W) -> # (B, C, H, W) 
#                 x = x.reshape(B, C, H, W)
#         return x


# #----------------------------------------------------------------------------
# # Reimplementation of the DDPM++ and NCSN++ architectures from the paper
# # "Score-Based Generative Modeling through Stochastic Differential
# # Equations". Equivalent to the original implementation by Song et al.,
# # available at https://github.com/yang-song/score_sde_pytorch
# # taken from EDM repository https://github.com/NVlabs/edm/blob/main/training/networks.py#L372

# class SongUNet(nn.Module):
#     def __init__(self,
#         img_resolution,                     # Image resolution at input/output.
#         in_channels,                        # Number of color channels at input.
#         out_channels,                       # Number of color channels at output.
#         emb_dim_in           = 0,            # Input embedding dim.
#         augment_dim         = 0,            # Augmentation label dimensionality, 0 = no augmentation.

#         model_channels      = 128,          # Base multiplier for the number of channels.
#         channel_mult        = [1,2,2,2],    # Per-resolution multipliers for the number of channels.
#         channel_mult_emb    = 4,            # Multiplier for the dimensionality of the embedding vector.
#         num_blocks          = 4,            # Number of residual blocks per resolution.
#         attn_resolutions    = [16],         # List of resolutions with self-attention.
#         dropout             = 0.10,         # Dropout probability of intermediate activations.
#         label_dropout       = 0,            # Dropout probability of class labels for classifier-free guidance.

#         embedding_type      = 'positional', # Timestep embedding type: 'positional' for DDPM++, 'fourier' for NCSN++.
#         channel_mult_noise  = 0,            # Timestep embedding size: 1 for DDPM++, 2 for NCSN++.
#         encoder_type        = 'standard',   # Encoder architecture: 'standard' for DDPM++, 'residual' for NCSN++.
#         decoder_type        = 'standard',   # Decoder architecture: 'standard' for both DDPM++ and NCSN++.
#         resample_filter     = [1,1],        # Resampling filter: [1,1] for DDPM++, [1,3,3,1] for NCSN++.
#     ):
#         assert embedding_type in ['fourier', 'positional']
#         assert encoder_type in ['standard', 'skip', 'residual']
#         assert decoder_type in ['standard', 'skip']

#         super().__init__()
#         self.label_dropout = label_dropout
#         self.emb_dim_in = emb_dim_in
#         if emb_dim_in > 0:
#             emb_channels = model_channels * channel_mult_emb
#         else:
#             emb_channels = None
#         noise_channels = model_channels * channel_mult_noise
#         init = dict(init_mode='xavier_uniform')
#         init_zero = dict(init_mode='xavier_uniform', init_weight=1e-5)
#         init_attn = dict(init_mode='xavier_uniform', init_weight=np.sqrt(0.2))
#         block_kwargs = dict(
#             emb_channels=emb_channels, num_heads=1, dropout=dropout, skip_scale=np.sqrt(0.5), eps=1e-6,
#             resample_filter=resample_filter, resample_proj=True, adaptive_scale=False,
#             init=init, init_zero=init_zero, init_attn=init_attn,
#         )

#         # Mapping.
#         # self.map_label = Linear(in_features=label_dim, out_features=noise_channels, **init) if label_dim else None
#         # self.map_augment = Linear(in_features=augment_dim, out_features=noise_channels, bias=False, **init) if augment_dim else None
#         # self.map_layer0 = Linear(in_features=noise_channels, out_features=emb_channels, **init)
#         # self.map_layer1 = Linear(in_features=emb_channels, out_features=emb_channels, **init)
#         if emb_dim_in > 0:
#             self.map_layer0 = Linear(in_features=emb_dim_in, out_features=emb_channels, **init)
#             self.map_layer1 = Linear(in_features=emb_channels, out_features=emb_channels, **init)

#         if noise_channels > 0:
#             self.noise_map_layer0 = Linear(in_features=noise_channels, out_features=emb_channels, **init)
#             self.noise_map_layer1 = Linear(in_features=emb_channels, out_features=emb_channels, **init)

#         # Encoder.
#         self.enc = torch.nn.ModuleDict()
#         cout = in_channels
#         caux = in_channels
#         for level, mult in enumerate(channel_mult):
#             res = img_resolution >> level
#             if level == 0:
#                 cin = cout
#                 cout = model_channels
#                 self.enc[f'{res}x{res}_conv'] = Conv2d(in_channels=cin, out_channels=cout, kernel=3, **init)
#             else:
#                 self.enc[f'{res}x{res}_down'] = UNetBlock(in_channels=cout, out_channels=cout, down=True, **block_kwargs)
#                 if encoder_type == 'skip':
#                     self.enc[f'{res}x{res}_aux_down'] = Conv2d(in_channels=caux, out_channels=caux, kernel=0, down=True, resample_filter=resample_filter)
#                     self.enc[f'{res}x{res}_aux_skip'] = Conv2d(in_channels=caux, out_channels=cout, kernel=1, **init)
#                 if encoder_type == 'residual':
#                     self.enc[f'{res}x{res}_aux_residual'] = Conv2d(in_channels=caux, out_channels=cout, kernel=3, down=True, resample_filter=resample_filter, fused_resample=True, **init)
#                     caux = cout
#             for idx in range(num_blocks):
#                 cin = cout
#                 cout = model_channels * mult
#                 attn = (res in attn_resolutions)
#                 self.enc[f'{res}x{res}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout, attention=attn, **block_kwargs)
#         skips = [block.out_channels for name, block in self.enc.items() if 'aux' not in name]

#         # Decoder.
#         self.dec = torch.nn.ModuleDict()
#         for level, mult in reversed(list(enumerate(channel_mult))):
#             res = img_resolution >> level
#             if level == len(channel_mult) - 1:
#                 self.dec[f'{res}x{res}_in0'] = UNetBlock(in_channels=cout, out_channels=cout, attention=True, **block_kwargs)
#                 self.dec[f'{res}x{res}_in1'] = UNetBlock(in_channels=cout, out_channels=cout, **block_kwargs)
#             else:
#                 self.dec[f'{res}x{res}_up'] = UNetBlock(in_channels=cout, out_channels=cout, up=True, **block_kwargs)
#             for idx in range(num_blocks + 1):
#                 cin = cout + skips.pop()
#                 cout = model_channels * mult
#                 attn = (idx == num_blocks and res in attn_resolutions)
#                 self.dec[f'{res}x{res}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout, attention=attn, **block_kwargs)
#             if decoder_type == 'skip' or level == 0:
#                 if decoder_type == 'skip' and level < len(channel_mult) - 1:
#                     self.dec[f'{res}x{res}_aux_up'] = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=0, up=True, resample_filter=resample_filter)
#                 self.dec[f'{res}x{res}_aux_norm'] = GroupNorm(num_channels=cout, eps=1e-6)
#                 self.dec[f'{res}x{res}_aux_conv'] = Conv2d(in_channels=cout, out_channels=out_channels, kernel=3, init_weight=0.2, **init)# init_zero)

#     def forward(self, x, film_camera_emb=None, N_views_xa=1):

#         emb = None

#         if film_camera_emb is not None:
#             if self.emb_dim_in != 1:
#                 film_camera_emb = film_camera_emb.reshape(
#                     film_camera_emb.shape[0], 2, -1).flip(1).reshape(*film_camera_emb.shape) # swap sin/cos
#             film_camera_emb = silu(self.map_layer0(film_camera_emb))
#             film_camera_emb = silu(self.map_layer1(film_camera_emb))
#             emb = film_camera_emb

#         # Encoder.
#         skips = []
#         aux = x
#         for name, block in self.enc.items():
#             if 'aux_down' in name:
#                 aux = block(aux, N_views_xa)
#             elif 'aux_skip' in name:
#                 x = skips[-1] = x + block(aux, N_views_xa)
#             elif 'aux_residual' in name:
#                 x = skips[-1] = aux = (x + block(aux, N_views_xa)) / np.sqrt(2)
#             else:
#                 x = block(x, emb=emb, N_views_xa=N_views_xa) if isinstance(block, UNetBlock) \
#                     else block(x, N_views_xa=N_views_xa)
#                 skips.append(x)

#         # Decoder.
#         aux = None
#         tmp = None
#         for name, block in self.dec.items():
#             if 'aux_up' in name:
#                 aux = block(aux, N_views_xa)
#             elif 'aux_norm' in name:
#                 tmp = block(x, N_views_xa)
#             elif 'aux_conv' in name:
#                 tmp = block(silu(tmp), N_views_xa)
#                 aux = tmp if aux is None else tmp + aux
#             else:
#                 if x.shape[1] != block.in_channels:
#                     # skip connection is pixel-aligned which is good for
#                     # foreground features
#                     # but it's not good for gradient flow and background features
#                     x = torch.cat([x, skips.pop()], dim=1)
#                 x = block(x, emb=emb, N_views_xa=N_views_xa)
#         return aux

# # ================== End of implementation taken from EDM ===============
# # NVIDIA copyright does not apply to the code below this line

# class SingleImageSongUNetPredictor(nn.Module):
#     def __init__(self, cfg, out_channels, bias, scale):
#         super(SingleImageSongUNetPredictor, self).__init__()
#         self.out_channels = out_channels
#         self.cfg = cfg
#         if cfg.cam_embd.embedding is None:
#             #in_channels = 3
#             in_channels = 1
#             emb_dim_in = 0
#         else:
#            #in_channels = 3
#             in_channels = 1
#             emb_dim_in = 6 * cfg.cam_embd.dimension

#         self.encoder = SongUNet(cfg.data.training_resolution, 
#                                 in_channels, 
#                                 sum(out_channels),
#                                 model_channels=cfg.model.base_dim,
#                                 num_blocks=cfg.model.num_blocks,
#                                 emb_dim_in=emb_dim_in,
#                                 channel_mult_noise=0,
#                                 attn_resolutions=cfg.model.attention_resolutions)
#         self.out = nn.Conv2d(in_channels=sum(out_channels), 
#                                  out_channels=sum(out_channels),
#                                  kernel_size=1)

#         start_channels = 0
#         for out_channel, b, s in zip(out_channels, bias, scale):
#             nn.init.xavier_uniform_(
#                 self.out.weight[start_channels:start_channels+out_channel,
#                                 :, :, :], s)
#             nn.init.constant_(
#                 self.out.bias[start_channels:start_channels+out_channel], b)
#             start_channels += out_channel

#     def forward(self, x, film_camera_emb=None, N_views_xa=1):
#         x = self.encoder(x, 
#                          film_camera_emb=film_camera_emb,
#                          N_views_xa=N_views_xa)

#         return self.out(x)

# def networkCallBack(cfg, name, out_channels, **kwargs):
#     if name == "SingleUNet":
#         return SingleImageSongUNetPredictor(cfg, out_channels, **kwargs)
#     else:
#         raise NotImplementedError


# def generate_pixel_coords(H, W, device='cpu'):
#     """
#     生成大小为 (H, W, 2) 的像素网格坐标:
#       pixel_coords[y, x] = (x, y)
#     这与上层网络中:
#       x_d = (uv[:,0] - offU) * du
#       y_d = (uv[:,1] - offV) * dv
#     的使用方式相匹配。
#     """
#     # 注意 meshgrid 的 indexing='ij' 使得 vv 对应 y, uu 对应 x
#     v_range = torch.arange(H, device=device, dtype=torch.float32)
#     u_range = torch.arange(W, device=device, dtype=torch.float32)
#     vv, uu = torch.meshgrid(v_range, u_range, indexing='ij')  # vv.shape = (H, W), uu.shape = (H, W)

#     # pixel_coords[..., 0] = x, pixel_coords[..., 1] = y
#     pixel_coords = torch.stack([uu, vv], dim=-1)  # (H, W, 2)
#     return pixel_coords

# class GaussianSplatPredictor(nn.Module):
#     """
#     针对 CBCT 重建的Predictor。
#     - 输入: (B, N_view, C, H, W) 的投影图像 + (可选)相机/角度信息
#     - 输出: 一个 dict，包含高斯参数 (xyz, scaling, rotation, density)
#     """

#     def __init__(self, cfg, is_ct=False):
#         """
#         Args:
#             cfg: 配置字典/对象
#         """
#         super(GaussianSplatPredictor, self).__init__()
#         self.cfg = cfg
#         self.emb_dim_in = 6 * cfg.cam_embd.dimension if cfg.cam_embd.embedding is not None else 0
#         self.angle_embed_dim = cfg.cam_embd.dimension
#         self.camera_embed = PositionalEmbedding(
#             num_channels=self.angle_embed_dim,
#             max_positions=10000, 
#             endpoint=False
#         )
#         if self.emb_dim_in > 0:
#             # 用线性层将 angle_embed 投射到目标维度，而非简单 repeat
#             self.angle_proj = nn.Linear(self.angle_embed_dim, self.emb_dim_in)
#         # ------- 拆分通道 + 初始化参数 -------
#         split_dims, scale_inits, bias_inits = self.get_splits_and_inits()
#         self.network = networkCallBack(
#             cfg, 
#             cfg.model.name,
#             out_channels=split_dims,
#             scale=scale_inits,
#             bias=bias_inits
#         )

#         # ------- 各种激活函数 --------
#         self.depth_act     = nn.Sigmoid()          
#         self.density_act   = nn.Softplus(beta=1.0) # 也可替换为 ReLU / Sigmoid
#         self.scaling_act   = torch.exp
#         self.rotation_act  = lambda x: nn.functional.normalize(x, dim=1)
#         self.offset_act    = lambda x: torch.tanh(x) * self.cfg.model.xyz_range  


#     def get_splits_and_inits(self):
#         """
#         这里仅做一个示例拆分:
#           depth(1), offset(2), density(1), scaling(3), rotation(4)
#           共 11 通道

#         同时给出对应的初始 scale / bias，可再自行微调。
#         """
#         split_dimensions = [1, 2, 1, 3, 4]  

#         scale_inits = [
#             self.cfg.model.depth_scale,    # depth
#             self.cfg.model.xyz_scale,      # offset
#             self.cfg.model.density_scale,  # density
#             self.cfg.model.scale_scale,    # scaling
#             1.0                            # rotation
#         ]
#         bias_inits = [
#             self.cfg.model.depth_bias,
#             self.cfg.model.xyz_bias,
#             self.cfg.model.density_bias,
#             np.log(self.cfg.model.scale_bias),  # exp()前的log
#             0.0
#         ]
#         return split_dimensions, scale_inits, bias_inits

#     def flatten_vector(self, x):
#         """
#         (B, C, H, W) -> (B, H*W, C)
#         """
#         B, C, H, W = x.shape
#         x = x.reshape(B, C, H*W)             # (B, C, HW)
#         x = x.permute(0, 2, 1).contiguous()  # (B, HW, C)
#         return x

#     def make_contiguous(self, tensor_dict):
#         """
#         确保输出张量转为 contiguous。
#         """
#         return {k: v.contiguous() for k, v in tensor_dict.items()}

#     def multi_view_union(self, tensor_dict, B, N_view):
#         """
#         将 (B*N_view, N, ...) reshape 成 (B, N_view*N, ...) 以合并多视图。
#         """
#         for k, v in tensor_dict.items():
#             # (B*N_view, N, ...)
#             v = v.reshape(B, N_view, *v.shape[1:])   # (B, N_view, N, ...)
#             v = v.reshape(B, -1, *v.shape[3:])       # (B, N_view*N, ...)
#             tensor_dict[k] = v
#         return tensor_dict

#     def transform_rotations(self, rotations, source_cv2wT_quat):
#         """
#         将预测的相机坐标系内的四元数 rotation 转到世界坐标系。
#         rotations: (B*N_view, N, 4)
#         source_cv2wT_quat: (B*N_view, 4)
#         """
#         # 广播
#         Mq = source_cv2wT_quat.unsqueeze(1).expand(*rotations.shape)
#         out_quat = quaternion_raw_multiply(Mq, rotations)
#         return out_quat
#     def pack_camera_params(self, cameras,scanner_cfg, device='cpu'):
#         """
#         将 camera_params(list[dict]) 中的关键几何字段堆叠成批量张量，以便向量化计算。
#         假设每个 dict 至少包含:
#         {
#             "view_to_world": (4, 4) 的张量,
#             ...
#             "angle": (可选) 角度信息
#         }
#         同时 scanner_cfg 中包含全局参数:
#             "DSO": float,
#             "dDetector": (du, dv),
#             "offDetector": (offU, offV)
#         """
#         n = len(cameras)  # B * N_view

#         # 收集每个相机的 view_to_world
#         v2w_list = [cam["view_to_world"].to(device) for cam in cameras]

#         # 对 scanner_cfg 中的参数进行重复，使其与相机数目匹配
#         DSO_list = [scanner_cfg["DSO"]] * n
#         du_val, dv_val = scanner_cfg["dDetector"]
#         offU_val, offV_val = scanner_cfg["offDetector"]
#         du_list = [du_val] * n
#         dv_list = [dv_val] * n
#         offU_list = [offU_val] * n
#         offV_list = [offV_val] * n

#         # 转为 Tensor，形状均为 (n,)
#         DSO_t   = torch.tensor(DSO_list,   dtype=torch.float32, device=device)
#         du_t    = torch.tensor(du_list,    dtype=torch.float32, device=device)
#         dv_t    = torch.tensor(dv_list,    dtype=torch.float32, device=device)
#         offU_t  = torch.tensor(offU_list,  dtype=torch.float32, device=device)
#         offV_t  = torch.tensor(offV_list,  dtype=torch.float32, device=device)

#         # 堆叠 view_to_world 形状: (n, 4, 4)
#         v2w_t   = torch.stack(v2w_list, dim=0)

#         return DSO_t, du_t, dv_t, offU_t, offV_t, v2w_t

#     def backproject_cbct_vec(
#     self,
#     depth_map,       # (B*N, H, W, 1)
#     offset_map,      # (B*N, H, W, 2)
#     DSO_t,           # (B*N,)    源到旋转中心距离
#     du_t, dv_t,      # (B*N,)    探测器像素尺寸
#     offU_t, offV_t,  # (B*N,)    探测器中心偏移
#     v2w_t,           # (B*N,4,4) view_to_world 变换矩阵
#     pixel_coords,    # (H, W, 2) 预先生成的像素网格坐标(u,v)
#     chunk_size=None  # 可选的分块大小，若显存压力大可以设置
#     ):
#         """
#         将网络输出的 (depth, offset) 从投影坐标反投影到世界坐标系下的 3D 点。
#         对于 CBCT 场景，视为锥束相机几何。

#         参数:
#             depth_map:   (B*N, H, W, 1)，网络预测的深度 (相机光心到场景点的距离)。
#             offset_map:  (B*N, H, W, 2)，网络预测的在相机坐标系内的 2D 偏移。
#             DSO_t:       (B*N,) 源到旋转中心(或检测器坐标系原点)的距离。
#             du_t, dv_t:  (B*N,) 探测器像素尺寸 (mm/像素)。
#             offU_t, offV_t: (B*N,) 探测器中心相对光轴的偏移 (单位：像素)。
#             v2w_t:       (B*N, 4, 4) 每个视角的 view_to_world 齐次变换矩阵。
#             pixel_coords:(H, W, 2) 不随 batch 变化的像素坐标网格。
#             chunk_size:  分块大小，若为 None 则一次性处理所有像素，否则分块避免显存不足。

#         返回:
#             pos_world: (B*N, H*W, 3) 在世界坐标系下的 3D 点。
#             这里假设相机坐标系中，原点位于 X 射线源位置，
#             且 z 轴指向探测器平面（DSO 为正）。
#         """
#         device = depth_map.device
#         BtimesN, H, W, _ = depth_map.shape

#         # 先将标量参数扩展到 (B*N,1,1) 以方便后续广播
#         DSO  = DSO_t.view(-1, 1, 1)
#         du   = du_t.view(-1, 1, 1)
#         dv   = dv_t.view(-1, 1, 1)
#         offU = offU_t.view(-1, 1, 1)
#         offV = offV_t.view(-1, 1, 1)

#         # 将 pixel_coords (H, W, 2) 扩展到 (B*N, H, W, 2)
#         # 注意：expand 并不复制数据；前提是 B*N 维度可广播
#         uv = pixel_coords.to(device)  # (H, W, 2)
#         uv = uv.unsqueeze(0).expand(BtimesN, -1, -1, -1)  # (B*N, H, W, 2)

#         # 分块处理的函数，避免一次把 (H*W) 都转成 (B*N, H*W, 3) 占用过多显存
#         def process_chunk(h_start, h_end):
#             """
#             处理指定行范围 [h_start, h_end) 的反投影，以减少显存压力。
#             """
#             # 取出该分块
#             uv_chunk      = uv[:, h_start:h_end, :, :]          # (B*N, chunkH, W, 2)
#             depth_chunk   = depth_map[:, h_start:h_end, :, :]     # (B*N, chunkH, W, 1)
#             offset_chunk  = offset_map[:, h_start:h_end, :, :]      # (B*N, chunkH, W, 2)  —— 注意：offset 现在为 2 通道

#             chunkH = uv_chunk.shape[1]

#             # 计算像素射线 (x_d, y_d, z_d) 并单位化
#             uv_x = uv_chunk[..., 0] - offU  # (B*N, chunkH, W)
#             uv_y = uv_chunk[..., 1] - offV

#             x_d = uv_x * du  # (B*N, chunkH, W)
#             y_d = uv_y * dv
#             z_d = DSO.expand_as(x_d)  # (B*N, chunkH, W)

#             # 构造射线向量（相机坐标系），并单位化
#             ray_dir = torch.stack([x_d, y_d, z_d], dim=-1)   # (B*N, chunkH, W, 3)
#             ray_dir_norm = torch.norm(ray_dir, dim=-1, keepdim=True) + 1e-8
#             ray_dir_unit = ray_dir / ray_dir_norm  # (B*N, chunkH, W, 3)

#             # 计算原始 3D 点（沿射线方向）
#             raw_point = depth_chunk * ray_dir_unit  # (B*N, chunkH, W, 3)

#             # --- 下面构造与 ray_dir_unit 正交的平面 ---
#             # 选定全局 up 向量
#             up = torch.tensor([0, 1, 0], device=device, dtype=ray_dir_unit.dtype)
#             up = up.view(1,1,1,3).expand_as(ray_dir_unit)  # (B*N, chunkH, W, 3)
            
#             # 若 ray_dir 与 up 过于平行，则用备用 up 向量
#             dot = (ray_dir_unit * up).sum(dim=-1, keepdim=True)  # (B*N, chunkH, W, 1)
#             # 扩展 mask 至与 up 相同形状，避免逐元素替换出错
#             mask = (torch.abs(dot) > 0.999).expand_as(up)
#             up_alt = torch.tensor([1, 0, 0], device=device, dtype=ray_dir_unit.dtype)
#             up_alt = up_alt.view(1,1,1,3).expand_as(up)
#             up = torch.where(mask, up_alt, up)

#             # 计算正交基 p = normalize(cross(up, ray_dir_unit))
#             p = torch.cross(up, ray_dir_unit, dim=-1)
#             p = p / (torch.norm(p, dim=-1, keepdim=True) + 1e-8)
#             # 计算 q = normalize(cross(ray_dir_unit, p))
#             q = torch.cross(ray_dir_unit, p, dim=-1)
#             q = q / (torch.norm(q, dim=-1, keepdim=True) + 1e-8)

#             # 提取 offset 的 2 个分量
#             offset_x = offset_chunk[..., 0:1]  # (B*N, chunkH, W, 1)
#             offset_y = offset_chunk[..., 1:2]  # (B*N, chunkH, W, 1)

#             # 结合原始点和横向偏移得到最终相机坐标下的 3D 点
#             pos_cam = raw_point + offset_x * p + offset_y * q  # (B*N, chunkH, W, 3)

#             # 转为齐次坐标
#             pos_cam_flat = pos_cam.view(BtimesN, chunkH*W, 3)  # (B*N, chunkH*W, 3)
#             ones = torch.ones(BtimesN, chunkH*W, 1, dtype=pos_cam.dtype, device=device)
#             pos_cam_h = torch.cat([pos_cam_flat, ones], dim=-1)  # (B*N, chunkH*W, 4)

#             # 做 view_to_world 变换
#             pos_world_h = torch.bmm(pos_cam_h, v2w_t.transpose(1, 2))  # (B*N, chunkH*W, 4)
#             pos_world_chunk = pos_world_h[..., :3] / (pos_world_h[..., 3:] + 1e-8)  # (B*N, chunkH*W, 3)

#             return pos_world_chunk

#         # 若不需要分块，直接一次性处理
#         if not chunk_size or chunk_size >= H:
#             pos_world_all = process_chunk(0, H)  # (B*N, H*W, 3)
#         else:
#             # 否则分多次把结果拼起来
#             pos_world_list = []
#             start = 0
#             while start < H:
#                 end = min(start + chunk_size, H)
#                 chunk_res = process_chunk(start, end)  # (B*N, chunkH*W, 3)
#                 pos_world_list.append(chunk_res)
#                 start = end
#             # 在第2维(即像素维度)拼接
#             pos_world_all = torch.cat(pos_world_list, dim=1)  # (B*N, H*W, 3)

#         return pos_world_all

#     def forward(
#         self,
#         x,                       # (B, N_view, C, H, W) 投影图像
#         source_cv2wT_quat=None,  # (N_view,4)的张量
#         camera_params=None,      # list[dict], 长度 = B*N_view
#         scanner_cfg=None,
#         activate_output=True
#     ):
#         """
#         1) x -> 网络输出 (depth, offset, density, scaling, rotation)
#         2) 激活函数处理
#         3) backproject_cbct -> xyz
#         4) rotation -> world space (可选)
#         5) 整理输出 (B, N_view, ...)
#         """

#         # 确保 camera_params 长度正确
#         B, N_views = x.shape[:2]

#         # 合并 B, N_views 维度
#         x = x.reshape(B*N_views, *x.shape[2:])  # (B*N_views, C, H, W)
#         x = x.contiguous()
#         H, W = x.shape[2:]

#         self.pixel_coords = generate_pixel_coords(H, W).to(x.device)
#         # ============ 相机/角度嵌入修复 =============
#         angle_emb = None
#         if camera_params is not None:
#             # 检查camera_params长度是否与B*N_views匹配
#             if len(camera_params) != B*N_views:

#                 if len(camera_params) == N_views:  
#                     camera_params = camera_params * B  
#                 else:
#                     # 如果长度不是N_views的倍数，可能有其他问题
#                     print(f"警告: camera_params长度 ({len(camera_params)}) 不是B*N_views ({B*N_views}) 的倍数")
#                     # 用第一个相机参数填充至所需长度
#                     first_cam = camera_params[0]
#                     camera_params = [first_cam] * (B*N_views)
            
#             # 现在camera_params长度应该等于B*N_views
#             # 提取角度信息
#             angles = []
#             for i in range(B*N_views):
#                 cam_i = camera_params[i]
#                 angles.append(cam_i["angle"])


#             angle_t = torch.tensor(angles, device=x.device, dtype=torch.float32).unsqueeze(-1)
            

#             base_emb = self.camera_embed(angle_t)  # (B*N_views, angle_embed_dim)
#             if self.emb_dim_in > 0:
#                 angle_emb = self.angle_proj(base_emb)  # (B*N_views, emb_dim_in)
#         # --------------- 网络前向 ---------------
#         # 假设我们的 network 支持传入 angle_emb 等条件
#         net_out = self.network(x, film_camera_emb=angle_emb, N_views_xa=N_views)
        
#         # 检查 split 是否正确
#         split_dims = [1, 2, 1, 3, 4]   # 根据实际模型调整
#         if net_out.shape[1] != sum(split_dims):
#             raise ValueError(f"网络输出通道数 {net_out.shape[1]} 与预期总和 {sum(split_dims)} 不匹配!")
        
#         # 根据 get_splits_and_inits() 中的顺序拆分
#         depth, offset, density, scaling, rotation = net_out.split(split_dims, dim=1)

#         # ============ 激活处理 ============
#         if activate_output:
#             depth_acted    = self.depth_act(depth)
#             density_acted  = self.density_act(density)
#             scaling_acted  = self.scaling_act(scaling)
#             rotation_acted = self.rotation_act(rotation)
#             offset_acted   = self.offset_act(offset)
#         else:
#             depth_acted    = depth
#             density_acted  = density
#             scaling_acted  = scaling
#             rotation_acted = rotation
#         if getattr(self.cfg.model, "use_zrange", False):
#             z_near = getattr(self.cfg.model, "z_near", 0.01)
#             z_far  = getattr(self.cfg.model, "z_far", 100)
#             # 注意判断 z_far > z_near
#             if z_far > z_near:
#                 depth_acted = z_near + (z_far - z_near) * depth_acted
#             else:
#                 raise ValueError(f"cfg.model.z_near={z_near} 不应 >= z_far={z_far} !")        
#         # 若只想各向同性，可把 scaling 的 3 通道变为相同数值
#         if self.cfg.model.isotropic:
#             scaling_acted = scaling_acted[:, :1, ...].repeat(1, 3, 1, 1)

#         # ============ 后向投影得到 xyz ============
#         if camera_params is None or scanner_cfg is None:
#             raise ValueError("必须提供 camera_params 和 scanner_cfg 用于 CBCT 后向投影")
#         DSO_t, du_t, dv_t, offU_t, offV_t, v2w_t = self.pack_camera_params(camera_params, scanner_cfg, device=x.device)


#         depth_reshape  = depth_acted.permute(0,2,3,1)   # (B*N, H, W, 1)
#         offset_reshape = offset_acted.permute(0,2,3,1)     # (B*N, H, W, 2)

#         pos_world = self.backproject_cbct_vec(
#             depth_map   = depth_reshape,
#             offset_map  = offset_reshape,
#             DSO_t       = DSO_t,
#             du_t        = du_t,
#             dv_t        = dv_t,
#             offU_t      = offU_t,
#             offV_t      = offV_t,
#             v2w_t       = v2w_t,
#             pixel_coords= self.pixel_coords,
#             chunk_size  = 32  # 若显存充足可设为 None
#         )  # pos_world: (B*N, H*W, 3)

#         # ============ rotation -> world space (若需要) ============
#         rotation_flat = self.flatten_vector(rotation_acted)  # (B*N, H*W, 4)
#         if source_cv2wT_quat is not None:
#             scq = source_cv2wT_quat.view(B*N_views, 4)
#             rot_world = self.transform_rotations(rotation_flat, scq)
#         else:
#             rot_world = rotation_flat

#         # ============ 整理输出 ============
#         density_flat = self.flatten_vector(density_acted)   # (B*N, HW, 1)
#         scaling_flat = self.flatten_vector(scaling_acted)   # (B*N, HW, 3)

#         out_dict = {
#             "xyz":      pos_world,   # (B*N, HW, 3)
#             "rotation": rot_world,   # (B*N, HW, 4)
#             "density":  density_flat,# (B*N, HW, 1)
#             "scaling":  scaling_flat # (B*N, HW, 3)
#         }

#         # 将 (B*N, ...) -> (B, N_view, ...)
#         out_dict = self.multi_view_union(out_dict, B, N_views)
#         out_dict = self.make_contiguous(out_dict)

#         return out_dict
import torch
import torch.nn as nn
import torchvision 

import numpy as np
import torch
from torch.nn.functional import silu

from einops import rearrange, repeat

from utils.general_utils import matrix_to_quaternion, quaternion_raw_multiply
from utils.graphics_utils import fov2focal

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
        self.out_channels = out_channels
        self.cfg = cfg
        if cfg.cam_embd.embedding is None:
             #in_channels = 3
            in_channels = 1
            emb_dim_in = 0
        else:
           #in_channels = 3
            in_channels = 1
            emb_dim_in = 6 * cfg.cam_embd.dimension

        self.encoder = SongUNet(cfg.data.training_resolution, 
                                in_channels, 
                                sum(out_channels),
                                model_channels=cfg.model.base_dim,
                                num_blocks=cfg.model.num_blocks,
                                emb_dim_in=emb_dim_in,
                                channel_mult_noise=0,
                                attn_resolutions=cfg.model.attention_resolutions)
        self.out = nn.Conv2d(in_channels=sum(out_channels), 
                                 out_channels=sum(out_channels),
                                 kernel_size=1)

        start_channels = 0
        for out_channel, b, s in zip(out_channels, bias, scale):
            nn.init.xavier_uniform_(
                self.out.weight[start_channels:start_channels+out_channel,
                                :, :, :], s)
            nn.init.constant_(
                self.out.bias[start_channels:start_channels+out_channel], b)
            start_channels += out_channel

    def forward(self, x, film_camera_emb=None, N_views_xa=1):
        x = self.encoder(x, 
                         film_camera_emb=film_camera_emb,
                         N_views_xa=N_views_xa)

        return self.out(x)

def networkCallBack(cfg, name, out_channels, **kwargs):
    if name == "SingleUNet":
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
    针对 CBCT 重建的Predictor。
    - 输入: (B, N_view, C, H, W) 的投影图像 + (可选)相机/角度信息
    - 输出: 一个 dict，包含高斯参数 (xyz, scaling, rotation, density)
    """

    def __init__(self, cfg):
        """
        Args:
            cfg: 配置字典/对象
        """
        super(GaussianSplatPredictor, self).__init__()
        self.cfg = cfg
        self.emb_dim_in = 6 * cfg.cam_embd.dimension if cfg.cam_embd.embedding is not None else 0
        self.angle_embed_dim = cfg.cam_embd.dimension
        self.cam_embedding_map = PositionalEmbedding(self.cfg.cam_embd.dimension) 
        if self.emb_dim_in > 0:
            # 用线性层将 angle_embed 投射到目标维度，而非简单 repeat
            self.angle_proj = nn.Linear(self.angle_embed_dim, self.emb_dim_in)
        # ------- 拆分通道 + 初始化参数 -------
        split_dims, scale_inits, bias_inits = self.get_splits_and_inits()
        self.network = networkCallBack(
            cfg, 
            cfg.model.name,
            out_channels=split_dims,
            scale=scale_inits,
            bias=bias_inits
        )

        # ------- 各种激活函数 --------
        self.depth_act     = nn.Sigmoid()          
        self.density_act   = nn.Softplus(beta=1.0) # 也可替换为 ReLU / Sigmoid
        self.scaling_act   = torch.exp
        self.rotation_act  = lambda x: nn.functional.normalize(x, dim=1)
        self.offset_act    = lambda x: torch.tanh(x) * self.cfg.model.xyz_range  


    def get_splits_and_inits(self):
        """
        这里仅做一个示例拆分:
          depth(1), offset(2), density(1), scaling(3), rotation(4)
          共 11 通道

        同时给出对应的初始 scale / bias，可再自行微调。
        """
        split_dimensions = [1, 2, 1, 3, 4]  

        scale_inits = [
            self.cfg.model.depth_scale,    # depth
            self.cfg.model.xyz_scale,      # offset
            self.cfg.model.density_scale,  # density
            self.cfg.model.scale_scale,    # scaling
            1.0                            # rotation
        ]
        bias_inits = [
            self.cfg.model.depth_bias,
            self.cfg.model.xyz_bias,
            self.cfg.model.density_bias,
            np.log(self.cfg.model.scale_bias),  # exp()前的log
            0.0
        ]
        return split_dimensions, scale_inits, bias_inits

    def flatten_vector(self, x):
        """
        (B, C, H, W) -> (B, H*W, C)
        """
        B, C, H, W = x.shape
        x = x.reshape(B, C, H*W)             # (B, C, HW)
        x = x.permute(0, 2, 1).contiguous()  # (B, HW, C)
        return x

    def make_contiguous(self, tensor_dict):
        """
        确保输出张量转为 contiguous。
        """
        return {k: v.contiguous() for k, v in tensor_dict.items()}

    def multi_view_union(self, tensor_dict, B, N_view):
        """
        将 (B*N_view, N, ...) reshape 成 (B, N_view*N, ...) 以合并多视图。
        """
        for k, v in tensor_dict.items():
            # (B*N_view, N, ...)
            v = v.reshape(B, N_view, *v.shape[1:])   # (B, N_view, N, ...)
            v = v.reshape(B, -1, *v.shape[3:])       # (B, N_view*N, ...)
            tensor_dict[k] = v
        return tensor_dict

    def transform_rotations(self, rotations, source_cv2wT_quat):
        """
        将预测的相机坐标系内的四元数 rotation 转到世界坐标系。
        rotations: (B*N_view, N, 4)
        source_cv2wT_quat: (B*N_view, 4)
        """
        # 广播
        Mq = source_cv2wT_quat.unsqueeze(1).expand(*rotations.shape)
        out_quat = quaternion_raw_multiply(Mq, rotations)
        return out_quat
    def pack_camera_params(self, cameras, scanner_cfg, device='cpu'):
        """
        将 camera_params 列表中的关键几何字段堆叠成批量张量，以便向量化计算。
        参数:
            cameras: list[dict]，长度 = B*N_view，每个元素包含:
                "view_to_world": (4, 4) 的张量
            scanner_cfg: list[dict]，长度 = B，每个字典包含:
                "DSO": float,
                "dDetector": (du, dv),
                "offDetector": (offU, offV)
        返回:
            DSO_t: (B*N_view,)
            du_t, dv_t: (B*N_view,)
            offU_t, offV_t: (B*N_view,)
            v2w_t: (B*N_view, 4, 4)
        """
        B = len(scanner_cfg)
        N_view = len(cameras) // B  # 每个样本的视角数

        # 收集每个相机的 view_to_world
        v2w_list = [cam["view_to_world"].to(device) for cam in cameras]

        # 对 scanner_cfg 中的参数进行重复，使其与相机数目匹配
        DSO_list = []
        du_list = []
        dv_list = []
        offU_list = []
        offV_list = []
        for j in range(len(cameras)):
            sample_idx = j // N_view
            scanner = scanner_cfg[sample_idx]
            DSO_list.append(scanner["DSO"])
            du_val, dv_val = scanner["dDetector"]
            offU_val, offV_val = scanner["offDetector"]
            du_list.append(du_val)
            dv_list.append(dv_val)
            offU_list.append(offU_val)
            offV_list.append(offV_val)

        # 转为 Tensor，形状均为 (B*N_view,)
        DSO_t   = torch.tensor(DSO_list,   dtype=torch.float32, device=device)
        du_t    = torch.tensor(du_list,    dtype=torch.float32, device=device)
        dv_t    = torch.tensor(dv_list,    dtype=torch.float32, device=device)
        offU_t  = torch.tensor(offU_list,  dtype=torch.float32, device=device)
        offV_t  = torch.tensor(offV_list,  dtype=torch.float32, device=device)

        # 堆叠 view_to_world 形状: (B*N_view, 4, 4)
        v2w_t   = torch.stack(v2w_list, dim=0)

        return DSO_t, du_t, dv_t, offU_t, offV_t, v2w_t

    def backproject_cbct_vec(
    self,
    depth_map,       # (B*N, H, W, 1)
    offset_map,      # (B*N, H, W, 2)
    DSO_t,           # (B*N,)    源到旋转中心距离
    du_t, dv_t,      # (B*N,)    探测器像素尺寸
    offU_t, offV_t,  # (B*N,)    探测器中心偏移
    v2w_t,           # (B*N,4,4) view_to_world 变换矩阵
    pixel_coords,    # (H, W, 2) 预先生成的像素网格坐标(u,v)
    chunk_size=None  # 可选的分块大小，若显存压力大可以设置
    ):
        """
        将网络输出的 (depth, offset) 从投影坐标反投影到世界坐标系下的 3D 点。
        对于 CBCT 场景，视为锥束相机几何。

        参数:
            depth_map:   (B*N, H, W, 1)，网络预测的深度 (相机光心到场景点的距离)。
            offset_map:  (B*N, H, W, 2)，网络预测的在相机坐标系内的 2D 偏移。
            DSO_t:       (B*N,) 源到旋转中心(或检测器坐标系原点)的距离。
            du_t, dv_t:  (B*N,) 探测器像素尺寸 (mm/像素)。
            offU_t, offV_t: (B*N,) 探测器中心相对光轴的偏移 (单位：像素)。
            v2w_t:       (B*N, 4, 4) 每个视角的 view_to_world 齐次变换矩阵。
            pixel_coords:(H, W, 2) 不随 batch 变化的像素坐标网格。
            chunk_size:  分块大小，若为 None 则一次性处理所有像素，否则分块避免显存不足。

        返回:
            pos_world: (B*N, H*W, 3) 在世界坐标系下的 3D 点。
            这里假设相机坐标系中，原点位于 X 射线源位置，
            且 z 轴指向探测器平面（DSO 为正）。
        """
        device = depth_map.device
        BtimesN, H, W, _ = depth_map.shape

        # 先将标量参数扩展到 (B*N,1,1) 以方便后续广播
        DSO  = DSO_t.view(-1, 1, 1)
        du   = du_t.view(-1, 1, 1)
        dv   = dv_t.view(-1, 1, 1)
        offU = offU_t.view(-1, 1, 1)
        offV = offV_t.view(-1, 1, 1)

        # 将 pixel_coords (H, W, 2) 扩展到 (B*N, H, W, 2)
        # 注意：expand 并不复制数据；前提是 B*N 维度可广播
        uv = pixel_coords.to(device)  # (H, W, 2)
        uv = uv.unsqueeze(0).expand(BtimesN, -1, -1, -1)  # (B*N, H, W, 2)


        def process_chunk(h_start, h_end):
            """
            处理指定行范围 [h_start, h_end) 的反投影，以减少显存压力。
            """
            # 取出该分块
            uv_chunk      = uv[:, h_start:h_end, :, :]          # (B*N, chunkH, W, 2)
            depth_chunk   = depth_map[:, h_start:h_end, :, :]     # (B*N, chunkH, W, 1)
            offset_chunk  = offset_map[:, h_start:h_end, :, :]      # (B*N, chunkH, W, 2)  —— 注意：offset 现在为 2 通道

            chunkH = uv_chunk.shape[1]

            # 计算像素射线 (x_d, y_d, z_d) 并单位化
            uv_x = uv_chunk[..., 0] - offU  # (B*N, chunkH, W)
            uv_y = uv_chunk[..., 1] - offV

            x_d = uv_x * du  # (B*N, chunkH, W)
            y_d = uv_y * dv
            z_d = DSO.expand_as(x_d)  # (B*N, chunkH, W)

            # 构造射线向量（相机坐标系），并单位化
            ray_dir = torch.stack([x_d, y_d, z_d], dim=-1)   # (B*N, chunkH, W, 3)
            ray_dir_norm = torch.norm(ray_dir, dim=-1, keepdim=True) + 1e-8
            ray_dir_unit = ray_dir / ray_dir_norm  # (B*N, chunkH, W, 3)

            # 计算原始 3D 点（沿射线方向）
            raw_point = depth_chunk * ray_dir_unit  # (B*N, chunkH, W, 3)

            # --- 下面构造与 ray_dir_unit 正交的平面 ---
            # 选定全局 up 向量
            up = torch.tensor([0, 1, 0], device=device, dtype=ray_dir_unit.dtype)
            up = up.view(1,1,1,3).expand_as(ray_dir_unit)  # (B*N, chunkH, W, 3)
            
            # 若 ray_dir 与 up 过于平行，则用备用 up 向量
            dot = (ray_dir_unit * up).sum(dim=-1, keepdim=True)  # (B*N, chunkH, W, 1)
            # 扩展 mask 至与 up 相同形状，避免逐元素替换出错
            mask = (torch.abs(dot) > 0.999).expand_as(up)
            up_alt = torch.tensor([1, 0, 0], device=device, dtype=ray_dir_unit.dtype)
            up_alt = up_alt.view(1,1,1,3).expand_as(up)
            up = torch.where(mask, up_alt, up)

            # 计算正交基 p = normalize(cross(up, ray_dir_unit))
            p = torch.cross(up, ray_dir_unit, dim=-1)
            p = p / (torch.norm(p, dim=-1, keepdim=True) + 1e-8)
            # 计算 q = normalize(cross(ray_dir_unit, p))
            q = torch.cross(ray_dir_unit, p, dim=-1)
            q = q / (torch.norm(q, dim=-1, keepdim=True) + 1e-8)

            # 提取 offset 的 2 个分量
            offset_x = offset_chunk[..., 0:1]  # (B*N, chunkH, W, 1)
            offset_y = offset_chunk[..., 1:2]  # (B*N, chunkH, W, 1)

            # 结合原始点和横向偏移得到最终相机坐标下的 3D 点
            pos_cam = raw_point + offset_x * p + offset_y * q  # (B*N, chunkH, W, 3)

            # 转为齐次坐标
            pos_cam_flat = pos_cam.view(BtimesN, chunkH*W, 3)  # (B*N, chunkH*W, 3)
            ones = torch.ones(BtimesN, chunkH*W, 1, dtype=pos_cam.dtype, device=device)
            pos_cam_h = torch.cat([pos_cam_flat, ones], dim=-1)  # (B*N, chunkH*W, 4)

            pos_world_h = torch.bmm(
                v2w_t.transpose(1,2),       # 注意这里多了个 .transpose(1,2)
                pos_cam_h.transpose(1,2)
            ).transpose(1,2)
            pos_world_chunk = pos_world_h[..., :3] / (pos_world_h[..., 3:] + 1e-8)  # (B*N, chunkH*W, 3)

            return pos_world_chunk

        # 若不需要分块，直接一次性处理
        if not chunk_size or chunk_size >= H:
            pos_world_all = process_chunk(0, H)  # (B*N, H*W, 3)
        else:
            # 否则分多次把结果拼起来
            pos_world_list = []
            start = 0
            while start < H:
                end = min(start + chunk_size, H)
                chunk_res = process_chunk(start, end)  # (B*N, chunkH*W, 3)
                pos_world_list.append(chunk_res)
                start = end
            # 在第2维(即像素维度)拼接
            pos_world_all = torch.cat(pos_world_list, dim=1)  # (B*N, H*W, 3)

        return pos_world_all
    def get_camera_embeddings(self, view_to_world_tensor):
        """
        获取相机嵌入。
        Args:
            view_to_world_tensor (torch.Tensor): 形状为 (B, N_view, 4, 4) 的张量，来自 camera_params 的 "view_to_world" 字段
        Returns:
            torch.Tensor: 相机嵌入，形状为 (B, N_view, emb_dim)。
        """
        b, n_view = view_to_world_tensor.shape[:2]
        if self.cfg.cam_embd.embedding == "index":
            cam_embedding = torch.arange(n_view, 
                                         dtype=view_to_world_tensor.dtype,
                                         device=view_to_world_tensor.device,
                                         ).unsqueeze(0).expand(b, n_view).unsqueeze(2)  # 创建索引嵌入
        elif self.cfg.cam_embd.embedding == "pose":
            # 连接 view_to_world 的第 4 行和第 3 行的前三个元素
            cam_embedding = torch.cat([view_to_world_tensor[:, :, 3, :3],
                                       view_to_world_tensor[:, :, 2, :3]], dim=2)
        else:
            raise ValueError("未知的 cam_embd.embedding 类型")
        cam_embedding = rearrange(cam_embedding, 'b n_view c -> (b n_view) c')
        cam_embedding = self.cam_embedding_map(cam_embedding)
        cam_embedding = rearrange(cam_embedding, '(b n_view) c -> b n_view c', b=b, n_view=n_view)
        return cam_embedding
    # TO DO: 修改数据输入维度

    def forward(
        self,
        x,                       # (B, N_view, C, H, W) 投影图像
        source_cv2wT_quat=None,  # (B,N_view,4)的张量
        camera_params=None,      # list[dict]，长度 = B*N_view，每个元素包含 "angle" 和 "view_to_world"
        scanner_cfg=None,        # list[dict]，长度 = B，每个字典包含 "DSO", "dDetector", "offDetector"
        activate_output=True
    ):
        """
        1) x -> 网络输出 (depth, offset, density, scaling, rotation)
        2) 激活函数处理
        3) backproject_cbct -> xyz
        4) rotation -> world space (可选)
        5) 整理输出 (B, N_view, ...)
        """
        ''' 
        "input_images": batch_input_images,           # [B, input_images_count, C, H, W]
        "camera_params": batch_camera_params,         # 列表，长度=B×input_images,每个元素为字典，包含"angle"shape为[input_images_count]；"view_to_world"shape为[ 4, 4]
        "scanner_cfg": scanner_cfg_list,              # 列表，长度 = B
        "bbox": batch_bbox,                           # [B, ...]
        "source_cv2wT_quat": batch_source_cv2wT_quat,   # [B, input_images_count, 4]
        "cameras": cameras_list                       # 原始 cameras 列表，每个 sample 为 list[Camera]'''
        # 确保 camera_params 长度正确
        B, N_views = x.shape[:2]
        # 合并 B, N_views 维度
        x = x.reshape(B*N_views, *x.shape[2:])  # (B*N_views, C, H, W)
        x = x.contiguous()
        H, W = x.shape[2:]

        self.pixel_coords = generate_pixel_coords_centered(H, W).to(x.device)
        # ============ 相机/角度嵌入修复 =============
        view_to_world_list = [cam["view_to_world"] for cam in camera_params]  # 每个元素形状 (4,4)
        view_to_world_tensor = torch.stack(view_to_world_list, dim=0)  # (B*N_views, 4, 4)
        view_to_world_tensor = view_to_world_tensor.reshape(B, N_views, 4, 4)
        cam_embedding = self.get_camera_embeddings(view_to_world_tensor)  # (B, N_views, emb_dim)
        film_camera_emb = cam_embedding.reshape(B * N_views, cam_embedding.shape[2])



        # --------------- 网络前向 ---------------
        # 假设我们的 network 支持传入 angle_emb 等条件
        net_out = self.network(x, film_camera_emb=film_camera_emb, N_views_xa=N_views)
        
        # 检查 split 是否正确
        split_dims = [1, 2, 1, 3, 4]   # 根据实际模型调整
        if net_out.shape[1] != sum(split_dims):
            raise ValueError(f"网络输出通道数 {net_out.shape[1]} 与预期总和 {sum(split_dims)} 不匹配!")
        
        # 根据 get_splits_and_inits() 中的顺序拆分
        depth, offset, density, scaling, rotation = net_out.split(split_dims, dim=1)

        # ============ 激活处理 ============
        if activate_output:
            depth_acted    = self.depth_act(depth)
            density_acted  = self.density_act(density)
            scaling_acted  = self.scaling_act(scaling)
            rotation_acted = self.rotation_act(rotation)
            offset_acted   = self.offset_act(offset)

        else:
            depth_acted    = depth
            density_acted  = density
            scaling_acted  = scaling
            rotation_acted = rotation
        if getattr(self.cfg.model, "use_zrange", False):
            z_near = getattr(self.cfg.model, "z_near", 0.01)
            z_far  = getattr(self.cfg.model, "z_far", 100)
            # 注意判断 z_far > z_near
            if z_far > z_near:
                depth_acted = z_near + (z_far - z_near) * depth_acted
            else:
                raise ValueError(f"cfg.model.z_near={z_near} 不应 >= z_far={z_far} !")        

        # 若只想各向同性，可把 scaling 的 3 通道变为相同数值
        if self.cfg.model.isotropic:
            scaling_acted = scaling_acted[:, :1, ...].repeat(1, 3, 1, 1)

        # ============ 后向投影得到 xyz ============
        if camera_params is None or scanner_cfg is None:
            raise ValueError("必须提供 camera_params 和 scanner_cfg 用于 CBCT 后向投影")
        DSO_t, du_t, dv_t, offU_t, offV_t, v2w_t = self.pack_camera_params(camera_params, scanner_cfg, device=x.device)


        depth_reshape  = depth_acted.permute(0,2,3,1)   # (B*N, H, W, 1)
        offset_reshape = offset_acted.permute(0,2,3,1)     # (B*N, H, W, 2)

        pos_world = self.backproject_cbct_vec(
            depth_map   = depth_reshape,
            offset_map  = offset_reshape,
            DSO_t       = DSO_t,
            du_t        = du_t,
            dv_t        = dv_t,
            offU_t      = offU_t,
            offV_t      = offV_t,
            v2w_t       = v2w_t,
            pixel_coords= self.pixel_coords,
            chunk_size  = None  # 若显存充足可设为 None
        )  # pos_world: (B*N, H*W, 3)

        # ============ rotation -> world space (若需要) ============
        rotation_flat = self.flatten_vector(rotation_acted)  # (B*N, H*W, 4)

        if source_cv2wT_quat is not None:
            scq = source_cv2wT_quat.view(B*N_views, 4)
            rot_world = self.transform_rotations(rotation_flat, scq)
        else:
            rot_world = rotation_flat

        # ============ 整理输出 ============
        density_flat = self.flatten_vector(density_acted)   # (B*N, HW, 1)
        scaling_flat = self.flatten_vector(scaling_acted)   # (B*N, HW, 3)
        offset_flat = self.flatten_vector(offset_acted)   # (B*N, HW, 2 )
        out_dict = {
            "xyz":      pos_world,       # (B*N, HW, 3)
            "rotation": rot_world,       # (B*N, HW, 4)
            "density":  density_flat,    # (B*N, HW, 1)
            "scaling":  scaling_flat,    # (B*N, HW, 3)
            "offset":   offset_flat      # (B*N, HW, 2 )  <-- 你需要在 forward 里保存 offset_acted 并 reshape。
        }
        # 将 (B*N, ...) -> (B, N_view, ...)
        out_dict = self.multi_view_union(out_dict, B, N_views)
        out_dict = self.make_contiguous(out_dict)

        return out_dict
