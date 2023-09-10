import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from mmcv.cnn.bricks.transformer import FFN, build_dropout
from mmcv.cnn.utils.weight_init import trunc_normal_
from mmcv.cnn import build_norm_layer, constant_init, trunc_normal_init
from mmcv.runner import BaseModule, ModuleList
from mmcv.utils import to_2tuple

import pdb

class WindowMSA(BaseModule):
    """Window based multi-head self-attention (W-MSA) module with relative
    position bias.

    Args:
        embed_dims (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): The height and width of the window.
        qkv_bias (bool, optional):  If True, add a learnable bias to q, k, v.
            Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        attn_drop_rate (float, optional): Dropout ratio of attention weight.
            Default: 0.0
        proj_drop_rate (float, optional): Dropout ratio of output. Default: 0.
        init_cfg (dict | None, optional): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 window_size,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop_rate=0.,
                 proj_drop_rate=0.,
                 init_cfg=None):

        super().__init__()
        self.embed_dims = embed_dims
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_embed_dims = embed_dims // num_heads
        self.scale = qk_scale or head_embed_dims**-0.5
        self.init_cfg = init_cfg

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1),
                        num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # About 2x faster than original impl
        Wh, Ww = self.window_size
        rel_index_coords = self.double_step_seq(2 * Ww - 1, Wh, 1, Ww)
        rel_position_index = rel_index_coords + rel_index_coords.T
        rel_position_index = rel_position_index.flip(1).contiguous()
        self.register_buffer('relative_position_index', rel_position_index)
        
        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop_rate)
        self.softmax = nn.Softmax(dim=-1)

    # 实现W-MSA和SW-MSA机制
    # attn和mask的计算公式为：
    # attn = softmax((qk)/sqrt(d_k) + b + mask)v
    # mask的值为-100，softmax后的值为0，相当于忽略掉对应的值
    # b为相对位置偏差，用于实现SW-MSA机制
    def forward(self, x, mask=None):
        """
        Args:

            x (tensor): input features with shape of (num_windows*B, N, C)
            mask (tensor | None, Optional): mask with shape of (num_windows,
                Wh*Ww, Wh*Ww), value should be between (-inf, 0].
        """

        # 使用全连接层将输入特征图一分为三，分别为q、k、v，形状从[B,N,C]变为[B,N,3,C]
        # 为了实现Mult-Head并行计算机制，把输入特征图拆分成num_heads份，形状从[B,N,3,C]变为[B,N,3,H,C//H]
        # 最后再将形状变成[3,B,H,N,C//H]，以便于拆分成q、k、v，便于计算qk的乘积，拆分后q、k、v的形状为[B,H,N,C//H]
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # scale就是论文中的缩放因子1/sqrt(d_k)，用于缩放q和k的乘积
        q = q * self.scale

        # 计算attn = (qk)/sqrt(d_k)，计算前q和k的形状为[B, H, N, C//H]
        # 计算后，attn的形状是[B, H, N, N]
        attn = (q @ k.transpose(-2, -1))

        # 计算相对位置偏差，并加到q和k的乘积上，也就是attn = softmax((qk)/sqrt(d_k) + b)v 中的b
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1],
                -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        # 将mask加到attn上，再进行softmax，mask的值为-100，softmax后的值为0，相当于忽略掉对应的值
        # 该mask主要用于实现Shifted Window Attention机制
        # 如果mask为None，则相当于进行W-MSA，如果mask不为None，则相当于进行SW-MSA
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B // nW, nW, self.num_heads, N,
                             N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)

        # 对attn进行dropout
        attn = self.attn_drop(attn)

        # 计算注意力的最后一步，即attn和v的乘积，形状是[B,H,N,C//H]
        # 再将Multi-Head的结果的形状还原成[B, N, C]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # 使用全连接层对特征进行映射
        x = self.proj(x)
        # 对输出特征进行dropout
        x = self.proj_drop(x)
        return x

    @staticmethod
    def double_step_seq(step1, len1, step2, len2):
        seq1 = torch.arange(0, step1 * len1, step1)
        seq2 = torch.arange(0, step2 * len2, step2)
        return (seq1[:, None] + seq2[None, :]).reshape(1, -1)


class ShiftWindowMSA(BaseModule):
    """Shifted Window Multihead Self-Attention Module.

    Args:
        embed_dims (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): The height and width of the window.
        shift_size (int, optional): The shift step of each window towards
            right-bottom. If zero, act as regular window-msa. Defaults to 0.
        qkv_bias (bool, optional): If True, add a learnable bias to q, k, v.
            Default: True
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Defaults: None.
        attn_drop_rate (float, optional): Dropout ratio of attention weight.
            Defaults: 0.
        proj_drop_rate (float, optional): Dropout ratio of output.
            Defaults: 0.
        dropout_layer (dict, optional): The dropout_layer used before output.
            Defaults: dict(type='DropPath', drop_prob=0.).
        init_cfg (dict, optional): The extra config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims, # 输入特征图的通道数C
                 num_heads,  # 多头注意力的头数H
                 window_size,  # 窗口的大小，默认值为7，也就是MSA的计算单元的大小
                 shift_size=0, # 循环移位的步长，这里是7//2=3，也就是不进行循环移位
                 qkv_bias=True, # 使用全连接对qkv进行空间映射时，是否使用偏置
                 qk_scale=None, # 缩放因子，用于缩放q和k的乘积，将覆盖默认值1/sqrt(d_k)
                 attn_drop_rate=0, # 注意力权重的dropout率
                 proj_drop_rate=0, # 输出特征的dropout率
                 dropout_layer=dict(type='DropPath', drop_prob=0.),
                 init_cfg=None):
        super().__init__(init_cfg)

        self.window_size = window_size
        self.shift_size = shift_size
        assert 0 <= self.shift_size < self.window_size

        self.w_msa = WindowMSA(
            embed_dims=embed_dims,
            num_heads=num_heads,
            window_size=to_2tuple(window_size),
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=proj_drop_rate,
            init_cfg=None)

        self.drop = build_dropout(dropout_layer)

    def forward(self, query, hw_shape):
        B, L, C = query.shape
        H, W = hw_shape
        assert L == H * W, 'input feature has wrong size'
        query = query.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        query = F.pad(query, (0, 0, 0, pad_r, 0, pad_b))
        H_pad, W_pad = query.shape[1], query.shape[2]

        # 对输入特征图进行循环移位，计算SW_MSA中循环移位所需要的mask
        # cyclic shift
        if self.shift_size > 0:
            shifted_query = torch.roll(
                query,
                shifts=(-self.shift_size, -self.shift_size),
                dims=(1, 2))

            # calculate attention mask for SW-MSA
            img_mask = torch.zeros((1, H_pad, W_pad, 1), device=query.device)
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size,
                              -self.shift_size), slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size,
                              -self.shift_size), slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            # nW, window_size, window_size, 1
            mask_windows = self.window_partition(img_mask)
            mask_windows = mask_windows.view(
                -1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0,
                                              float(-100.0)).masked_fill(
                                                  attn_mask == 0, float(0.0))
        else:
            shifted_query = query
            attn_mask = None

        # 将输入特征图[B, H, W, C]划分成若干个窗口，每个窗口为7*7，输出特征图的形状为[B*H/7*W/7, 7, 7, C]
        # 窗口是MSA的计算单元，计算窗口中每个像素与其他49个像素的关系
        # nW*B, window_size, window_size, C
        query_windows = self.window_partition(shifted_query)
        # nW*B, window_size*window_size, C
        # 将窗口中的像素展开成一维向量，形状为[B*H/7*W/7, 49, C]
        query_windows = query_windows.view(-1, self.window_size**2, C)

        # 对窗口进行W-MSA或SW-MSA计算，输入输出的形状都是[B*H/7*W/7, 49, C]
        # W-MSA/SW-MSA (nW*B, window_size*window_size, C)
        attn_windows = self.w_msa(query_windows, mask=attn_mask)

        # 将输出特征图的形状从[B*H/7*W/7, 49, C]变为[B*H/7*W/7, 7, 7, C]
        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size,
                                         self.window_size, C)

        # 将拆分的若干个窗口[B*H/7*W/7, 7, 7, C]还原成特征图，输出形状是[B, H, W, C]
        # B H' W' C
        shifted_x = self.window_reverse(attn_windows, H_pad, W_pad)

        # 还原循环移位后的特征图
        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(
                shifted_x,
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        x = self.drop(x)
        return x

    # @info 将拆分若干个窗口还原成特征图，输出形状是[B, H, W, C]
    def window_reverse(self, windows, H, W):
        """
        Args:
            windows: (num_windows*B, window_size, window_size, C)
            H (int): Height of image
            W (int): Width of image
        Returns:
            x: (B, H, W, C)
        """
        # windows的形状是[B*H/7*W/7, 7, 7, C]
        window_size = self.window_size
        # 从B*H/7*W/7中还原出B
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        # 将windows的形状变成[B, H/7, W/7, 7, 7, C]
        x = windows.view(B, H // window_size, W // window_size, window_size,
                         window_size, -1)
        # 将windows的形状变成[B, H/7, 7, W/7, 7, C]，最后再还原成[B, H, W, C]
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x

    # @info 将输入特征图拆分成若干个窗口，每个窗口为7*7，输出特征图的形状为[B*H/7*W/7, 7, 7, C]
    def window_partition(self, x):
        """
        Args:
            x: (B, H, W, C)
        Returns:
            windows: (num_windows*B, window_size, window_size, C)
        """
        B, H, W, C = x.shape
        window_size = self.window_size
        x = x.view(B, H // window_size, window_size, W // window_size,
                   window_size, C)
        # 特征图形状变成[B, H/7, W/7, 7, 7, C]
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        # 将不同Batch的窗口展开成一维向量，形状为[B*H/7*W/7, 7, 7, C]
        windows = windows.view(-1, window_size, window_size, C)
        
        return windows

# Swin Transformer的具体实现
# 即实现W-MSA和SW-MSA机制
# W-MSA是Window Multihead Self-Attention的缩写
# SW-MSA是Shifted Window Multihead Self-Attention的缩写
class SwinBlock(BaseModule):
    """"
    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        window_size (int, optional): The local window scale. Default: 7.
        shift (bool, optional): whether to shift window or not. Default False.
        qkv_bias (bool, optional): enable bias for qkv if True. Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        drop_rate (float, optional): Dropout rate. Default: 0.
        attn_drop_rate (float, optional): Attention dropout rate. Default: 0.
        drop_path_rate (float, optional): Stochastic depth rate. Default: 0.
        act_cfg (dict, optional): The config dict of activation function.
            Default: dict(type='GELU').
        norm_cfg (dict, optional): The config dict of normalization.
            Default: dict(type='LN').
        with_cp (bool, optional): Use checkpoint or not. Using checkpoint
            will save some memory while slowing down the training speed.
            Default: False.
        init_cfg (dict | list | None, optional): The init config.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 window_size=7,
                 shift=False,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 with_cp=False,
                 init_cfg=None):

        super(SwinBlock, self).__init__()

        self.init_cfg = init_cfg
        self.with_cp = with_cp
        
        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]

        # 构造W-MSA或SW-MSA
        # 每个窗口的大小为7*7，窗口是MSA的计算单元，计算窗口中每个像素与其他49个像素的关系
        # 如果shift为True，则使用SW-MSA，否则使用W-MSA
        # 如果使用SW-MSA，则每个窗口会进行循环移位，移位的步长为窗口的一半，即7/2=3
        self.attn = ShiftWindowMSA(
            embed_dims=embed_dims, # 输入特征图的通道数
            num_heads=num_heads, # 多头注意力的头数
            window_size=window_size, # 窗口大小，默认为7，也就是每个窗口为7*7，MSA只在这7*7的窗口内进行计算
            shift_size=window_size // 2 if shift else 0, # 循环移动的步长，默认为窗口的一半，7/2=3
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            init_cfg=None)

        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.ffn = FFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            num_fcs=2,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            act_cfg=act_cfg,
            add_identity=True,
            init_cfg=None)

    def forward(self, x):
        # convert x to (B, L, C) where L = H * W
        B, C, H, W = x.shape
        hw_shape = (H, W)
        x = x.permute(0, 2, 3, 1).contiguous().view(B, -1, C)

        def _inner_forward(x):
            identity = x
            x = self.norm1(x)
            x = self.attn(x, hw_shape)

            x = x + identity

            identity = x
            x = self.norm2(x)
            x = self.ffn(x, identity=identity)

            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)

        x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        return x

