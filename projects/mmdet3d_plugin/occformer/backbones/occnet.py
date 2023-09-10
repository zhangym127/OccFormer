import torch
import torch.nn as nn
import torch.utils.checkpoint as cp

from mmdet3d.models.builder import BACKBONES
from mmcv.runner import BaseModule
from mmcv.cnn import constant_init, trunc_normal_init, caffe2_xavier_init
from .dualpath_block import DualpathTransformerBlock

import pdb

@BACKBONES.register_module()
class OccupancyEncoder(BaseModule):
    def __init__(
            self,
            in_channels,
            num_stage=4,
            block_numbers=[2, 2, 2, 2],
            block_inplanes=[64, 128, 256, 512],
            block_strides=[1, 2, 2, 2],
            out_indices=(0, 1, 2, 3),
            norm_cfg=dict(type='BN3d', requires_grad=True),
            with_cp=True,
            **kwargs,
        ):
        
        super().__init__()
        self.out_indices = out_indices
        
        # 占用编码器划分为4个阶段，每个阶段的输出通道数分别为64, 128, 256, 512
        # 下面分别为这四个阶段构造双路径Swin Transformer
        # build layers
        self.num_layers = 0
        self.layers = nn.ModuleList()
        for i in range(num_stage):
            self.layers.append(self._make_layer(
                block=DualpathTransformerBlock, 
                in_channels=in_channels, 
                channels=block_inplanes[i], 
                stride=block_strides[i],
                num_block=block_numbers[i],
                norm_cfg=norm_cfg,
                stage_index=i,
                **kwargs,
            ))
            in_channels = block_inplanes[i]
        
        # with torch.checkpoint
        self.with_cp = with_cp
        
    def _make_layer(self, block, in_channels, channels, 
            num_block, stride=1, norm_cfg=None, **kwargs):
        
        layers = []
        for _ in range(num_block):
            layers.append(
                block(in_channels, channels, stride=stride, norm_cfg=norm_cfg, 
                      layer_index=self.num_layers, **kwargs))
            
            in_channels = channels
            stride = 1
            self.num_layers += 1
        
        return nn.Sequential(*layers)

    # @info 按照事先设定的四个阶段，分层逐一调用双路径Swin Transformer进行特征提取
    # 每一层都会通过卷积对输入特征图进行下采样，尺寸减半，通道数翻倍
    # 每一层的输出作为下一层的输入，各层输出的特征图尺寸依次减半，通道数翻倍
    # 每一层的输出都被保存下来，最后将四层的多尺度特征图给到MSDA
    def forward(self, x):
        res = []
        for index, layer in enumerate(self.layers):
            if self.with_cp and x.requires_grad:
                x = cp.checkpoint(layer, x)
            else:
                x = layer(x)
            
            if index in self.out_indices:
                res.append(x)
        
        return res