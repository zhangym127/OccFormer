import torch
from torch import nn
from torch.nn import functional as F
from mmcv.cnn import Conv3d, ConvModule, caffe2_xavier_init, normal_init, xavier_init
from mmcv.cnn.bricks.transformer import (build_positional_encoding,
                                         build_transformer_layer_sequence)
from projects.mmdet3d_plugin.utils.point_generator import MlvlPointGenerator
from mmcv.cnn import build_norm_layer

# from mmdet.models.utils.transformer import MultiScaleDeformableAttention
from .multi_scale_deform_attn_3d import MultiScaleDeformableAttention3D

from mmdet.models import NECKS
from mmcv.runner import BaseModule, ModuleList

import pdb

# designed for multi-scale deformable attention: 
# each pixel within every level will access multi-scale features
@NECKS.register_module()
class MSDeformAttnPixelDecoder3D(BaseModule):
    def __init__(
            self,
            in_channels=[256, 512, 1024, 2048],
            strides=[4, 8, 16, 32],
            feat_channels=256,
            out_channels=256,
            num_outs=3,
            conv_cfg=dict(type='Conv3d'),
            norm_cfg=dict(type='GN', num_groups=32),
            act_cfg=dict(type='ReLU'),
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=6,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='MultiScaleDeformableAttention3D',
                        embed_dims=256,
                        num_heads=8,
                        num_levels=3,
                        num_points=4,
                        im2col_step=64,
                        dropout=0.0,
                        batch_first=False,
                        norm_cfg=None,
                        init_cfg=None),
                    feedforward_channels=1024,
                    ffn_dropout=0.0,
                    operation_order=('self_attn', 'norm', 'ffn', 'norm')),
                init_cfg=None),
            positional_encoding=dict(
                type='SinePositionalEncoding3D',
                num_feats=128,
                normalize=True),
            init_cfg=None):
        
        super().__init__(init_cfg=init_cfg)
        
        self.strides = strides
        self.num_input_levels = len(in_channels) # 4, 实际的in_channels=[128, 256, 512, 1024]
        self.num_encoder_levels = encoder.transformerlayers.attn_cfgs.num_levels # 实际值是3
        assert self.num_encoder_levels >= 1
        
        # build input conv for channel adapation
        # from top to down (low to high resolution)
        input_conv_list = []
        # for i in range(3, 0, -1):
        for i in range(self.num_input_levels - 1,
                self.num_input_levels - self.num_encoder_levels - 1, -1):
            input_conv = ConvModule(
                in_channels[i],
                feat_channels,
                kernel_size=1,
                norm_cfg=norm_cfg,
                conv_cfg=conv_cfg,
                act_cfg=None,
                bias=True)
            input_conv_list.append(input_conv)

        self.input_convs = ModuleList(input_conv_list)
        self.encoder = build_transformer_layer_sequence(encoder)
        self.postional_encoding = build_positional_encoding(positional_encoding)
        self.level_encoding = nn.Embedding(self.num_encoder_levels, feat_channels)
        
        # fpn-like structure
        self.lateral_convs = ModuleList()
        self.output_convs = ModuleList()
        self.use_bias = norm_cfg is None
        # from top to down (low to high resolution)
        # fpn for the rest features that didn't pass in encoder
        for i in range(self.num_input_levels - self.num_encoder_levels - 1, -1, -1):
            lateral_conv = ConvModule(
                in_channels[i],
                feat_channels,
                kernel_size=1,
                bias=self.use_bias,
                norm_cfg=norm_cfg,
                conv_cfg=conv_cfg,
                act_cfg=None)
            
            output_conv = ConvModule(
                feat_channels,
                feat_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=self.use_bias,
                norm_cfg=norm_cfg,
                conv_cfg=conv_cfg,
                act_cfg=act_cfg)
            
            self.lateral_convs.append(lateral_conv)
            self.output_convs.append(output_conv)

        self.mask_feature = Conv3d(
            feat_channels, out_channels, kernel_size=1, stride=1, padding=0)

        self.num_outs = num_outs
        self.point_generator = MlvlPointGenerator(strides)
    
    def init_weights(self):
        """Initialize weights."""
        for i in range(0, self.num_encoder_levels):
            xavier_init(
                self.input_convs[i].conv,
                gain=1,
                bias=0,
                distribution='uniform')

        for i in range(0, self.num_input_levels - self.num_encoder_levels):
            caffe2_xavier_init(self.lateral_convs[i].conv, bias=0)
            caffe2_xavier_init(self.output_convs[i].conv, bias=0)

        caffe2_xavier_init(self.mask_feature, bias=0)
        normal_init(self.level_encoding, mean=0, std=1)
        for p in self.encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

        for layer in self.encoder.layers:
            for attn in layer.attentions:
                if isinstance(attn, MultiScaleDeformableAttention3D):
                    attn.init_weights()
    
    def forward(self, feats):
        batch_size = feats[0].shape[0]
        encoder_input_list = []
        padding_mask_list = []
        level_positional_encoding_list = []
        spatial_shapes = []
        reference_points_list = []
        
        # 按照从后向前的顺序，将第3、2、1层的特征图拼接起来，作为编码器的输入
        # 第0层的特征图作为FPN的输入
        for i in range(self.num_encoder_levels):
            # 从最后一层输入开始
            level_idx = self.num_input_levels - i - 1
            feat = feats[level_idx]
            # 对输入特征进行卷积，完成空间映射
            feat_projected = self.input_convs[i](feat)
            X, Y, Z = feat.shape[-3:]
            
            # no padding
            # 获得查询位置编码
            padding_mask_resized = feat.new_zeros((batch_size, ) + feat.shape[-3:], dtype=torch.bool)
            pos_embed = self.postional_encoding(padding_mask_resized)
            level_embed = self.level_encoding.weight[i]
            level_pos_embed = level_embed.view(1, -1, 1, 1, 1) + pos_embed
            
            # 生成参考点，作为MSDA的输入
            # (h_i * w_i * d_i, 2)
            reference_points = self.point_generator.single_level_grid_priors(
                feat.shape[-3:], level_idx, device=feat.device)
            
            # normalize points to [0, 1]
            factor = feat.new_tensor([[Z, Y, X]]) * self.strides[level_idx]
            reference_points = reference_points / factor
            
            # shape (batch_size, c, x_i, y_i, z_i) -> (x_i * y_i * z_i, batch_size, c)
            feat_projected = feat_projected.flatten(2).permute(2, 0, 1)
            level_pos_embed = level_pos_embed.flatten(2).permute(2, 0, 1)
            padding_mask_resized = padding_mask_resized.flatten(1)

            encoder_input_list.append(feat_projected)
            padding_mask_list.append(padding_mask_resized)
            level_positional_encoding_list.append(level_pos_embed)
            spatial_shapes.append(feat.shape[-3:])
            reference_points_list.append(reference_points)
        
        # shape (batch_size, total_num_query),
        # total_num_query=sum([., x_i * y_i * z_i, .])
        padding_masks = torch.cat(padding_mask_list, dim=1)
        # shape (total_num_query, batch_size, c)
        encoder_inputs = torch.cat(encoder_input_list, dim=0)
        level_positional_encodings = torch.cat(level_positional_encoding_list, dim=0)
        
        device = encoder_inputs.device
        # shape (num_encoder_levels, 2), from low
        # resolution to high resolution
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=device)
        # shape (0, h_0*w_0, h_0*w_0+h_1*w_1, ...)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        reference_points = torch.cat(reference_points_list, dim=0)
        reference_points = reference_points[None, :, None].repeat(
            batch_size, 1, self.num_encoder_levels, 1)
        valid_radios = reference_points.new_ones(
            (batch_size, self.num_encoder_levels, 2))
        
        # 下面通过六层的编码器来对输入查询进行编码，结构上与BEVFormer完全相同，唯一的区别是仅使用了自注意力机制，没有使用交叉注意力
        # 关于自注意力机制，需要重点关注几个输入：
        # 1. query: 输入特征图，来自对输入特征图的卷积
        # 2. query_pos: 查询位置编码，来自对输入特征图的位置编码
        # 3. reference_points: 参考点，来自输入特征图的网格点
        # shape (num_total_query, batch_size, c)
        memory = self.encoder(
            query=encoder_inputs,
            key=None,
            value=None,
            query_pos=level_positional_encodings,
            key_pos=None,
            attn_masks=None,
            key_padding_mask=None,
            query_key_padding_mask=padding_masks,
            spatial_shapes=spatial_shapes,
            reference_points=reference_points,
            level_start_index=level_start_index,
            valid_radios=valid_radios)
        
        # (num_total_query, batch_size, c) -> (batch_size, c, num_total_query)
        memory = memory.permute(1, 2, 0)

        # memory的内容由多层的输出复合而成，下面分解出每一层的输出

        # 首先计算出每一层的查询数量
        # from low resolution to high resolution
        num_query_per_level = [e[0] * e[1] * e[2] for e in spatial_shapes]

        # 将memory按照每一层的查询数量进行切分，返回一个列表，列表中的每个元素对应一层的输出
        # 每一层的形状为：(batch_size, c, num_total_query)
        outs = torch.split(memory, num_query_per_level, dim=-1)

        # 将outs列表中的每一层输出张量的形状reshape成：(batch_size, c, x_i, y_i, z_i)
        outs = [
            x.reshape(batch_size, -1, spatial_shapes[i][0],
                spatial_shapes[i][1], 
                spatial_shapes[i][2]) for i, x in enumerate(outs)]
        
        # 下面处理第0层的特征图，将其作为FPN的输入，最后输出掩码特征图
        # build FPN path
        # for i in range(0, -1, -1): 循环1次，i=0
        for i in range(self.num_input_levels - self.num_encoder_levels - 1, -1,
                       -1):
            # 取第0层的输入特征图
            x = feats[i]
            # 对输入特征图进行卷积，完成空间映射
            cur_feat = self.lateral_convs[i](x)
            
            # 对输出张量的最后一层进行上采样，使其与当前层的输入特征图尺寸相同
            # 然后将上采样后的输出张量与第0层的输入特征图相加，得到当前层的输出
            y = cur_feat + F.interpolate(
                outs[-1],
                size=cur_feat.shape[-3:],
                mode='trilinear',
                align_corners=False,
            )
            
            # 对当前层的输出进行卷积，得到最终的输出
            y = self.output_convs[i](y)
            # 将最终的输出添加到outs列表的最后
            outs.append(y)
        
        # 将最后一层的输出y通过卷积层，得到掩码特征图
        outs[-1] = self.mask_feature(outs[-1])
        # 将outs列表中的张量逆序排列（特征图从大到小），掩码特征图在最前面
        return outs[::-1]