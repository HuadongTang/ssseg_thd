'''
Function:
    Implementation of UPerNet
Author:
    Zhenchao Jin
'''
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base import BaseSegmentor
from ..pspnet import PyramidPoolingModule
from ...backbones import BuildActivation, BuildNormalization
import numpy as np
from .objectcontext import ObjectContextBlock
from .spatialgather import SpatialGatherModule
class ConvBNReLU(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding='same',
                 **kwargs):
        super().__init__()

        self._conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=padding, **kwargs)
        # self._batch_norm = nn.BatchNorm2d(out_channels).cuda()
        self._batch_norm = nn.SyncBatchNorm(out_channels)
        self._relu = nn.ReLU()

    def forward(self, x):
        x = self._conv(x)
        x = self._batch_norm(x)
        x = self._relu(x)
        return x
class AggregationModule(nn.Module):
    """Aggregation Module"""

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
    ):
        super(AggregationModule, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        padding = kernel_size // 2

        self.reduce_conv = ConvBNReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )

        self.t1 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(padding, 0),
            groups=out_channels,
        )
        self.t2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=(1, kernel_size),
            padding=(0, padding),
            groups=out_channels,
        )

        self.p1 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=(1, kernel_size),
            padding=(0, padding),
            groups=out_channels,
        )
        self.p2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(padding, 0),
            groups=out_channels,
        )
        self.norm = nn.SyncBatchNorm(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        """Forward function."""
        x = self.reduce_conv(x)
        x1 = self.t1(x)
        x1 = self.t2(x1)

        x2 = self.p1(x)
        x2 = self.p2(x2)

        out = self.relu(self.norm(x1 + x2))
        return out

'''UPerNet'''
class UPerNet(BaseSegmentor):
    def __init__(self, cfg, mode):
        super(UPerNet, self).__init__(cfg, mode)
        align_corners, norm_cfg, act_cfg, head_cfg = self.align_corners, self.norm_cfg, self.act_cfg, cfg['head']
        # build feature2pyramid
        if 'feature2pyramid' in head_cfg:
            from ..base import Feature2Pyramid
            head_cfg['feature2pyramid']['norm_cfg'] = norm_cfg.copy()
            self.feats_to_pyramid_net = Feature2Pyramid(**head_cfg['feature2pyramid'])
        # build pyramid pooling module
        ppm_cfg = {
            'in_channels': head_cfg['in_channels_list'][-1],
            'out_channels': head_cfg['feats_channels'],
            'pool_scales': head_cfg['pool_scales'],
            'align_corners': align_corners,
            'norm_cfg': copy.deepcopy(norm_cfg),
            'act_cfg': copy.deepcopy(act_cfg),
        }
        self.ppm_net = PyramidPoolingModule(**ppm_cfg)
        # build lateral convs
        act_cfg_copy = copy.deepcopy(act_cfg)
        if 'inplace' in act_cfg_copy: act_cfg_copy['inplace'] = False
        self.lateral_convs = nn.ModuleList()
        for in_channels in head_cfg['in_channels_list'][:-1]:
            self.lateral_convs.append(nn.Sequential(
                nn.Conv2d(in_channels, head_cfg['feats_channels'], kernel_size=1, stride=1, padding=0, bias=False),
                BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg),
                BuildActivation(act_cfg_copy),
            ))
        # build fpn convs
        self.fpn_convs = nn.ModuleList()
        for in_channels in [head_cfg['feats_channels'],] * len(self.lateral_convs):
            self.fpn_convs.append(nn.Sequential(
                nn.Conv2d(in_channels, head_cfg['feats_channels'], kernel_size=3, stride=1, padding=1, bias=False),
                BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg),
                BuildActivation(act_cfg_copy),
            ))
        # build decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(2560, head_cfg['feats_channels'], kernel_size=3, stride=1, padding=1, bias=False),
            BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg),
            BuildActivation(act_cfg),
            nn.Dropout2d(head_cfg['dropout']),
            nn.Conv2d(head_cfg['feats_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0)
        )
        # build auxiliary decoder
        self.setauxiliarydecoder(cfg['auxiliary'])
        # freeze normalization layer if necessary
        if cfg.get('is_freeze_norm', False): self.freezenormalization()
        # layer names for training tricks
        self.layer_names = ['backbone_net', 'ppm_net', 'lateral_convs', 'feats_to_pyramid_net', 'decoder', 'auxiliary_decoder']

        self.query_conv = nn.Conv2d(512, 64, kernel_size=1)
        self.key_conv = nn.Conv2d(512, 64, kernel_size=1)
        self.aggregation = AggregationModule(1024, 512, 31)
        self.intra_conv = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
            BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg),
            BuildActivation(act_cfg),

        )
        self.inter_conv = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
            BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg),
            BuildActivation(act_cfg),
        )
        self.intra_feats_conv = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0),
            BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg),
            BuildActivation(act_cfg),
        )
        self.inter_feats_conv = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0),
            BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg),
            BuildActivation(act_cfg),
        )
        self.bottleneck_final = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding=0),
            BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg),
            BuildActivation(act_cfg),
        )
        # build spatial gather module
        spatialgather_cfg = {
            'scale': 1
        }
        self.spatial_gather_module = SpatialGatherModule(**spatialgather_cfg)
        # build object context block
        self.object_context_block = ObjectContextBlock(
            in_channels=512,
            transform_channels=256,
            scale=1,
            align_corners=align_corners,
            norm_cfg=copy.deepcopy(norm_cfg),
            act_cfg=copy.deepcopy(act_cfg),
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1,
                      bias=False),
            BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg),
            BuildActivation(act_cfg),
        )
    '''forward'''
    def forward(self, x, targets=None):
        img_size = x.size(2), x.size(3)
        # feed to backbone network
        backbone_outputs = self.transforminputs(self.backbone_net(x), selected_indices=self.cfg['backbone'].get('selected_indices'))
        predictions_aux = self.auxiliary_decoder(backbone_outputs[-2])
        feats = F.interpolate(backbone_outputs[-1], size=backbone_outputs[-2].size()[2:], mode='bilinear', align_corners=self.align_corners)
        value = self.aggregation(feats)
        batch_size, channels, height, width = value.size()
        proj_query = self.query_conv(value).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(value).view(batch_size, -1, width * height)
        context_prior_map = torch.bmm(proj_query, proj_key)
        context_prior_map = context_prior_map.permute(0, 2, 1)
        context_prior_map = torch.sigmoid(context_prior_map)
        inter_context_prior_map = 1 - context_prior_map
        value = value.view(batch_size, 512, -1)
        value = value.permute(0, 2, 1)
        intra_context = torch.bmm(context_prior_map, value)
        intra_context = intra_context.div(np.prod([height, width]))
        intra_context = intra_context.permute(0, 2, 1).contiguous()
        intra_context = intra_context.view(batch_size, 512,
                                           int(height),
                                           int(width))
        intra_context = self.intra_conv(intra_context)

        inter_context = torch.bmm(inter_context_prior_map, value)
        inter_context = inter_context.div(np.prod([height, width]))
        inter_context = inter_context.permute(0, 2, 1).contiguous()
        inter_context = inter_context.view(batch_size, 512,
                                           int(height),
                                           int(width))
        inter_context = self.inter_conv(inter_context)
        feats_ = self.bottleneck(feats)

        # feed to ocr module
        context = self.spatial_gather_module(feats_, predictions_aux)
        intra_feats = self.intra_feats_conv(torch.cat([feats_, intra_context], dim=1))
        inter_feats = self.inter_feats_conv(torch.cat([feats_, inter_context], dim=1))
        intra_object_context = self.object_context_block(intra_feats, context)
        inter_object_context = self.object_context_block(inter_feats, context)

        # feed to feats_to_pyramid_net
        if hasattr(self, 'feats_to_pyramid_net'): backbone_outputs = self.feats_to_pyramid_net(backbone_outputs)
        # feed to pyramid pooling module
        ppm_out = self.ppm_net(backbone_outputs[-1])
        # apply fpn
        inputs = backbone_outputs[:-1]
        lateral_outputs = [lateral_conv(inputs[i]) for i, lateral_conv in enumerate(self.lateral_convs)]
        lateral_outputs.append(ppm_out)
        for i in range(len(lateral_outputs) - 1, 0, -1):
            prev_shape = lateral_outputs[i - 1].shape[2:]
            lateral_outputs[i - 1] = lateral_outputs[i - 1] + F.interpolate(lateral_outputs[i], size=prev_shape, mode='bilinear', align_corners=self.align_corners)
        fpn_outputs = [self.fpn_convs[i](lateral_outputs[i]) for i in range(len(lateral_outputs) - 1)]
        fpn_outputs.append(lateral_outputs[-1])
        fpn_outputs = [F.interpolate(out, size=fpn_outputs[0].size()[2:], mode='bilinear', align_corners=self.align_corners) for out in fpn_outputs]
        fpn_out = torch.cat(fpn_outputs, dim=1)
        output = self.bottleneck_final(torch.cat([feats, intra_object_context, inter_object_context], dim=1))
        output = F.interpolate(output, size=fpn_outputs[0].size()[2:], mode='bilinear', align_corners=self.align_corners)
        out = torch.cat([output, fpn_out ], dim=1)
        # feed to decoder
        predictions = self.decoder(out)
        # forward according to the mode
        if self.mode == 'TRAIN':
            predictions = F.interpolate(predictions, size=img_size, mode='bilinear', align_corners=self.align_corners)
            predictions_aux = F.interpolate(predictions_aux, size=img_size, mode='bilinear',
                                            align_corners=self.align_corners)
            return self.calculatelosses(
                predictions={'loss_cls': predictions, 'loss_aux': predictions_aux, 'loss_aff': context_prior_map},
                targets=targets,
                losses_cfg=self.cfg['losses']
            )

        return predictions