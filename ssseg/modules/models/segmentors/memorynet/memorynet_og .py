'''
Function:
    Implementation of "Mining Contextual Information Beyond Image for Semantic Segmentation"
Author:
    Zhenchao Jin
'''
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from ..deeplabv3 import ASPP
from ..base import BaseSegmentor
from .memory import FeaturesMemory
from ..pspnet import PyramidPoolingModule
from ...backbones import BuildActivation, BuildNormalization
import numpy as np
# class ConvBNReLU(nn.Module):
#     def __init__(self,
#                  in_channels,
#                  out_channels,
#                  kernel_size,
#                  padding='same',
#                  **kwargs):
#         super().__init__()
#
#         self._conv = nn.Conv2d(
#             in_channels, out_channels, kernel_size, padding=padding, **kwargs)
#         # self._batch_norm = nn.BatchNorm2d(out_channels).cuda()
#         self._batch_norm = nn.SyncBatchNorm(out_channels)
#         self._relu = nn.ReLU()
#
#     def forward(self, x):
#         x = self._conv(x)
#         x = self._batch_norm(x)
#         x = self._relu(x)
#         return x
# class AggregationModule(nn.Module):
#     """Aggregation Module"""
#
#     def __init__(
#             self,
#             in_channels,
#             out_channels,
#             kernel_size,
#     ):
#         super(AggregationModule, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         padding = kernel_size // 2
#
#         self.reduce_conv = ConvBNReLU(
#             in_channels,
#             out_channels,
#             kernel_size=3,
#             padding=1,
#         )
#
#         self.t1 = nn.Conv2d(
#             out_channels,
#             out_channels,
#             kernel_size=(kernel_size, 1),
#             padding=(padding, 0),
#             groups=out_channels,
#         )
#         self.t2 = nn.Conv2d(
#             out_channels,
#             out_channels,
#             kernel_size=(1, kernel_size),
#             padding=(0, padding),
#             groups=out_channels,
#         )
#
#         self.p1 = nn.Conv2d(
#             out_channels,
#             out_channels,
#             kernel_size=(1, kernel_size),
#             padding=(0, padding),
#             groups=out_channels,
#         )
#         self.p2 = nn.Conv2d(
#             out_channels,
#             out_channels,
#             kernel_size=(kernel_size, 1),
#             padding=(padding, 0),
#             groups=out_channels,
#         )
#         self.norm = nn.SyncBatchNorm(out_channels)
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         """Forward function."""
#         x = self.reduce_conv(x)
#         x1 = self.t1(x)
#         x1 = self.t2(x1)
#
#         x2 = self.p1(x)
#         x2 = self.p2(x2)
#
#         out = self.relu(self.norm(x1 + x2))
#         return out

'''MemoryNet'''
class MemoryNet(BaseSegmentor):
    def __init__(self, cfg, mode):
        super(MemoryNet, self).__init__(cfg, mode)
        align_corners, norm_cfg, act_cfg, head_cfg = self.align_corners, self.norm_cfg, self.act_cfg, cfg['head']
        # build norm layer
        if 'norm_cfg' in head_cfg:
            self.norm_layers = nn.ModuleList()
            for in_channels in head_cfg['norm_cfg']['in_channels_list']:
                norm_cfg_copy = head_cfg['norm_cfg'].copy()
                norm_cfg_copy.pop('in_channels_list')
                norm_layer = BuildNormalization(placeholder=in_channels, norm_cfg=norm_cfg_copy)
                self.norm_layers.append(norm_layer)
        # build memory
        if head_cfg['downsample_backbone']['stride'] > 1:
            self.downsample_backbone = nn.Sequential(
                nn.Conv2d(head_cfg['in_channels'], head_cfg['in_channels'], **head_cfg['downsample_backbone']),
                BuildNormalization(placeholder=head_cfg['in_channels'], norm_cfg=norm_cfg),
                BuildActivation(act_cfg),
            )
        context_within_image_cfg = head_cfg['context_within_image']
        if context_within_image_cfg['is_on']:
            cwi_cfg = context_within_image_cfg['cfg']
            cwi_cfg.update({
                'in_channels': head_cfg['in_channels'],
                'out_channels': head_cfg['feats_channels'],
                'align_corners': align_corners,
                'norm_cfg': copy.deepcopy(norm_cfg),
                'act_cfg': copy.deepcopy(act_cfg),
            })
            supported_context_modules = {
                'aspp': ASPP,
                'ppm': PyramidPoolingModule,
            }
            self.context_within_image_module = supported_context_modules[context_within_image_cfg['type']](**cwi_cfg)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(head_cfg['in_channels'], head_cfg['feats_channels'], kernel_size=3, stride=1, padding=1, bias=False),
            BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg),
            BuildActivation(act_cfg),
        )
        self.memory_module = FeaturesMemory(
            num_classes=cfg['num_classes'], 
            feats_channels=head_cfg['feats_channels'], 
            transform_channels=head_cfg['transform_channels'],
            num_feats_per_cls=head_cfg['num_feats_per_cls'],
            out_channels=head_cfg['out_channels'],
            # use_context_within_image=context_within_image_cfg['is_on'],
            use_hard_aggregate=head_cfg['use_hard_aggregate'],
            norm_cfg=copy.deepcopy(norm_cfg),
            act_cfg=copy.deepcopy(act_cfg),
        )
        # build decoder
        self.decoder_stage1 = nn.Sequential(
            nn.Conv2d(head_cfg['feats_channels'], head_cfg['feats_channels'], kernel_size=1, stride=1, padding=0, bias=False),
            BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg),
            BuildActivation(act_cfg),
            nn.Dropout2d(head_cfg['dropout']),
            nn.Conv2d(head_cfg['feats_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0),
        )
        self.decoder_stage2 = nn.Sequential(
            nn.Conv2d(head_cfg['out_channels'], head_cfg['out_channels'], kernel_size=1, stride=1, padding=0, bias=False),
            BuildNormalization(placeholder=head_cfg['out_channels'], norm_cfg=norm_cfg),
            BuildActivation(act_cfg),
            nn.Dropout2d(head_cfg['dropout']),
            nn.Conv2d(head_cfg['out_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0)
        )
        # build auxiliary decoder
        self.setauxiliarydecoder(cfg['auxiliary'])
        # freeze normalization layer if necessary
        if cfg.get('is_freeze_norm', False): self.freezenormalization()
        # layer names for training tricks
        self.layer_names = [
            'backbone_net', 'bottleneck', 'memory_module', 'decoder_stage1', 'decoder_stage2', 'norm_layers',
            'downsample_backbone', 'context_within_image_module', 'auxiliary_decoder'
        ]
    '''forward'''
    def forward(self, x, targets=None, **kwargs):
        img_size = x.size(2), x.size(3)
        seg_size = int(x.size(2)/8), int(x.size(3)/8)
        # feed to backbone network
        backbone_outputs = self.transforminputs(self.backbone_net(x), selected_indices=self.cfg['backbone'].get('selected_indices'))
        if hasattr(self, 'norm_layers'):
            assert len(backbone_outputs) == len(self.norm_layers)
            for idx in range(len(backbone_outputs)):
                backbone_outputs[idx] = self.norm(backbone_outputs[idx], self.norm_layers[idx])
        if self.cfg['head']['downsample_backbone']['stride'] > 1:
            for idx in range(len(backbone_outputs)):
                backbone_outputs[idx] = self.downsample_backbone(backbone_outputs[idx])
        # feed to context within image module
        feats_ms = self.context_within_image_module(backbone_outputs[-1]) if hasattr(self, 'context_within_image_module') else None

        # feed to memory
        memory_input = self.bottleneck(backbone_outputs[-1])
        # memory_input = feats_ms

        preds_stage1 = self.decoder_stage1(memory_input)
        stored_memory, memory_output = self.memory_module(memory_input, preds_stage1, feats_ms)
        # feed to decoder
        preds_stage2 = memory_output
        preds_stage2 = self.decoder_stage2(memory_output)
        # forward according to the mode
        if self.mode == 'TRAIN':
            outputs_dict = self.forwardtrain(
                predictions=preds_stage2,
                targets=targets,
                backbone_outputs=backbone_outputs,
                losses_cfg=self.cfg['losses'],
                img_size=img_size,
                compute_loss=False,
            )
            preds_stage2 = outputs_dict.pop('loss_cls')
            preds_stage1 = F.interpolate(preds_stage1, size=img_size, mode='bilinear', align_corners=self.align_corners)
            outputs_dict.update({'loss_cls_stage1': preds_stage1, 'loss_cls_stage2': preds_stage2})
            with torch.no_grad():
                # self.memory_module.update(
                #     features=F.interpolate(memory_input, size=img_size, mode='bilinear', align_corners=self.align_corners), 
                #     segmentation=targets['seg_target'],
                #     learning_rate=kwargs['learning_rate'],
                #     **self.cfg['head']['update_cfg']
                # )
                self.memory_module.update(
                    features=memory_input,
                    segmentation=F.interpolate(targets['seg_target'].unsqueeze(1), size=seg_size, mode='nearest'),
                    learning_rate=kwargs['learning_rate'],
                    **self.cfg['head']['update_cfg']
                )
            loss, losses_log_dict = self.calculatelosses(
                predictions=outputs_dict, 
                targets=targets, 
                losses_cfg=self.cfg['losses']
            )
            if (kwargs['epoch'] > 1) and self.cfg['head']['use_loss']:
                loss_memory, loss_memory_log = self.calculatememoryloss(stored_memory)
                loss += loss_memory
                losses_log_dict['loss_memory'] = loss_memory_log
                total = losses_log_dict.pop('total') + losses_log_dict['loss_memory']
                losses_log_dict['total'] = total
            return loss, losses_log_dict
        return preds_stage2


    '''norm layer'''
    def norm(self, x, norm_layer):
        n, c, h, w = x.shape
        x = x.reshape(n, c, h * w).transpose(2, 1).contiguous()
        x = norm_layer(x)
        x = x.transpose(1, 2).reshape(n, c, h, w).contiguous()
        return x
    '''calculate memory loss'''
    def calculatememoryloss(self, stored_memory):
        num_classes, num_feats_per_cls, feats_channels = stored_memory.size()
        stored_memory = stored_memory.reshape(num_classes * num_feats_per_cls, feats_channels, 1, 1)
        preds_memory = self.decoder_stage2(stored_memory)
        target = torch.range(0, num_classes - 1).type_as(stored_memory).long()
        target = target.unsqueeze(1).repeat(1, num_feats_per_cls).view(-1)
        loss_memory = self.calculateloss(preds_memory, target, self.cfg['head']['loss_cfg'])
        if dist.is_available() and dist.is_initialized():
            value = loss_memory.data.clone()
            dist.all_reduce(value.div_(dist.get_world_size()))
        else:
            value = torch.Tensor([loss_memory.item()]).type_as(stored_memory)
        return loss_memory, value