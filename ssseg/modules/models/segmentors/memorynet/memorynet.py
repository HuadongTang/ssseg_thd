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
from ..base import SelfAttentionBlock

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
        # if head_cfg['downsample_backbone']['stride'] > 1:
        #     self.downsample_backbone = nn.Sequential(
        #         nn.Conv2d(head_cfg['in_channels'], head_cfg['in_channels'], **head_cfg['downsample_backbone']),
        #         BuildNormalization(placeholder=head_cfg['in_channels'], norm_cfg=norm_cfg),
        #         BuildActivation(act_cfg),
        #     )
        # context_within_image_cfg = head_cfg['context_within_image']
        # if context_within_image_cfg['is_on']:
        #     cwi_cfg = context_within_image_cfg['cfg']
        #     cwi_cfg.update({
        #         'in_channels': head_cfg['in_channels'],
        #         'out_channels': head_cfg['feats_channels'],
        #         'align_corners': align_corners,
        #         'norm_cfg': copy.deepcopy(norm_cfg),
        #         'act_cfg': copy.deepcopy(act_cfg),
        #     })
        #     supported_context_modules = {
        #         'aspp': ASPP,
        #         'ppm': PyramidPoolingModule,
        #     }
        #     self.context_within_image_module = supported_context_modules[context_within_image_cfg['type']](**cwi_cfg)
        self.bottleneck_ = nn.Sequential(
            nn.Conv2d(head_cfg['in_channels'], head_cfg['feats_channels'],
                      kernel_size=3, stride=1, padding=1, bias=False),
            BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg),
            BuildActivation(act_cfg),
        )

        self.bottleneck = nn.Sequential(
            nn.Conv2d(head_cfg['feats_channels'] * len(head_cfg['in_channels_list']), head_cfg['feats_channels'],
                      kernel_size=3, stride=1, padding=1, bias=False),
            BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg),
            BuildActivation(act_cfg),
        )
        # self.decoder_cls = nn.Sequential(
        #     nn.Conv2d(head_cfg['feats_channels'] + cfg['num_classes'], head_cfg['feats_channels'],
        #               kernel_size=3, stride=1, padding=1, bias=False),
        #     BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg),
        #     BuildActivation(act_cfg),
        #     nn.Dropout2d(head_cfg['dropout']),
        #     nn.Conv2d(head_cfg['feats_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0)
        # )

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
            nn.Conv2d(head_cfg['feats_channels'], head_cfg['feats_channels'], kernel_size=1, stride=1, padding=0,
                      bias=False),
            BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg),
            BuildActivation(act_cfg),
            nn.Dropout2d(head_cfg['dropout']),
            nn.Conv2d(head_cfg['feats_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0),
        )
        # self.decoder_stage2 = nn.Sequential(
        #     nn.Conv2d(head_cfg['out_channels'], head_cfg['out_channels'], kernel_size=1, stride=1, padding=0,
        #               bias=False),
        #     BuildNormalization(placeholder=head_cfg['out_channels'], norm_cfg=norm_cfg),
        #     BuildActivation(act_cfg),
        #     nn.Dropout2d(head_cfg['dropout']),
        #     nn.Conv2d(head_cfg['out_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0)
        # )
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
        for in_channels in [head_cfg['feats_channels'], ] * len(self.lateral_convs):
            self.fpn_convs.append(nn.Sequential(
                nn.Conv2d(in_channels, head_cfg['feats_channels'], kernel_size=3, stride=1, padding=1, bias=False),
                BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg),
                BuildActivation(act_cfg_copy),
            ))
        # build auxiliary decoder
        self.setauxiliarydecoder(cfg['auxiliary'])
        # freeze normalization layer if necessary
        if cfg.get('is_freeze_norm', False): self.freezenormalization()
        # layer names for training tricks
        self.layer_names = [
            'backbone_net', 'bottleneck', 'memory_module', 'decoder_stage1', 'decoder_stage2', 'norm_layers',
            'downsample_backbone', 'context_within_image_module', 'auxiliary_decoder'
        ]
        self.apd_proj = nn.Sequential(
            nn.Linear(head_cfg['feats_channels'] * 2, head_cfg['feats_channels'] // 2, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(head_cfg['feats_channels'] // 2, head_cfg['feats_channels']),
        )
        self.criterion = nn.CrossEntropyLoss(ignore_index=255)


    '''forward'''

    def forward(self, x, targets=None, **kwargs):
        img_size = x.size(2), x.size(3)
        seg_size = int(x.size(2) / 4), int(x.size(3) / 4)
        # feed to backbone network
        backbone_outputs = self.transforminputs(self.backbone_net(x),
                                                selected_indices=self.cfg['backbone'].get('selected_indices'))
        # if hasattr(self, 'norm_layers'):
        #     assert len(backbone_outputs) == len(self.norm_layers)
        #     for idx in range(len(backbone_outputs)):
        #         backbone_outputs[idx] = self.norm(backbone_outputs[idx], self.norm_layers[idx])
        # if self.cfg['head']['downsample_backbone']['stride'] > 1:
        #     for idx in range(len(backbone_outputs)):
        #         backbone_outputs[idx] = self.downsample_backbone(backbone_outputs[idx])
        # # feed to context within image module
        # feats_ms = self.context_within_image_module(backbone_outputs[-1]) if hasattr(self,
        #                                                                              'context_within_image_module') else None

        # feed to memory
        # memory_input = self.bottleneck_(backbone_outputs[-1])
        # feed to feats_to_pyramid_net
        # feed to pyramid pooling module
        ppm_out = self.ppm_net(backbone_outputs[-1])
        # apply fpn
        inputs = backbone_outputs[:-1]
        lateral_outputs = [lateral_conv(inputs[i]) for i, lateral_conv in enumerate(self.lateral_convs)]
        lateral_outputs.append(ppm_out)
        for i in range(len(lateral_outputs) - 1, 0, -1):
            prev_shape = lateral_outputs[i - 1].shape[2:]
            lateral_outputs[i - 1] = lateral_outputs[i - 1] + F.interpolate(lateral_outputs[i], size=prev_shape,
                                                                            mode='bilinear',
                                                                            align_corners=self.align_corners)
        fpn_outputs = [self.fpn_convs[i](lateral_outputs[i]) for i in range(len(lateral_outputs) - 1)]
        fpn_outputs.append(lateral_outputs[-1])
        fpn_outputs = [
            F.interpolate(out, size=fpn_outputs[0].size()[2:], mode='bilinear', align_corners=self.align_corners) for
            out in fpn_outputs]
        fpn_out = torch.cat(fpn_outputs, dim=1)
        memory_input = self.bottleneck(fpn_out)

        preds_stage1 = self.decoder_stage1(memory_input)

        # memory_in = F.interpolate(memory_input, size=feats_fpn.size()[2:], mode='bilinear', align_corners=self.align_corners)
        # pred = F.interpolate(preds_stage1, size=feats_fpn.size()[2:], mode='bilinear', align_corners=self.align_corners)
        # feats = self.decoder_feat(torch.cat([memory_in, feats_fpn], dim=1))
        stored_memory, memory_output = self.memory_module(memory_input, preds_stage1)

        # memory_output = torch.cat([feats_fpn, F.interpolate(memory_output, size=feats_fpn.size()[2:], mode='bilinear', align_corners=self.align_corners)], dim=1)
        # memory_output = self.decoder_cls(memory_output)
        # feed to decoder
        preds_stage2 = memory_output
        # preds_stage2 = self.decoder_stage2(memory_output)
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
            apd_pred = self.get_adaptive_perspective(x=memory_input, y=targets['seg_target'].unsqueeze(1),
                                                     new_proto=stored_memory.detach().squeeze(),
                                                     proto=stored_memory.squeeze())
            kl_loss = get_distill_loss(pred=memory_output, soft=apd_pred.detach(), target=targets['seg_target'])
            apd_pred = F.interpolate(apd_pred, size=img_size, mode='bilinear', align_corners=self.align_corners)
            pre_loss = self.criterion(apd_pred, targets['seg_target'].squeeze(1).long())
            outputs_dict.update({'loss_cls_stage1': preds_stage1, 'loss_cls_stage2': preds_stage2})
            with torch.no_grad():
                self.memory_module.update(
                    features=F.interpolate(memory_input, size=img_size, mode='bilinear',
                                           align_corners=self.align_corners),
                    segmentation=targets['seg_target'],
                    learning_rate=kwargs['learning_rate'],
                    **self.cfg['head']['update_cfg']
                )
            loss, losses_log_dict = self.calculatelosses(
                predictions=outputs_dict,
                targets=targets,
                losses_cfg=self.cfg['losses']
            )
            loss = loss + pre_loss + kl_loss
            pre_value =pre_loss.data.clone()
            kl_value = kl_loss.data.clone()
            dist.all_reduce(pre_value.div_(dist.get_world_size()))
            dist.all_reduce(kl_value.div_(dist.get_world_size()))
            losses_log_dict['pre_loss'] = pre_value
            losses_log_dict['kl_loss'] = kl_value
            total = losses_log_dict.pop('total') + losses_log_dict['kl_loss'] + losses_log_dict['pre_loss']
            # total = losses_log_dict.pop('total') + losses_log_dict['kl_loss']
            losses_log_dict['total'] = total
            if (kwargs['epoch'] > 1) and self.cfg['head']['use_loss']:
                loss_memory, loss_memory_log = self.calculatememoryloss(stored_memory)
                loss += loss_memory
                losses_log_dict['loss_memory'] = loss_memory_log
                total = losses_log_dict.pop('total') + losses_log_dict['loss_memory']
                losses_log_dict['total'] = total
            return loss, losses_log_dict
        return preds_stage2

    def get_adaptive_perspective(self, x, y, new_proto, proto):
        raw_x = x.clone()
        # y: [b, h, w]
        # x: [b, c, h, w]
        b, c, h, w = x.shape[:]
        y = F.interpolate(y.float(), size=(h, w), mode='nearest')  # b, 1, h, w
        unique_y = list(y.unique())
        if 255 in unique_y:
            unique_y.remove(255)
        # new_proto = self.conv_seg[1].weight.detach().data.squeeze() # [cls, 512]
        tobe_align = []
        label_list = []
        for tmp_y in unique_y:
            tmp_mask = (y == tmp_y).float()
            tmp_proto = (x * tmp_mask).sum(0).sum(-1).sum(-1) / (tmp_mask.sum(0).sum(-1).sum(-1) + 1e-12)
            onehot_vec = torch.zeros(new_proto.shape[0], 1).cuda()  # cls, 1
            onehot_vec[tmp_y.long()] = 1
            new_proto = new_proto * (1 - onehot_vec) + tmp_proto.unsqueeze(0) * onehot_vec
            tobe_align.append(tmp_proto.unsqueeze(0))
            label_list.append(tmp_y)
            # new_proto: [cls, 512]

        new_proto = torch.cat([new_proto, proto], -1)
        new_proto = self.apd_proj(new_proto)
        new_proto = new_proto.unsqueeze(-1).unsqueeze(-1)  # cls, 512, 1, 1
        new_proto = F.normalize(new_proto, 2, 1)
        raw_x = F.normalize(raw_x, 2, 1)
        pred = F.conv2d(raw_x, weight=new_proto) * 15
        return pred

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
        preds_memory = self.decoder_stage1(stored_memory)
        target = torch.range(0, num_classes - 1).type_as(stored_memory).long()
        target = target.unsqueeze(1).repeat(1, num_feats_per_cls).view(-1)
        loss_memory = self.calculateloss(preds_memory, target, self.cfg['head']['loss_cfg'])
        if dist.is_available() and dist.is_initialized():
            value = loss_memory.data.clone()
            dist.all_reduce(value.div_(dist.get_world_size()))
        else:
            value = torch.Tensor([loss_memory.item()]).type_as(stored_memory)
        return loss_memory, value


def cosine_similarity(x, y, eps=1e-8):
    return (x * y).sum(1) / (x.norm(dim=1) * y.norm(dim=1) + eps)


def pearson_correlation(x, y, eps=1e-8):
    return cosine_similarity(x - x.mean(1).unsqueeze(1), y - y.mean(1).unsqueeze(1), eps)


def inter_class_relation(y_s, y_t):
    return 1 - pearson_correlation(y_s, y_t).mean()


def intra_class_relation(y_s, y_t):
    return inter_class_relation(y_s.transpose(0, 1), y_t.transpose(0, 1))


def get_distill_loss(pred, soft, target, smoothness=0.5, eps=0):
    '''
    knowledge distillation loss
    '''
    b, c, h, w = soft.shape[:]
    soft.detach()
    target = F.interpolate(target.unsqueeze(1).float(), size=pred.shape[-2:], mode='nearest').squeeze(1).long()
    onehot = target.view(-1, 1)  # bhw, 1
    ignore_mask = (onehot == 255).float()
    onehot = onehot * (1 - ignore_mask)
    onehot = torch.zeros(b * h * w, c).cuda().scatter_(1, onehot.long(), 1)  # bhw, n
    onehot = onehot.contiguous().view(b, h, w, c).permute(0, 3, 1, 2)  # b, n, h, w
    # onehot = onehot.reshape(b*c, h*w)
    sm_soft = F.softmax(soft / 1, 1)
    # sm_soft = F.softmax(soft.view(-1,w*h)/4.0, dim=1)
    smoothed_label = smoothness * sm_soft + (1 - smoothness) * onehot
    if eps > 0:
        smoothed_label = smoothed_label * (1 - eps) + (1 - smoothed_label) * eps / (smoothed_label.shape[1] - 1)

        # inter_loss = inter_class_relation(F.log_softmax(pred, dim=1), smoothed_label)
    # intra_loss = intra_class_relation(F.log_softmax(pred, dim=1), smoothed_label)
    # loss = intra_loss + inter_loss
    loss = torch.mul(-1 * F.log_softmax(pred, dim=1), smoothed_label)  # b, n, h, w
    # loss = torch.mul(-1 * F.log_softmax(pred.view(-1,w*h)/4.0, dim=1), smoothed_label)

    sm_soft = F.softmax(soft / 1, 1)  # b, c, h, w
    # sm_soft = F.softmax(soft.view(-1,w*h)/4.0, dim=1)
    entropy_mask = -1 * (sm_soft * torch.log(sm_soft + 1e-12)).sum(1)
    loss = loss.sum(1)

    ### for class-wise entropy estimation
    unique_classes = list(target.unique())
    if 255 in unique_classes:
        unique_classes.remove(255)
    valid_mask = (target != 255).float()
    entropy_mask = entropy_mask * valid_mask
    loss_list = []
    weight_list = []
    for tmp_y in unique_classes:
        tmp_mask = (target == tmp_y).float()
        tmp_entropy_mask = entropy_mask * tmp_mask
        class_weight = 1
        tmp_loss = (loss * tmp_entropy_mask).sum() / (tmp_entropy_mask.sum() + 1e-12)
        loss_list.append(class_weight * tmp_loss)
        weight_list.append(class_weight)
    if len(weight_list) > 0:
        loss = sum(loss_list) / (sum(weight_list) + 1e-12)
    else:
        loss = torch.zeros(1).cuda().mean()
    return loss
