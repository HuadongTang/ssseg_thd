'''
Function:
    Implementation of FeaturesMemory
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from ..base import SelfAttentionBlock
from ...backbones import BuildActivation, BuildNormalization
from torch.nn import Softmax

'''FeaturesMemory'''
class FeaturesMemory(nn.Module):
    def __init__(self, num_classes, feats_channels, transform_channels, out_channels, use_context_within_image=True,
                 num_feats_per_cls=1, use_hard_aggregate=False, norm_cfg=None, act_cfg=None):
        super(FeaturesMemory, self).__init__()
        assert num_feats_per_cls > 0, 'num_feats_per_cls should be larger than 0'
        # set attributes
        self.num_classes = num_classes
        self.feats_channels = feats_channels
        self.transform_channels = transform_channels
        self.out_channels = out_channels
        self.num_feats_per_cls = num_feats_per_cls
        # self.use_context_within_image = use_context_within_image
        # self.use_hard_aggregate = use_hard_aggregate
        # init memory
        self.memory = nn.Parameter(torch.zeros(num_classes, num_feats_per_cls, feats_channels, dtype=torch.float), requires_grad=False)
        # self.query_conv = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1)
        # self.key_conv = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1)
        # self.value_conv = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1)
        # define self_attention module
        # if self.num_feats_per_cls > 1:
        #     self.self_attentions = nn.ModuleList()
        #     for _ in range(self.num_feats_per_cls):
        #         self_attention = SelfAttentionBlock(
        #             key_in_channels=feats_channels,
        #             query_in_channels=feats_channels,
        #             transform_channels=transform_channels,
        #             out_channels=feats_channels,
        #             share_key_query=False,
        #             query_downsample=None,
        #             key_downsample=None,
        #             key_query_num_convs=2,
        #             value_out_num_convs=1,
        #             key_query_norm=True,
        #             value_out_norm=True,
        #             matmul_norm=True,
        #             with_out_project=True,
        #             norm_cfg=norm_cfg,
        #             act_cfg=act_cfg,
        #         )
        #         self.self_attentions.append(self_attention)
        # self.fuse_memory_conv = nn.Sequential(
        #     nn.Conv2d(num_classes * self.num_feats_per_cls, num_classes, kernel_size=1, stride=1, padding=0, bias=False),
        #     BuildNormalization(placeholder=num_classes, norm_cfg=norm_cfg),
        #     BuildActivation(act_cfg),
        # )
        # else:
        #     self.self_attention = SelfAttentionBlock(
        #         key_in_channels=feats_channels,
        #         query_in_channels=feats_channels,
        #         transform_channels=transform_channels,
        #         out_channels=feats_channels,
        #         share_key_query=False,
        #         query_downsample=None,
        #         key_downsample=None,
        #         key_query_num_convs=2,
        #         value_out_num_convs=1,
        #         key_query_norm=True,
        #         value_out_norm=True,
        #         matmul_norm=True,
        #         with_out_project=True,
        #         norm_cfg=norm_cfg,
        #         act_cfg=act_cfg,
        #     )
        # whether need to fuse the contextual information within the input image
        # self.bottleneck = nn.Sequential(
        #     nn.Conv2d(feats_channels * 2, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
        #     BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg),
        #     BuildActivation(act_cfg),
        # )
        # if use_context_within_image:
        #     self.self_attention_ms = SelfAttentionBlock(
        #         key_in_channels=feats_channels,
        #         query_in_channels=feats_channels,
        #         transform_channels=transform_channels,
        #         out_channels=feats_channels,
        #         share_key_query=False,
        #         query_downsample=None,
        #         key_downsample=None,
        #         key_query_num_convs=2,
        #         value_out_num_convs=1,
        #         key_query_norm=True,
        #         value_out_norm=True,
        #         matmul_norm=True,
        #         with_out_project=True,
        #         norm_cfg=norm_cfg,
        #         act_cfg=act_cfg,
        #     )
        #     self.bottleneck_ms = nn.Sequential(
        #         nn.Conv2d(feats_channels * 2, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
        #         BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg),
        #         BuildActivation(act_cfg),
        #     )
        self.proj = nn.Sequential(
            nn.Linear(self.feats_channels * 2, self.feats_channels // 2, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.feats_channels // 2, self.feats_channels),
        )


    '''forward'''
    def forward(self, feats, preds=None, feats_ms=None):
        batch_size, num_channels, h, w = feats.size()
        for idx in range(self.num_feats_per_cls):
            memory = self.memory.data[:, idx, :]
            pred = self.post_refine_proto_v2(x=feats, pred=preds, proto=memory)

        return self.memory.data, pred

        # extract the history features
        # --(B, num_classes, H, W) --> (B*H*W, num_classes)
        # weight_cls = preds.permute(0, 2, 3, 1).contiguous()
        # weight_cls = weight_cls.reshape(-1, self.num_classes)
        # weight_cls = F.softmax(weight_cls, dim=-1)
        # if self.use_hard_aggregate:
        #     labels = weight_cls.argmax(-1).reshape(-1, 1)
        #     onehot = torch.zeros_like(weight_cls).scatter_(1, labels.long(), 1)
        #     weight_cls = onehot
        # # --(B*H*W, num_classes) * (num_classes, C) --> (B*H*W, C)
        # selected_memory_list = []
        # for idx in range(self.num_feats_per_cls):
        #     memory = self.memory.data[:, idx, :]
        #     selected_memory = torch.matmul(weight_cls, memory)
        #     selected_memory_list.append(selected_memory.unsqueeze(1))
        # # calculate selected_memory according to the num_feats_per_cls
        # if self.num_feats_per_cls > 1:
        #     relation_selected_memory_list = []
        #     for idx, selected_memory in enumerate(selected_memory_list):
        #         # --(B*H*W, C) --> (B, H, W, C)
        #         selected_memory = selected_memory.view(batch_size, h, w, num_channels)
        #         # --(B, H, W, C) --> (B, C, H, W)
        #         selected_memory = selected_memory.permute(0, 3, 1, 2).contiguous()
        #         # --append
        #         relation_selected_memory_list.append(self.self_attentions[idx](feats, selected_memory))
        #     # --concat
        #     selected_memory = torch.cat(relation_selected_memory_list, dim=1)
        #     selected_memory = self.fuse_memory_conv(selected_memory)
        # else:
        #     assert len(selected_memory_list) == 1
        #     selected_memory = selected_memory_list[0].squeeze(1)
        #     # --(B*H*W, C) --> (B, H, W, C)
        #     selected_memory = selected_memory.view(batch_size, h, w, num_channels)
        #     # --(B, H, W, C) --> (B, C, H, W)
        #     selected_memory = selected_memory.permute(0, 3, 1, 2).contiguous()
        #     mem_att_k = self.key_conv(selected_memory).view(batch_size, 64, h*w)
        #     mem_att_q = self.query_conv(selected_memory).view(batch_size, 64, h*w).permute(0, 2, 1)
        #     mem_att_v = self.value_conv(feats).view(batch_size, num_channels, h*w)
        #     energy = torch.bmm(mem_att_q, mem_att_k)
        #     attention = F.softmax(energy, dim=-1)
        #     selected_memory = torch.bmm(mem_att_v, attention.permute(0, 2, 1)).contiguous()
        #     selected_memory = selected_memory.reshape(batch_size, -1, h, w)
        #     # --feed into the self attention module
        #     # selected_memory = self.self_attention(feats, selected_memory)
        #     # selected_memory_ = self.self_attention(feats_, selected_memory)
        # # return
        # memory_output = self.bottleneck(torch.cat([feats, selected_memory], dim=1))
        # if self.use_context_within_image:
        #     feats_ms = self.self_attention_ms(feats, feats_ms)
        #     memory_output = self.bottleneck_ms(torch.cat([feats_ms, memory_output], dim=1))
        # return self.memory.data, memory_output

    def get_pred(self, x, proto):
        b, c, h, w = x.size()[:]
        if len(proto.shape[:]) == 3:
            # x: [b, c, h, w]
            # proto: [b, cls, c]
            cls_num = proto.size(1)
            x = x / torch.norm(x, 2, 1, True)
            proto = proto / torch.norm(proto, 2, -1, True)  # b, n, c
            x = x.contiguous().view(b, c, h*w)  # b, c, hw
            pred = proto @ x  # b, cls, hw
        elif len(proto.shape[:]) == 2:
            # x: [b, c, h, w]
            # proto: [cls, c]
            cls_num = proto.size(0)
            x = x / torch.norm(x, 2, 1, True)
            proto = proto / torch.norm(proto, 2, 1, True)
            x = x.contiguous().view(b, c, h*w)  # b, c, hw
            proto = proto.unsqueeze(0)  # 1, cls, c
            pred = proto @ x  # b, cls, hw
        pred = pred.contiguous().view(b, cls_num, h, w)
        return pred * 15

    def post_refine_proto_v2(self, x, pred, proto):
        raw_x = x.clone()
        b, c, h, w = raw_x.shape[:]
        pred = pred.view(b, proto.shape[0], h*w)
        pred = F.softmax(pred, 1)   # b, n, hw
        pred_proto = (pred @ raw_x.view(b, c, h*w).permute(0, 2, 1)) / (pred.sum(-1).unsqueeze(-1) + 1e-12)
        # proto = torch.amax(proto, dim=1)
        pred_proto = torch.cat([pred_proto, proto.unsqueeze(0).repeat(pred_proto.shape[0], 1, 1)], -1)  # b, n, 2c
        pred_proto = self.proj(pred_proto)
        new_pred = self.get_pred(raw_x, pred_proto)
        return new_pred
    '''update'''
    def update(self, features, segmentation, ignore_index=255, strategy='cosine_similarity', momentum_cfg=None, learning_rate=None):
        assert strategy in ['mean', 'cosine_similarity']
        batch_size, num_channels, h, w = features.size()
        momentum = momentum_cfg['base_momentum']
        if momentum_cfg['adjust_by_learning_rate']:
            momentum = momentum_cfg['base_momentum'] / momentum_cfg['base_lr'] * learning_rate
        # use features to update memory

        tempmask = segmentation.long()
        tempmask[tempmask == 255] = self.num_classes
        tempmask = F.one_hot(tempmask, num_classes=self.num_classes + 1)
        tempmask = tempmask.view(batch_size, -1, self.num_classes + 1)
        denominator = tempmask.sum(1).unsqueeze(dim=1)

        query =features.view(batch_size, num_channels, -1)
        nominator = torch.matmul(query, tempmask.float())

        nominator = torch.t(nominator.sum(0))  # batchwise sum
        denominator = denominator.sum(0)  # batchwise sum
        denominator = denominator.squeeze()

        for slot in range(self.num_classes):
            if denominator[slot] != 0:
                feats_cls = (1 - momentum) * self.memory[slot].data  + (momentum * nominator[slot] / denominator[slot])
                self.memory[slot].data.copy_(feats_cls)
        # syn the memory
        if dist.is_available() and dist.is_initialized():
            memory = self.memory.data.clone()
            dist.all_reduce(memory.div_(dist.get_world_size()))
            self.memory = nn.Parameter(memory, requires_grad=False)
