import torch
import torch.nn as nn
import torch.nn.functional as F

class AffinityLoss(nn.Module):
    def __init__(self, scale_factor=1.0, weight=None, ignore_index=255, reduction='mean', loss_weight=1.0):
        super(AffinityLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.num_classes = 182

    def _construct_ideal_affinity_matrix(self, label, label_size):
        label = torch.unsqueeze(label, axis=1)
        scaled_labels = F.interpolate(
            label.float(), size=label_size, mode="nearest")
        # print('scaled_labels1', scaled_labels.size())

        scaled_labels = scaled_labels.squeeze_().long()
        # print('scaled_labels2', scaled_labels.size())
        # astype('int64')
        scaled_labels[scaled_labels == 255] = self.num_classes
        # scaled_labels = scaled_labels.astype('int64')
        one_hot_labels = F.one_hot(scaled_labels, self.num_classes + 1)

        one_hot_labels = one_hot_labels.view((one_hot_labels.shape[0],
                                                 -1, self.num_classes + 1)).float()

        ideal_affinity_matrix = torch.bmm(one_hot_labels,
                                           one_hot_labels.permute(0, 2, 1))
        return ideal_affinity_matrix

    def forward(self, prediction, target):
        cls_score = prediction
        label = target
        _, _, cls_size = cls_score.size()
        _, label_h, label_w = label.size()
        # print('cls', cls_score.size())
        # print('label', label.size())
        # print(cls_score.shape[2:])
        ideal_affinity_matrix = self._construct_ideal_affinity_matrix(label, [int(label_h/8),int(label_w/8)])
        # ideal_affinity_matrix = self._construct_ideal_affinity_matrix(label, cls_score.shape[2:])s


        unary_term = F.binary_cross_entropy(cls_score, ideal_affinity_matrix)
        # unary_term = lovasz_hinge(cls_score, ideal_affinity_matrix)

        # diagonal_matrix = (1 - torch.eye(ideal_affinity_matrix.shape[1])).to(ideal_affinity_matrix.get_device())
        # vtarget = diagonal_matrix * ideal_affinity_matrix
        # recall_part = torch.sum(cls_score * vtarget, dim=2)
        # denominator = torch.sum(vtarget, dim=2)
        # denominator = denominator.masked_fill_(~(denominator > 0), 1)
        # recall_part = recall_part.div_(denominator)
        # recall_label = torch.ones_like(recall_part)
        #
        # # recall_part[recall_part < 0.0] = 0.0
        # # recall_part[recall_part > 1.0] = 1.0
        # recall_loss = F.binary_cross_entropy(recall_part, recall_label)
        #
        # spec_part = torch.sum((1 - cls_score) * (1 - ideal_affinity_matrix), dim=2)
        # denominator = torch.sum(1 - ideal_affinity_matrix, dim=2)
        # denominator = denominator.masked_fill_(~(denominator > 0), 1)
        # spec_part = spec_part.div_(denominator)
        # spec_label = torch.ones_like(spec_part)
        # # spec_part[spec_part < 0.0] = 0.0
        # # spec_part[spec_part > 1.0] = 1.0
        # spec_loss = F.binary_cross_entropy(spec_part, spec_label)
        #
        # precision_part = torch.sum(cls_score * vtarget, dim=2)
        # denominator = torch.sum(cls_score, dim=2)
        # denominator = denominator.masked_fill_(~(denominator > 0), 1)
        # precision_part = precision_part.div_(denominator)
        # precision_label = torch.ones_like(precision_part)
        # # precision_part[precision_part < 0.0] = 0.0
        # # precision_part[precision_part > 1.0] = 1.0
        # precision_loss = F.binary_cross_entropy(precision_part, precision_label)
        # global_term = recall_loss + spec_loss + precision_loss
        # loss_cls = unary_term + global_term


        return unary_term