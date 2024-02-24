'''memorynet_deeplabv3_resnet50os8_ade20k'''
import copy
from .base_cfg import SEGMENTOR_CFG
from .._base_ import DATASET_CFG_ADE20k_512x512, DATALOADER_CFG_BS16


# deepcopy
SEGMENTOR_CFG = copy.deepcopy(SEGMENTOR_CFG)
# modify dataset config
SEGMENTOR_CFG['dataset'] = DATASET_CFG_ADE20k_512x512.copy()
# modify dataloader config
SEGMENTOR_CFG['dataloader'] = DATALOADER_CFG_BS16.copy()
# modify scheduler config
SEGMENTOR_CFG['scheduler']['max_epochs'] = 130
# modify other segmentor configs
SEGMENTOR_CFG['num_classes'] = 150
SEGMENTOR_CFG['backbone'] = {
    'type': 'ResNet', 'depth': 50, 'structure_type': 'resnet50conv3x3stem',
    'pretrained': True, 'outstride': 8, 'use_conv3x3_stem': True, 'selected_indices': (0, 1, 2, 3),
}
# SEGMENTOR_CFG['head']['use_loss'] = False
SEGMENTOR_CFG['work_dir'] = 'memorynet_deeplabv3_resnet50os8_ade20k_deeplab_mem_seg64Tnewklin_womemloss_pearson'
SEGMENTOR_CFG['logfilepath'] = 'memorynet_deeplabv3_resnet50os8_ade20k_deeplab_mem_seg64Tnewklin_womemloss_pearson/memorynet_deeplabv3_resnet50os8_ade20k.log'
SEGMENTOR_CFG['resultsavepath'] = 'memorynet_deeplabv3_resnet50os8_ade20k_deeplab_mem_seg64Tnewklin_womemloss_pearson/memorynet_deeplabv3_resnet50os8_ade20k_results.pkl'
