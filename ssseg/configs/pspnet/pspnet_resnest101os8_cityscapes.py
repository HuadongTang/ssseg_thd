'''pspnet_resnest101os8_cityscapes'''
import copy
from .base_cfg import SEGMENTOR_CFG
from .._base_ import DATASET_CFG_CITYSCAPES_512x1024, DATALOADER_CFG_BS8


# deepcopy
SEGMENTOR_CFG = copy.deepcopy(SEGMENTOR_CFG)
# modify dataset config
SEGMENTOR_CFG['dataset'] = DATASET_CFG_CITYSCAPES_512x1024.copy()
# modify dataloader config
SEGMENTOR_CFG['dataloader'] = DATALOADER_CFG_BS8.copy()
# modify scheduler config
SEGMENTOR_CFG['scheduler']['max_epochs'] = 220
# modify other segmentor configs
SEGMENTOR_CFG['num_classes'] = 19
SEGMENTOR_CFG['backbone'] = {
    'type': 'ResNeSt', 'depth': 101, 'structure_type': 'resnest101', 'pretrained': True, 'outstride': 8, 'selected_indices': (2, 3),
}
SEGMENTOR_CFG['work_dir'] = 'pspnet_resnest101os8_cityscapes'
SEGMENTOR_CFG['logfilepath'] = 'pspnet_resnest101os8_cityscapes/pspnet_resnest101os8_cityscapes.log'
SEGMENTOR_CFG['resultsavepath'] = 'pspnet_resnest101os8_cityscapes/pspnet_resnest101os8_cityscapes_results.pkl'