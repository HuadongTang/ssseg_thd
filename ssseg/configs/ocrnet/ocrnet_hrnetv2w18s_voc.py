'''ocrnet_hrnetv2w18s_voc'''
import copy
from .base_cfg import SEGMENTOR_CFG
from .._base_ import DATASET_CFG_VOCAUG_512x512, DATALOADER_CFG_BS16


# deepcopy
SEGMENTOR_CFG = copy.deepcopy(SEGMENTOR_CFG)
# modify dataset config
SEGMENTOR_CFG['dataset'] = DATASET_CFG_VOCAUG_512x512.copy()
# modify dataloader config
SEGMENTOR_CFG['dataloader'] = DATALOADER_CFG_BS16.copy()
# modify scheduler config
SEGMENTOR_CFG['scheduler']['max_epochs'] = 60
# modify other segmentor configs
SEGMENTOR_CFG['num_classes'] = 21
SEGMENTOR_CFG['backbone'] = {
    'type': 'HRNet', 'structure_type': 'hrnetv2_w18_small', 'arch': 'hrnetv2_w18_small', 'pretrained': True, 'selected_indices': (0, 0),
}
SEGMENTOR_CFG['work_dir'] = 'ocrnet_hrnetv2w18s_voc'
SEGMENTOR_CFG['logfilepath'] = 'ocrnet_hrnetv2w18s_voc/ocrnet_hrnetv2w18s_voc.log'
SEGMENTOR_CFG['resultsavepath'] = 'ocrnet_hrnetv2w18s_voc/ocrnet_hrnetv2w18s_voc_results.pkl'