# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class LabelmeFacadeDataset(CustomDataset):
    """LabelmeFacadeDataset dataset.

    In segmentation map annotation for STARE, 0 stands for background, which is
    included in 2 categories. ``reduce_zero_label`` is fixed to False. The
    ``img_suffix`` is fixed to '.png' and ``seg_map_suffix`` is fixed to
    '.ah.png'.
    """

    CLASSES = ('various', 'building', 'car', 'door', 'pavement', 'road', 
      'sky', 'vegetation', 'window')

    PALETTE = [[0, 0, 0], [128, 0, 0], [128, 0, 128], [128, 128, 0], [128, 128, 128], 
      [128, 64, 0], [0, 128, 128], [0, 128, 0], [0, 0, 128]]

    def __init__(self, **kwargs):
        super(LabelmeFacadeDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)
        assert osp.exists(self.img_dir)
