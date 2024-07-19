import copy
from typing import Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from mmdet.models.detectors.base import BaseDetector
from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList
from mmdet.utils import InstanceList, OptConfigType, OptMultiConfig

from mmengine.runner import Runner
from mmengine.registry import RUNNERS
from mmengine.config import Config, DictAction


config = 'mmdet/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
cfg = Config.fromfile(config)
cfg.work_dir = 'workspace'
runner = RUNNERS.build(cfg)

print('what is the model looks like', runner.model)


