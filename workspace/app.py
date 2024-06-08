from mmengine.utils import get_git_hash
from mmengine.utils.dl_utils import collect_env as collect_base_env

import mmdet

import torch
import torchvision
from torchvision.models import detection
from torchvision.models.detection import faster_rcnn

# from mmdet.models import build_detector
from mmdet.apis import init_detector
# from mmcv.utils import Config
# from mmcv.runner import Config

import argparse
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

from PIL import Image
import torchvision.transforms as transforms

from mmdet.apis import DetInferencer

def collect_env():
    """Collect the information of the running environments."""
    env_info = collect_base_env()
    env_info['MMDetection'] = f'{mmdet.__version__}+{get_git_hash()[:7]}'
    return env_info

for name, val in collect_env().items():
    print(f'{name}: {val}')


# Load configuration file
cfg = Config.fromfile('/home/exouser/mmdetection/workspace/20240405_073629/vis_data/config.py')

# Build the detector
# model = build_detector(cfg.model)

# Initialize the detector with the pre-trained weights
model = init_detector(cfg, '/home/exouser/mmdetection/workspace/epoch_12.pth', device='cuda:0')

# cfg = Config.fromfile('/home/exouser/mmdetection/workspace/20240405_073629/vis_data/config.py')

runner = Runner.from_cfg(cfg)

model = runner.model


# /home/exouser/data/kitti/train2017/000000.png use this image to test the model
# Define the transformation
transform = transforms.Compose([
    transforms.ToTensor()
])

# Load the image
image_path = '/home/exouser/data/kitti/train2017/000000.png'
image = Image.open(image_path)

# Apply the transformation to convert the image to a tensor
tensor_image = transform(image)


weights = '/home/exouser/mmdetection/workspace/epoch_12.pth'
state_dict = torch.load(weights)
model.load_state_dict(state_dict['state_dict'])

tesnsor_image1 = tensor_image.unsqueeze(0)
tesnsor_image1.shape

model.eval()
output = model(tesnsor_image1)
# _, predicted_label = torch.max(output, 1)
# print(predicted_label)
print(output)

device = 'cuda:0'

# Initialize the DetInferencer
inferencer = DetInferencer(cfg, weights, device)

# Use the detector to do inference
# img = './demo/demo.jpg'
# result = inferencer(image_path, out_dir='./00demo')
inferencer(image_path)['prediections']