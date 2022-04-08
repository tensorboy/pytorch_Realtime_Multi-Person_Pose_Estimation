import os
import re
import sys
sys.path.append('.')
import cv2
import math
import time
import scipy
import argparse
import matplotlib
import numpy as np
import pylab as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from scipy.ndimage.morphology import generate_binary_structure
from scipy.ndimage.filters import gaussian_filter, maximum_filter

from lib.network.rtpose_vgg import get_model
from lib.network import im_transform
from evaluate.coco_eval import get_outputs, handle_paf_and_heat
from lib.utils.common import Human, BodyPart, CocoPart, CocoColors, CocoPairsRender, draw_humans
from lib.utils.paf_to_pose import paf_to_pose_cpp
from lib.config import cfg, update_config


NOW_WHAT = 'bean'
TRUNK = 'vgg19'
FROM_SCRATCH = True
img_name = 'subway.jpg'

# weight_path = 'pose_model.pth'
weight_path = 'network/weight/4.8/best_bean.pth'

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', help='experiment configure file name',
                    default='./experiments/vgg19_368x368_sgd.yaml', type=str)

parser.add_argument('--weight', type=str,
                    default=weight_path)

parser.add_argument('opts',
                    help="Modify config options using the command-line",
                    default=None,
                    nargs=argparse.REMAINDER)
args = parser.parse_args()

# update config file
update_config(cfg, args)

model = get_model(trunk=TRUNK, dataset=NOW_WHAT)
wts_dict = torch.load(args.weight)

# for model trained from scratch
if FROM_SCRATCH:
    wts_dict = {k.replace('module.', ''): v for k, v in wts_dict.items()}

model.load_state_dict(wts_dict)
model = torch.nn.DataParallel(model).cuda()
model.float()
model.eval()


test_image = os.path.join('./readme/', img_name)
oriImg = cv2.imread(test_image) # B,G,R order
shape_dst = np.min(oriImg.shape[0:2])

# Get results of original image
with torch.no_grad():
    paf, heatmap, im_scale = get_outputs(oriImg, model, TRUNK)

print("im_scale:", im_scale)

humans = paf_to_pose_cpp(heatmap, paf, cfg)

out = draw_humans(oriImg, humans)
cv2.imwrite(img_name.split('.')[0] + '_result.png', out)   

