import argparse
import time
import os
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim.lr_scheduler as lr_scheduler
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from test_tube import Experiment

import os.path as osp
import sys
import cv2
import math


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


this_dir = osp.abspath(osp.dirname(__file__))

# Add Framework to PYTHONPATH
lib_path = osp.join(this_dir, '../..')
add_path(lib_path)
from lib.network.rtpose_vgg import get_model, use_vgg
from lib.datasets import coco, transforms, datasets
from lib.config import cfg, update_config
from lib.utils.paf_to_pose import paf_to_pose

def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default='../../experiments/vgg19_368x368_sgd_lr1.yaml',
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('-o', '--output', default=None,
                        help='output file')
    parser.add_argument('--disable-cuda', action='store_true',
                        help='disable CUDA')
    parser.add_argument('--outputDir', default='/data/rtpose/', type=str, metavar='DIR',
                        help='path to where the log saved')
    parser.add_argument('--dataDir', default='/data', type=str, metavar='DIR',
                        help='path to where the data saved')
    args = parser.parse_args()

    # add args.device
    args.device = torch.device('cpu')
    args.pin_memory = False
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        args.pin_memory = True

    return args


def get_result(person_to_joint_assoc, joint_list):
    """Build the outputs to be evaluated
    :param image_id: int, the id of the current image
    :param person_to_joint_assoc: numpy array of joints associations
    :param joint_list: list, list of joints
    :param outputs: list of dictionaries with the following keys: image_id,
                    category_id, keypoints, score
    """
    outputs = []
    for ridxPred in range(len(person_to_joint_assoc)):

        keypoints = np.zeros((18, 3))

        for part in range(18):
            index = int(person_to_joint_assoc[ridxPred, part])

            if -1 == index:
                keypoints[part, 0] = 0
                keypoints[part, 1] = 0
                keypoints[part, 2] = 0

            else:
                keypoints[part, 0] = joint_list[index, 0] + 0.5
                keypoints[part, 1] = joint_list[index, 1] + 0.5
                keypoints[part, 2] = 1.

        outputs.append(keypoints)
    return outputs

args = parse_args()
update_config(cfg, args)
print("Loading dataset...")
# load train data
preprocess = transforms.Compose([
    transforms.Normalize(),
    transforms.RandomRotate(max_rotate_degree=40),
    transforms.RandomApply(transforms.HFlip(), 0.5),
    transforms.RescaleRelative(),
    transforms.Crop(cfg.DATASET.IMAGE_SIZE),
    transforms.CenterPad(cfg.DATASET.IMAGE_SIZE),

])

def kp_connections(keypoints):
    kp_lines = [
        [keypoints.index('neck'), keypoints.index('right_hip')],  
        [keypoints.index('right_hip'), keypoints.index('right_knee')],
        [keypoints.index('right_knee'), keypoints.index('right_ankle')],
        [keypoints.index('neck'), keypoints.index('left_hip')],                
        [keypoints.index('left_hip'), keypoints.index('left_knee')],
        [keypoints.index('left_knee'), keypoints.index('left_ankle')],
        [keypoints.index('neck'), keypoints.index('right_shoulder')],          
        [keypoints.index('right_shoulder'), keypoints.index('right_elbow')],
        [keypoints.index('right_elbow'), keypoints.index('right_wrist')],     
        [keypoints.index('right_shoulder'), keypoints.index('right_eye')],        
        [keypoints.index('neck'), keypoints.index('left_shoulder')], 
        [keypoints.index('left_shoulder'), keypoints.index('left_elbow')],
        [keypoints.index('left_elbow'), keypoints.index('left_wrist')],
        [keypoints.index('left_shoulder'), keypoints.index('left_eye')],               
        [keypoints.index('neck'), keypoints.index('nose')],                      
        [keypoints.index('nose'), keypoints.index('right_eye')],
        [keypoints.index('nose'), keypoints.index('left_eye')],        
        [keypoints.index('right_eye'), keypoints.index('right_ear')],
        [keypoints.index('left_eye'), keypoints.index('left_ear')]
    ]
    return kp_lines
    
def get_keypoints():
    """Get the COCO keypoints and their left/right flip coorespondence map."""
    # Keypoints are not available in the COCO json for the test split, so we
    # provide them here.
    keypoints = [
        'nose',
        'neck',
        'right_shoulder',
        'right_elbow',
        'right_wrist',   
        'left_shoulder',
        'left_elbow',
        'left_wrist',
        'right_hip',
        'right_knee',
        'right_ankle',
        'left_hip',
        'left_knee',
        'left_ankle',
        'right_eye',                                                                    
        'left_eye',
        'right_ear',
        'left_ear']

    return keypoints
    
limb_thickness = 4

joint_to_limb_heatmap_relationship = kp_connections(get_keypoints())


NUM_LIMBS = len(joint_to_limb_heatmap_relationship)
colors = [
    [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0],
    [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255],
    [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], [170, 0, 255],
    [255, 0, 255], [255, 0, 170], [255, 0, 85], [255, 0, 0]]
    
def inverse_transform_image(torch_image):
    
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    
    image = torch_image.numpy()
    image = image.transpose(1,2,0)    
    image = image*std+mean
    image = image*255
    image = image.astype(np.uint8)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    return image
    
def inverse_vgg_preprocess(image):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    image = image.transpose((1,2,0))
    
    for i in range(3):
        image[:, :, i] = image[:, :, i] * stds[i]
        image[:, :, i] = image[:, :, i] + means[i]
    image = image.copy()[:,:,::-1]
    image = image*255
    
    return image         


train_datas = [datasets.CocoKeypoints(
    root=cfg.DATASET.TRAIN_IMAGE_DIR,
    annFile=item,
    preprocess=preprocess,
    image_transform=transforms.image_transform_train,
    target_transforms=None,
    n_images=None,
) for item in cfg.DATASET.TRAIN_ANNOTATIONS]

train_data = torch.utils.data.ConcatDataset(train_datas)

train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU*len(cfg.GPUS), shuffle=True,
    pin_memory=cfg.PIN_MEMORY, num_workers=cfg.WORKERS, drop_last=True)   
    

val_data = datasets.CocoKeypoints(
    root=cfg.DATASET.VAL_IMAGE_DIR,
    annFile=cfg.DATASET.VAL_ANNOTATIONS,
    preprocess=preprocess,
    image_transform=transforms.image_transform_train,
    target_transforms=None,
    n_images=None,
)
val_loader = torch.utils.data.DataLoader(
    val_data, batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS), shuffle=False,
    pin_memory=cfg.PIN_MEMORY, num_workers=cfg.WORKERS, drop_last=True) 



vis_dir = '/home/tensorboy/data/coco/images/vis'
for batch_idx, (image, heatmaps, pafs) in enumerate(train_loader):
    print(batch_idx)
    print(image.shape)
    print(heatmaps.shape)
    print(pafs.shape)
    index = np.random.randint(0,72)
    image = image[index].cpu().numpy()
    img = inverse_vgg_preprocess(image)
    heatmap = heatmaps[index].cpu().numpy().transpose((1,2,0))
    paf = pafs[index].cpu().numpy().transpose((1,2,0))
    param = {'thre1': 0.1, 'thre2': 0.05, 'thre3': 0.5} 
    
    joint_list, person_to_joint_assoc = paf_to_pose(heatmap, paf)

    keypoints = get_result(person_to_joint_assoc, joint_list)
    print(len(keypoints))
    new_keypoints = []
    for keypoint in keypoints:
        keypoint[:,:2]=keypoint[:,:2]
        new_keypoints.append(keypoint)
        for j in range(8):
            joint = keypoint[j]
            print(joint)
            if joint[2]==1:
                cv2.circle(img, tuple(joint[0:2].astype(int)), 2, colors[j], thickness=-1)               
        for k in range(NUM_LIMBS):
            src_joint = keypoint[joint_to_limb_heatmap_relationship[k][0]]
            dst_joint = keypoint[joint_to_limb_heatmap_relationship[k][1]]
            if src_joint[2]==1 and dst_joint[2]==1:
                coords_center = tuple(
                np.round((src_joint[:2]+dst_joint[:2])/2.).astype(int))
                # joint_coords[0,:] is the coords of joint_src; joint_coords[1,:]
                # is the coords of joint_dst
                limb_dir = src_joint - dst_joint
                limb_length = np.linalg.norm(limb_dir)
                # Get the angle of limb_dir in degrees using atan2(limb_dir_x,
                # limb_dir_y)
                angle = math.degrees(math.atan2(limb_dir[1], limb_dir[0]))

                # For faster plotting, just plot over canvas instead of constantly
                # copying it
                polygon = cv2.ellipse2Poly(
                    coords_center, (int(limb_length / 2), limb_thickness), int(angle), 0, 360, 1)
                cv2.fillConvexPoly(img, polygon, colors[k])            
    #    print(one_dict.keys())
    #if output_picture_name:
    cv2.imwrite(os.path.join(vis_dir, str(batch_idx)+'.png'), img)    
