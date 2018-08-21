import copy
import json
import math
# general package
import os
import random
import struct
import sys
import time
from collections import OrderedDict
from math import sqrt

import cv2
import matplotlib.cm
import numpy as np
import pandas as pd
import scipy
import scipy.io as sio
import torchvision.datasets as dset
import torchvision.transforms as transforms
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from scipy.ndimage.filters import gaussian_filter

# torch package
import torch
import torch.nn as nn
import torch.utils.data as data
#from network.Ying_model import get_ying_model
from network.rtpose_vgg import get_model

#import utils
from training.datasets.coco_data.preprocessing import (inception_preprocess,
                                              rtpose_preprocess,
                                              ssd_preprocess, vgg_preprocess)
from network import im_transform
from evaluate.coco_eval import get_multiplier, get_outputs, handle_paf_and_heat

blocks = {}
block0 = ['conv1_1',
          'conv1_2',
          'pool1_stage1',
          'conv2_1',
          'conv2_2',
          'pool2_stage1',
          'conv3_1',
          'conv3_2',
          'conv3_3',
          'conv3_4',
          'pool3_stage1',
          'conv4_1',
          'conv4_2',
          'conv4_3_CPM',
          'conv4_4_CPM']

blocks['block0']=[]
for i, item in enumerate(block0):
    if 'conv' in item:
        blocks['block0'].append(item)
        blocks['block0'].append(item+'_relu')        
    elif 'pool' in item:
        blocks['block0'].append(item)         


# Stage 1
block1_1 = ['conv5_1_CPM_L1',
                      'conv5_2_CPM_L1',
                      'conv5_3_CPM_L1',
                      'conv5_4_CPM_L1',
                      'conv5_5_CPM_L1']

block1_2 = ['conv5_1_CPM_L2', 
                      'conv5_2_CPM_L2', 
                      'conv5_3_CPM_L2',
                      'conv5_4_CPM_L2', 
                      'conv5_5_CPM_L2']     

blocks['block1_1']=[]                                       
for i in range(len(block1_1)):
    item = block1_1[i]
    blocks['block1_1'].append(item)
    
    if i==(len(block1_1)-1):
        break
        
    blocks['block1_1'].append(item+'_relu')        
  
blocks['block1_2']=[]                                       
for i in range(len(block1_2)):
    item = block1_2[i]
    blocks['block1_2'].append(item)
    
    if i==(len(block1_2)-1):
        break
        
    blocks['block1_2'].append(item+'_relu')   
    
   
# Stages 2 - 6
for i in range(2, 7):
    block1 = [
        'Mconv1_stage%d_L1' % i,
        'Mconv2_stage%d_L1' % i,
        'Mconv3_stage%d_L1' % i,
        'Mconv4_stage%d_L1' % i,
        'Mconv5_stage%d_L1' % i,
        'Mconv6_stage%d_L1' % i,
        'Mconv7_stage%d_L1' % i]

    block2 = [
        'Mconv1_stage%d_L2' % i,
        'Mconv2_stage%d_L2' % i,
        'Mconv3_stage%d_L2' % i,
        'Mconv4_stage%d_L2' % i,
        'Mconv5_stage%d_L2' % i,
        'Mconv6_stage%d_L2' % i,
        'Mconv7_stage%d_L2' % i
    ]

    blocks['block%d_1' % i] = []
    blocks['block%d_2' % i] = []
    
    for k in range(len(block1)):
        item = block1[k]
        blocks['block%d_1' % i].append(item)
        
        if k==(len(block1)-1):
            break
            
        blocks['block%d_1' % i].append(item+'_relu')        
      
                                     
    for k in range(len(block2)):
        item = block2[k]
        blocks['block%d_2' % i].append(item)
        
        if k==(len(block2)-1):
            break
            
        blocks['block%d_2' % i].append(item+'_relu')     



class FeatureExtractor(nn.Module):
    def __init__(self, submodule):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule

    def forward(self, x):
        outputs = []
        for name, module in self.submodule._modules.items():
        
            x = module(x)
            
            z = x.data.cpu().numpy()
            outputs += [z]
            
        return outputs, x

feature_save_path = '../all_yoga_features_our_best_model/'

if __name__ == "__main__":
    '''
    MS COCO annotation order:
    0: nose	   		1: l eye		2: r eye	3: l ear	4: r ear
    5: l shoulder	6: r shoulder	7: l elbow	8: r elbow
    9: l wrist		10: r wrist		11: l hip	12: r hip	13: l knee
    14: r knee		15: l ankle		16: r ankle

    The order in this work:
    (0-'nose'	1-'neck' 2-'right_shoulder' 3-'right_elbow' 4-'right_wrist'
    5-'left_shoulder' 6-'left_elbow'	    7-'left_wrist'  8-'right_hip'  
    9-'right_knee'	 10-'right_ankle'	11-'left_hip'   12-'left_knee' 
    13-'left_ankle'	 14-'right_eye'	    15-'left_eye'   16-'right_ear' 
    17-'left_ear' )
    '''
    orderCOCO = [0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10]

    mid_1 = [1, 8,  9, 1,  11, 12, 1, 2, 3,
             2,  1, 5, 6, 5,  1, 0,  0,  14, 15]

    mid_2 = [8, 9, 10, 11, 12, 13, 2, 3, 4,
             16, 5, 6, 7, 17, 0, 14, 15, 16, 17]

    # This txt file is get at the caffe_rtpose repository:
    # https://github.com/CMU-Perceptual-Computing-Lab/caffe_rtpose/blob/master/image_info_val2014_1k.txt

    image_dir = '/data/coco/val2014/'
    save_dir  = '/data/coco/val2014_features/'

    model = get_model('vgg19')
    model = torch.nn.DataParallel(model).cuda()
#    model = get_ying_model(stages=5, have_bn=True, have_bias=False)
#    Load our model
    weight_name = './network/weight/pose_model_scratch.pth'
    model.load_state_dict(torch.load(weight_name))
    model = model.module
#    model.load_state_dict(torch.load('pose_model.pth'))
#    model.load_state_dict(torch.load('../caffe_model/dilated3_5stage_merged.pth'))
#    model.load_state_dict(torch.load('../caffe_model/dilated3_remove_stage1_test.pth'))

    model.eval()
    model.float()
    model.cuda()
    
    feature_extractor = FeatureExtractor(model.model0)
    
    feature_extractor1_1 = FeatureExtractor(model.model1_1)
    feature_extractor1_2 = FeatureExtractor(model.model1_2)
    
    feature_extractor2_1 = FeatureExtractor(model.model2_1)
    feature_extractor2_2 = FeatureExtractor(model.model2_2)
    
    feature_extractor3_1 = FeatureExtractor(model.model3_1)
    feature_extractor3_2 = FeatureExtractor(model.model3_2)
    
    feature_extractor4_1 = FeatureExtractor(model.model4_1)
    feature_extractor4_2 = FeatureExtractor(model.model4_2)
    
    feature_extractor5_1 = FeatureExtractor(model.model5_1)
    feature_extractor5_2 = FeatureExtractor(model.model5_2)
    
    feature_extractor6_1 = FeatureExtractor(model.model6_1)
    feature_extractor6_2 = FeatureExtractor(model.model6_2)    
                                 
    
    images = os.listdir(image_dir)
    # iterate all val images
    feed_size = 368
    for a_path in images:
        one_path = image_dir + a_path
        print(one_path)

        oriImg = cv2.imread(one_path)
      
        im_croped, im_scale, real_shape = im_transform.crop_with_factor(
            oriImg, feed_size, factor=8, is_ceil=True)
        print('size of im crop', im_croped.shape)
            
        im_data = vgg_preprocess(im_croped)

        #im_data = im_data.transpose([2, 0, 1]).astype(np.float32)        
        
        batch_images = np.expand_dims(im_data,0)
       
        batch_var = torch.from_numpy(batch_images).cuda().float()
         
        all_outputs = []
         
        outputs0, out1 = feature_extractor(batch_var)
        
        outputs1_1, out1_1 = feature_extractor1_1(out1)
        outputs1_2, out1_2 = feature_extractor1_2(out1)
        out2 = torch.cat([out1_1, out1_2, out1], 1)

        outputs2_1, out2_1 = feature_extractor2_1(out2)
        outputs2_2, out2_2 = feature_extractor2_2(out2)
        out3 = torch.cat([out2_1, out2_2, out1], 1)


        outputs3_1, out3_1 = feature_extractor3_1(out3)
        outputs3_2, out3_2 = feature_extractor3_2(out3)
        out4 = torch.cat([out3_1, out3_2, out1], 1)


        outputs4_1,  out4_1 = feature_extractor4_1(out4)
        outputs4_2, out4_2 = feature_extractor4_2(out4)
        out5 = torch.cat([out4_1, out4_2, out1], 1)

        outputs5_1, out5_1 = feature_extractor5_1(out5)
        outputs5_2,  out5_2 = feature_extractor5_2(out5)
        out6 = torch.cat([out5_1, out5_2, out1], 1)

        outputs6_1,  out6_1 = feature_extractor6_1(out6)
        outputs6_2,  out6_2 = feature_extractor6_2(out6)

        all_outputs = outputs0 +outputs1_1 + outputs1_2 + outputs2_1 + outputs2_2 + \
        outputs3_1  + outputs3_2  + outputs4_1  + outputs4_2  + \
        outputs5_1  + outputs5_2  + outputs6_1  + outputs6_2 \
        
        all_names = blocks['block0']+blocks['block1_1']+blocks['block1_2']+blocks['block2_1']\
        +blocks['block2_2']+blocks['block3_1']+blocks['block3_2']+blocks['block4_1']+blocks['block4_2']\
        +blocks['block5_1']+blocks['block5_2']+blocks['block6_1']+blocks['block6_2']

        for name,tensor in zip(all_names,all_outputs):
            tensor = tensor[0]
            for n in range(tensor.shape[0]):
                one_save_dir = os.path.join(save_dir, a_path.split('.')[0], name)
                try:
                    os.makedirs(one_save_dir)
                except OSError:
                    pass
                a_tensor = tensor[n]                
                a_tensor = (a_tensor-np.min(a_tensor))/(np.max(a_tensor)-np.min(a_tensor))
                a_image = np.clip(a_tensor*255,0,255).astype(np.uint8)
                print(a_image.shape)
                one_save_path = os.path.join(one_save_dir, str(n)+'.jpg') 
                cv2.imwrite(one_save_path, a_image)
