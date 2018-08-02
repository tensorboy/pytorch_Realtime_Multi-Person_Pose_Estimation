import math
import os
import re
import sys
import time
from collections import OrderedDict

import numpy as np
import skimage.io
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable

import caffe
import cv2
import util
from config_reader import config_reader

caffemodel = '../model/_trained_COCO/pose_iter_440000.caffemodel'
deployFile = '../model/_trained_COCO/pose_deploy.prototxt'
caffe.set_mode_cpu()
net = caffe.Net(deployFile, caffemodel, caffe.TEST)

#layers_caffe = dict(zip(list(net._layer_names), net.layers))
#print 'Number of layers: %i' % len(layers_caffe.keys())

blocks = {}

block0  = [{'conv1_1':[3,64,3,1,1]},{'conv1_2':[64,64,3,1,1]},{'pool1_stage1':[2,2,0]},{'conv2_1':[64,128,3,1,1]},{'conv2_2':[128,128,3,1,1]},{'pool2_stage1':[2,2,0]},{'conv3_1':[128,256,3,1,1]},{'conv3_2':[256,256,3,1,1]},{'conv3_3':[256,256,3,1,1]},{'conv3_4':[256,256,3,1,1]},{'pool3_stage1':[2,2,0]},{'conv4_1':[256,512,3,1,1]},{'conv4_2':[512,512,3,1,1]},{'conv4_3_CPM':[512,256,3,1,1]},{'conv4_4_CPM':[256,128,3,1,1]}]

blocks['block1_1']  = [{'conv5_1_CPM_L1':[128,128,3,1,1]},{'conv5_2_CPM_L1':[128,128,3,1,1]},{'conv5_3_CPM_L1':[128,128,3,1,1]},{'conv5_4_CPM_L1':[128,512,1,1,0]},{'conv5_5_CPM_L1':[512,38,1,1,0]}]

blocks['block1_2']  = [{'conv5_1_CPM_L2':[128,128,3,1,1]},{'conv5_2_CPM_L2':[128,128,3,1,1]},{'conv5_3_CPM_L2':[128,128,3,1,1]},{'conv5_4_CPM_L2':[128,512,1,1,0]},{'conv5_5_CPM_L2':[512,19,1,1,0]}]

for i in range(2,7):
    blocks['block%d_1'%i]  = [{'Mconv1_stage%d_L1'%i:[185,128,7,1,3]},{'Mconv2_stage%d_L1'%i:[128,128,7,1,3]},{'Mconv3_stage%d_L1'%i:[128,128,7,1,3]},{'Mconv4_stage%d_L1'%i:[128,128,7,1,3]},
{'Mconv5_stage%d_L1'%i:[128,128,7,1,3]},{'Mconv6_stage%d_L1'%i:[128,128,1,1,0]},{'Mconv7_stage%d_L1'%i:[128,38,1,1,0]}]
    blocks['block%d_2'%i]  = [{'Mconv1_stage%d_L2'%i:[185,128,7,1,3]},{'Mconv2_stage%d_L2'%i:[128,128,7,1,3]},{'Mconv3_stage%d_L2'%i:[128,128,7,1,3]},{'Mconv4_stage%d_L2'%i:[128,128,7,1,3]},
{'Mconv5_stage%d_L2'%i:[128,128,7,1,3]},{'Mconv6_stage%d_L2'%i:[128,128,1,1,0]},{'Mconv7_stage%d_L2'%i:[128,19,1,1,0]}]

             
def make_layers(cfg_dict):
    layers = []
    for i in range(len(cfg_dict)-1):
        one_ = cfg_dict[i]
        for k,v in one_.iteritems():      
            if 'pool' in k:
                layers += [nn.MaxPool2d(kernel_size=v[0], stride=v[1], padding=v[2] )]
            else:
                conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride = v[3], padding=v[4])
                layers += [conv2d, nn.ReLU(inplace=True)]
    one_ = cfg_dict[-1].keys()
    k = one_[0]
    v = cfg_dict[-1][k]
    conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride = v[3], padding=v[4])
    layers += [conv2d]
    return nn.Sequential(*layers)
    
layers = []
for i in range(len(block0)):
    one_ = block0[i]
    for k,v in one_.iteritems():      
        if 'pool' in k:
            layers += [nn.MaxPool2d(kernel_size=v[0], stride=v[1], padding=v[2] )]
        else:
            conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride = v[3], padding=v[4])
            layers += [conv2d, nn.ReLU(inplace=True)]  
       
models = {}           
models['block0']=nn.Sequential(*layers)        

for k,v in blocks.iteritems():
    models[k] = make_layers(v)
                
    
class pose_model(nn.Module):
    def __init__(self,model_dict,transform_input=False):
        super(pose_model, self).__init__()
        self.model0   = model_dict['block0']
        self.model1_1 = model_dict['block1_1']        
        self.model2_1 = model_dict['block2_1']  
        self.model3_1 = model_dict['block3_1']  
        self.model4_1 = model_dict['block4_1']  
        self.model5_1 = model_dict['block5_1']  
        self.model6_1 = model_dict['block6_1']  
        
        self.model1_2 = model_dict['block1_2']        
        self.model2_2 = model_dict['block2_2']  
        self.model3_2 = model_dict['block3_2']  
        self.model4_2 = model_dict['block4_2']  
        self.model5_2 = model_dict['block5_2']  
        self.model6_2 = model_dict['block6_2']
        
        
    def forward(self, x):    
        out1 = self.model0(x)
        
        out1_1 = self.model1_1(out1)
        out1_2 = self.model1_2(out1)
        out2  = torch.cat([out1_1,out1_2,out1],1)
        
        out2_1 = self.model2_1(out2)
        out2_2 = self.model2_2(out2)
        out3   = torch.cat([out2_1,out2_2,out1],1)
        
        out3_1 = self.model3_1(out3)
        out3_2 = self.model3_2(out3)
        out4   = torch.cat([out3_1,out3_2,out1],1)

        out4_1 = self.model4_1(out4)
        out4_2 = self.model4_2(out4)
        out5   = torch.cat([out4_1,out4_2,out1],1)  
        
        out5_1 = self.model5_1(out5)
        out5_2 = self.model5_2(out5)
        out6   = torch.cat([out5_1,out5_2,out1],1)         
              
        out6_1 = self.model6_1(out6)
        out6_2 = self.model6_2(out6)
        
        return out6_1,out6_2        

model = pose_model(models)     
model.eval()

pose_model_keys = model.state_dict().keys() 

match_dict = {}  
match_dict['model0.0']  = 'conv1_1'
match_dict['model0.2']  = 'conv1_2'
match_dict['model0.5']  = 'conv2_1'
match_dict['model0.7'] = 'conv2_2'
match_dict['model0.10'] = 'conv3_1'
match_dict['model0.12'] = 'conv3_2'
match_dict['model0.14'] = 'conv3_3'
match_dict['model0.16'] = 'conv3_4'
match_dict['model0.19'] = 'conv4_1'
match_dict['model0.21'] = 'conv4_2'
match_dict['model0.23'] = 'conv4_3_CPM'
match_dict['model0.25'] = 'conv4_4_CPM'


for i in range(1,7):
    for j in range(1,3):
        caffe_block = blocks['block%d_%d'%(i,j)]
        caffe_keys =[]
        for m in range(len(caffe_block)):
            caffe_keys+=caffe_block[m].keys()
        pytorch_block = []
        for item in pose_model_keys:
            if 'model%d_%d'%(i,j) in item:
                pytorch_block.append(item)
        for k in range(len(pytorch_block)):
            pytorch_block[k]=pytorch_block[k].rsplit('.',1)[0]
        pytorch_block = list(set(pytorch_block))
        pytorch_block = sorted(pytorch_block,key=lambda x: int(x.rsplit('.',1)[-1]))
        match_dict.update(dict(zip(pytorch_block,caffe_keys)))
        
#from copy import deepcopy
#new_state_dict = deepcopy(model.state_dict())
new_state_dict = OrderedDict()

for var_name in model.state_dict().keys():
    print var_name
    if 'weight' in var_name:
        name_in_caffe = match_dict[var_name.rsplit('.',1)[0]]
        data = net.params[name_in_caffe][0].data
    elif 'bias' in var_name:
        name_in_caffe = match_dict[var_name.rsplit('.',1)[0]]
        data = net.params[name_in_caffe][1].data  
    else:
        print 'bad'
    new_state_dict[var_name] = torch.from_numpy(data).float()
    
model.load_state_dict(new_state_dict)   
#model.cuda()
model.eval()



param_, model_ = config_reader()

test_image = '../sample_image/ski.jpg'
#test_image = 'a.jpg'
oriImg = cv2.imread(test_image) # B,G,R order

multiplier = [x * model_['boxsize'] / oriImg.shape[0] for x in param_['scale_search']]

#heatmap_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 19))
#paf_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 38))

for m in range(len(multiplier)):
    scale = multiplier[m]
    imageToTest = cv2.resize(oriImg, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    imageToTest_padded, pad = util.padRightDownCorner(imageToTest, model_['stride'], model_['padValue'])

    feed = np.transpose(np.float32(imageToTest_padded[:,:,:,np.newaxis]), (3,2,0,1))/256 - 0.5
   
    output1,output2 = model(Variable(torch.from_numpy(feed).float(), volatile=True))

    output1 = output1.cpu().data.numpy()
    output2 = output2.cpu().data.numpy()
    
    net.blobs['data'].reshape(*(1, 3, imageToTest_padded.shape[0], imageToTest_padded.shape[1]))
    net.blobs['data'].data[...] = feed
    
    output_blobs = net.forward()
    
    output1_ = net.blobs[output_blobs.keys()[0]].data
    output2_ = net.blobs[output_blobs.keys()[1]].data
    
    print 'output1 have %10.10f%% relative error'%(np.linalg.norm(output1-output1_)/np.linalg.norm(output1_)*100)
    print 'output2 have %10.10f%% relative error'%(np.linalg.norm(output2-output2_)/np.linalg.norm(output2_)*100)
    


torch.save(model.state_dict(), '../model/pose_model.pth')    
