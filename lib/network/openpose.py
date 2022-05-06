from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
from torch.nn import init
import pickle


def make_vgg19_block():
    """Builds a vgg19 block from a dictionary
    Args:
        block: a dictionary
    """
    block = [{'conv1_1': [3, 64, 3, 1, 1]},
            {'conv1_2': [64, 64, 3, 1, 1]},
            {'pool1_stage1': [2, 2, 0]},
            {'conv2_1': [64, 128, 3, 1, 1]},
            {'conv2_2': [128, 128, 3, 1, 1]},
            {'pool2_stage1': [2, 2, 0]},
            {'conv3_1': [128, 256, 3, 1, 1]},
            {'conv3_2': [256, 256, 3, 1, 1]},
            {'conv3_3': [256, 256, 3, 1, 1]},
            {'conv3_4': [256, 256, 3, 1, 1]},
            {'pool3_stage1': [2, 2, 0]},
            {'conv4_1': [256, 512, 3, 1, 1]},
            {'conv4_2': [512, 512, 3, 1, 1]},
            {'conv4_3_CPM': [512, 256, 3, 1, 1]},
            {'conv4_4_CPM': [256, 128, 3, 1, 1]}]
    layers = []
    for i in range(len(block)):
        one_ = block[i]
        for k, v in one_.items():
            if 'pool' in k:
                layers += [nn.MaxPool2d(kernel_size=v[0], stride=v[1],
                                        padding=v[2])]
            elif k in ['conv4_2', 'conv4_3_CPM', 'conv4_4_CPM']:
                conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1],
                                   kernel_size=v[2], stride=v[3],
                                   padding=v[4])
                layers += [conv2d, nn.PReLU(num_parameters=v[1])]
            else:
                conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1],
                                   kernel_size=v[2], stride=v[3],
                                   padding=v[4])
                layers += [conv2d, nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.Mconv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=kernel_size, stride=stride, padding=padding)
        self.MPrelu = nn.PReLU(num_parameters=out_channels)

    def forward(self, x):
        x = self.Mconv(x)
        x = self.MPrelu(x)
        return x

class StageBlock(nn.Module):
    """ L1/L2 StageBlock Template """
    def __init__(self, in_channels, inner_channels, innerout_channels, out_channels):
        super(StageBlock, self).__init__()
        self.Mconv1_0 = ConvBlock(in_channels, inner_channels)
        self.Mconv1_1 = ConvBlock(inner_channels, inner_channels)
        self.Mconv1_2 = ConvBlock(inner_channels, inner_channels)
        self.Mconv2_0 = ConvBlock(inner_channels * 3, inner_channels)
        self.Mconv2_1 = ConvBlock(inner_channels, inner_channels)
        self.Mconv2_2 = ConvBlock(inner_channels, inner_channels)
        self.Mconv3_0 = ConvBlock(inner_channels * 3, inner_channels)
        self.Mconv3_1 = ConvBlock(inner_channels, inner_channels)
        self.Mconv3_2 = ConvBlock(inner_channels, inner_channels)
        self.Mconv4_0 = ConvBlock(inner_channels * 3, inner_channels)
        self.Mconv4_1 = ConvBlock(inner_channels, inner_channels)
        self.Mconv4_2 = ConvBlock(inner_channels, inner_channels)
        self.Mconv5_0 = ConvBlock(inner_channels * 3, inner_channels)
        self.Mconv5_1 = ConvBlock(inner_channels, inner_channels)
        self.Mconv5_2 = ConvBlock(inner_channels, inner_channels)
        self.Mconv6 = ConvBlock(inner_channels * 3, innerout_channels, kernel_size=1, stride=1, padding=0)
        self.Mconv7 = nn.Conv2d(in_channels=innerout_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        out1_1 = self.Mconv1_0(x)
        out2_1 = self.Mconv1_1(out1_1)
        out3_1 = self.Mconv1_2(out2_1)
        x_cat_1 = torch.cat([out1_1, out2_1, out3_1], 1)
        out1_2 = self.Mconv2_0(x_cat_1)
        out2_2 = self.Mconv2_1(out1_2)
        out3_2 = self.Mconv2_2(out2_2)
        x_cat_2 = torch.cat([out1_2, out2_2, out3_2], 1)
        out1_3 = self.Mconv3_0(x_cat_2)
        out2_3 = self.Mconv3_1(out1_3)
        out3_3 = self.Mconv3_2(out2_3)
        x_cat_3 = torch.cat([out1_3, out2_3, out3_3], 1)
        out1_4 = self.Mconv4_0(x_cat_3)
        out2_4 = self.Mconv4_1(out1_4)
        out3_4 = self.Mconv4_2(out2_4)
        x_cat_4 = torch.cat([out1_4, out2_4, out3_4], 1)
        out1_5 = self.Mconv5_0(x_cat_4)
        out2_5 = self.Mconv5_1(out1_5)
        out3_5 = self.Mconv5_2(out2_5)
        x_cat_5 = torch.cat([out1_5, out2_5, out3_5], 1)
        out_6 = self.Mconv6(x_cat_5)
        stage_output = self.Mconv7(out_6)
        return stage_output

class OpenPose_Model(nn.Module):
    def __init__(self, l2_stages=4, l1_stages=2,
                paf_out_channels=14, heat_out_channels=9):
        '''
        :param feature_extractor:
        :param l2_stages:
        :param l1_stages:
        :param paf_out_channels:
        :param heat_out_channels:
        :param stage_input_mode: either 'from_first_stage' (original)
        or 'from_previous_stage' (i.e. take x_out from previous stage as
        input to next stage).
        '''
        super(OpenPose_Model, self).__init__()
        self.stages = [0, 1]
        # Backbone - feature extractor
        self.feature_extractor = make_vgg19_block()
        # L2 Stages
        L2_IN_CHS = [128]
        L2_INNER_CHS = [96]
        L2_INNEROUT_CHS = [256]
        L2_OUT_CHS = [paf_out_channels]
        for _ in range(l2_stages - 1):
            L2_IN_CHS.append(128 + paf_out_channels)
            L2_INNER_CHS.append(128)
            L2_INNEROUT_CHS.append(512)
            L2_OUT_CHS.append(paf_out_channels)
        self.l2_stages = nn.ModuleList([
            StageBlock(in_channels=L2_IN_CHS[i], inner_channels=L2_INNER_CHS[i], 
                       innerout_channels=L2_INNEROUT_CHS[i], out_channels=L2_OUT_CHS[i])
            for i in range(len(L2_IN_CHS))
        ])
        # L1 Stages
        L1_IN_CHS = [128 + paf_out_channels]
        L1_INNER_CHS = [96]
        L1_INNEROUT_CHS = [256]
        L1_OUT_CHS = [heat_out_channels]
        for _ in range(l1_stages - 1):
            L1_IN_CHS.append(128 + paf_out_channels + heat_out_channels)
            L1_INNER_CHS.append(128)
            L1_INNEROUT_CHS.append(512)
            L1_OUT_CHS.append(heat_out_channels)
        self.l1_stages = nn.ModuleList([
            StageBlock(in_channels=L1_IN_CHS[i], inner_channels=L1_INNER_CHS[i], 
                       innerout_channels=L1_INNEROUT_CHS[i], out_channels=L1_OUT_CHS[i])
            for i in range(len(L1_IN_CHS))
        ])
        self._initialize_weights_norm()

    def forward(self, x):
        saved_for_loss = []
        features = self.feature_extractor(x)
        paf_ret, heat_ret = [], []
        x_in = features
        # L2 Stage inference
        for l2_stage in self.l2_stages:
            paf_pred = l2_stage(x_in)
            x_in = torch.cat([features, paf_pred], 1)
            paf_ret.append(paf_pred)
        # L1 Stage inference
        for l1_stage in self.l1_stages:
            heat_pred = l1_stage(x_in)
            x_in = torch.cat([features, heat_pred, paf_pred], 1)
            heat_ret.append(heat_pred)
        saved_for_loss.append(paf_ret)
        saved_for_loss.append(heat_ret)
        return [(paf_ret[-2], heat_ret[-2]), (paf_ret[-1], heat_ret[-1])], saved_for_loss

    def _initialize_weights_norm(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:  # mobilenet conv2d doesn't add bias
                    init.constant_(m.bias, 0.001)
            elif isinstance(m, nn.PReLU):
                init.normal_(m.weight, std=0.01)
                
            
    def init_w_pretrained_weights(self, pkl_weights='/home/tomas/Desktop/AIFI/internal-repos/aifi-pose/network/weights/openpose/openpose.pkl'):
        with open(pkl_weights, 'rb') as f:
            weights = pickle.load(f, encoding='latin1')
        conv_idxs = [i
            for i, d in enumerate(weights)
            if 'conv' in d['name'] and 'split' not in d['name'] and 'concat' not in d['name']
        ]
        prelu_idxs = [i
            for i, d in enumerate(weights)
            if 'prelu' in d['name'] and 'split' not in d['name'] and 'concat' not in d['name']
        ]
        conv_idxs = iter(conv_idxs)
        prelu_idxs = iter(prelu_idxs)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                idx = next(conv_idxs)
                m.weight = torch.nn.Parameter(torch.Tensor(weights[idx]['weights'][0]))
                m.bias = torch.nn.Parameter(torch.Tensor(weights[idx]['weights'][1]))
            elif isinstance(m, nn.PReLU):
                idx = next(prelu_idxs)
                m.weight = torch.nn.Parameter(torch.Tensor(weights[idx]['weights'][0]))

    
def use_vgg(model):

    url = 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'
    vgg_state_dict = model_zoo.load_url(url)
    vgg_keys = vgg_state_dict.keys()

    # load weights of vgg
    weights_load = {}
    # weight+bias,weight+bias.....(repeat 10 times)
    for i in range(20):
        weights_load[list(model.state_dict().keys())[i]
                     ] = vgg_state_dict[list(vgg_keys)[i]]

    state = model.state_dict()
    state.update(weights_load)
    model.load_state_dict(state)
    print('load imagenet pretrained model')
