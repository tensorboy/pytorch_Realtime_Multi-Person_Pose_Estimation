from collections import OrderedDict

import torchvision

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.utils.model_zoo as model_zoo
from tnn.network.base_model import BaseModel
from torch.nn import init

class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, bn, **kwargs):
        super(BasicConv2d, self).__init__()
        self.bn = bn
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        if self.bn:
            self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        return F.relu(x, inplace=True)


class InceptionA(nn.Module):

    def __init__(self, in_channels, pool_features, have_bn, have_bias):
        super(InceptionA, self).__init__()
        self.branch1x1 = BasicConv2d(
            in_channels, 64, kernel_size=1, bn=have_bn, bias=have_bias)

        self.branch5x5_1 = BasicConv2d(
            in_channels, 48, kernel_size=1, bn=have_bn,  bias=have_bias)
        self.branch5x5_2 = BasicConv2d(
            48, 64, kernel_size=5, padding=2, bn=have_bn, bias=have_bias)

        self.branch3x3dbl_1 = BasicConv2d(
            in_channels, 64, kernel_size=1, bn=have_bn, bias=have_bias)
        self.branch3x3dbl_2 = BasicConv2d(
            64, 96, kernel_size=3, padding=1, bn=have_bn, bias=have_bias)
        self.branch3x3dbl_3 = BasicConv2d(
            96, 96, kernel_size=3, padding=1, bn=have_bn, bias=have_bias)

        self.branch_pool = BasicConv2d(
            in_channels, pool_features, kernel_size=1, bn=have_bn, bias=have_bias)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class dilation_layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding='same_padding', dilation=1):
        super(dilation_layer, self).__init__()
        if padding == 'same_padding':
            padding = int((kernel_size - 1) / 2 * dilation)
        self.Dconv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=kernel_size, padding=padding, dilation=dilation)
        self.Drelu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.Dconv(x)
        x = self.Drelu(x)
        return x


class stage_block(nn.Module):
    """This class makes sure the paf and heatmap branch out in every stage"""

    def __init__(self, in_channels, out_channels):
        super(stage_block, self).__init__()
        self.Dconv_1 = dilation_layer(in_channels, out_channels=64)
        self.Dconv_2 = dilation_layer(in_channels=64, out_channels=64)
        self.Dconv_3 = dilation_layer(
            in_channels=64, out_channels=64, dilation=2)
        self.Dconv_4 = dilation_layer(
            in_channels=64, out_channels=32, dilation=4)
        self.Dconv_5 = dilation_layer(
            in_channels=32, out_channels=32, dilation=8)
        self.Mconv_6 = nn.Conv2d(
            in_channels=256, out_channels=128, kernel_size=1, padding=0)
        self.Mrelu_6 = nn.ReLU(inplace=True)
        # Branch out
        self.paf = nn.Conv2d(
            in_channels=128, out_channels=14, kernel_size=1, padding=0)
        self.heatmap = nn.Conv2d(
            in_channels=128, out_channels=9, kernel_size=1, padding=0)

    def forward(self, x):
        x_1 = self.Dconv_1(x)
        x_2 = self.Dconv_2(x_1)
        x_3 = self.Dconv_3(x_2)
        x_4 = self.Dconv_4(x_3)
        x_5 = self.Dconv_5(x_4)
        x_cat = torch.cat([x_1, x_2, x_3, x_4, x_5], 1)
        x_out = self.Mconv_6(x_cat)
        x_out = self.Mrelu_6(x_out)
        paf = self.paf(x_out)
        heatmap = self.heatmap(x_out)
        return paf, heatmap


class feature_extractor(nn.Module):
    def __init__(self, have_bn, have_bias):
        super(feature_extractor, self).__init__()
        print("loading layers from inception_v3...")
        '''
        annotations in inception_v3
        self.Conv2d_1a_3x3 = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=3, padding=1)
        self.Conv2d_3b_1x1 = BasicConv2d(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = BasicConv2d(80, 192, kernel_size=3)
        self.Mixed_5b = InceptionA(192, pool_features=32)
        self.Mixed_5c = InceptionA(256, pool_features=64)
        '''
        self.conv1_3x3_s2 = BasicConv2d(
            3, 32, kernel_size=3, stride=2, padding=1, bn=have_bn, bias=have_bias)

        self.conv2_3x3_s1 = BasicConv2d(
            32, 32, kernel_size=3, stride=1, padding=1, bn=have_bn, bias=have_bias)

        self.conv3_3x3_s1 = BasicConv2d(
            32, 64, kernel_size=3, stride=1, padding=1, bn=have_bn, bias=have_bias)

        self.conv4_3x3_reduce = BasicConv2d(
            64, 80, kernel_size=1, stride=1, padding=1, bn=have_bn, bias=have_bias)
        self.conv4_3x3 = BasicConv2d(
            80, 192, kernel_size=3, bn=have_bn,  bias=have_bias)

        self.inception_a1 = InceptionA(
            192, pool_features=32, have_bn=have_bn, have_bias=have_bias)
        self.inception_a2 = InceptionA(
            256, pool_features=64, have_bn=have_bn, have_bias=have_bias)

    def forward(self, x):
        x = self.conv1_3x3_s2(x)
        x = self.conv2_3x3_s1(x)
        x = self.conv3_3x3_s1(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, ceil_mode=True)
        x = self.conv4_3x3_reduce(x)
        x = self.conv4_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, ceil_mode=True)
        x = self.inception_a1(x)
        x = self.inception_a2(x)
        # 46 x 46 x 288
        return x


class Ying_model(nn.Module):
    def __init__(self, stages=5, have_bn=True, have_bias=False):
        super(Ying_model, self).__init__()
        self.stages = stages
        self.feature_extractor = feature_extractor(
            have_bn=have_bn, have_bias=have_bias)
        self.stage_0 = nn.Sequential(nn.Conv2d(in_channels=288, out_channels=256, kernel_size=3, padding=1),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(
                                         in_channels=256, out_channels=128, kernel_size=3, padding=1),
                                     nn.ReLU(inplace=True))
        for i in range(stages):
            setattr(self, 'stage{}'.format(i + 2), stage_block(in_channels=128, out_channels=14) if i == 0 else
                    stage_block(in_channels=151, out_channels=14))

        self._initialize_weights_norm()

    def forward(self, x):
        saved_for_loss = []
        paf_ret, heat_ret = [], []        
        x_in = self.feature_extractor(x)
        x_in_0 = self.stage_0(x_in)
        x_in = x_in_0
        for i in range(self.stages):
            x_PAF_pred, x_heatmap_pred = getattr(
                self, 'stage{}'.format(i + 2))(x_in)
            paf_ret.append(x_PAF_pred)
            heat_ret.append(x_heatmap_pred)
            if i != self.stages - 1:
                x_in = torch.cat([x_PAF_pred, x_heatmap_pred, x_in_0], 1)
        saved_for_loss.append(paf_ret)
        saved_for_loss.append(heat_ret)                
        return [(paf_ret[-2], heat_ret[-2]), (paf_ret[-1], heat_ret[-1])], saved_for_loss

    def _initialize_weights_norm(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.normal(m.weight, std=0.01)
                if m.bias is not None:  # mobilenet conv2d doesn't add bias
                    init.constant(m.bias, 0.001)

    @staticmethod
    def build_loss(saved_for_loss, heat_temp, heat_weight,
                   vec_temp, vec_weight, batch_size, gpus):
        names = build_names()
        saved_for_log = OrderedDict()
        criterion = nn.MSELoss(size_average=True).cuda()
        total_loss = 0
        for j in range(5):
            pred1 = saved_for_loss[0][j] * vec_weight
            """
            print("pred1 sizes")
            print(saved_for_loss[2*j].data.size())
            print(vec_weight.data.size())
            print(vec_temp.data.size())
            """
            gt1 = vec_temp * vec_weight
            pred2 = saved_for_loss[1][j] * heat_weight
            gt2 = heat_weight * heat_temp
            """
            print("pred2 sizes")
            print(saved_for_loss[2*j+1].data.size())
            print(heat_weight.data.size())
            print(heat_temp.data.size())
            """
            # Compute losses
            loss1 = criterion(pred1, gt1) 
            loss2 = criterion(pred2, gt2) 
            total_loss += loss1
            total_loss += loss2
            # print(total_loss)
            # Get value from Variable and save for log
            saved_for_log[names[2 * j]] = loss1.data[0]
            saved_for_log[names[2 * j + 1]] = loss2.data[0]
        return total_loss, saved_for_log


def get_ying_model(stages=5, have_bn=False, have_bias=True):
    return Ying_model(stages=stages, have_bn=have_bn, have_bias=have_bias)


def use_inception(model, model_path):

    url = 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth'
    incep_state_dict = model_zoo.load_url(url, model_dir=model_path)
    incep_keys = incep_state_dict.keys()

    # load weights of vgg
    weights_load = {}
    # weight+bias,weight+bias.....(repeat 10 times)
    for i in range(60):
        weights_load[list(model.state_dict().keys())[i]
                     ] = incep_state_dict[list(incep_keys)[i]]

    state = model.state_dict()
    state.update(weights_load)
    model.load_state_dict(state)
    print('load imagenet pretrained model: {}'.format(model_path))


def build_names():
    names = []

    for j in range(1, 6):
        for k in range(1, 3):
            names.append('loss_stage%d_L%d' % (j, k))
    return names
