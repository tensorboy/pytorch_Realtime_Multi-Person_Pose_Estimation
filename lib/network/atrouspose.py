# camera-ready

import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import torchvision.models as models

def conv(in_channels, out_channels, kernel_size=3, padding=1, bn=True, dilation=1, stride=1, relu=True, bias=True):
    modules = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)]
    if bn:
        modules.append(nn.BatchNorm2d(out_channels))
    if relu:
        modules.append(nn.ReLU(inplace=True))
    return nn.Sequential(*modules)

class ASPP_ASP(nn.Module):
    def __init__(self, in_, out_=16):
        super(ASPP_ASP, self).__init__()

        self.conv_1x1_1 = nn.Conv2d(in_, 128, kernel_size=3, stride=1, padding=1, dilation=1)
        self.bn_conv_1x1_1 = nn.BatchNorm2d(128)

        self.conv_3x3_1 = nn.Conv2d(in_, 128, kernel_size=3, stride=1, padding=4, dilation=4)
        self.bn_conv_3x3_1 = nn.BatchNorm2d(128)

        self.conv_3x3_2 = nn.Conv2d(in_, 128, kernel_size=3, stride=1, padding=8, dilation=8)
        self.bn_conv_3x3_2 = nn.BatchNorm2d(128)

        self.conv_3x3_3 = nn.Conv2d(in_, 128, kernel_size=3, stride=1, padding=16, dilation=16)
        self.bn_conv_3x3_3 = nn.BatchNorm2d(128)

        self.bn_out = nn.BatchNorm2d(512)

    def forward(self, feature_map):

        out_1x1 = F.relu(self.bn_conv_1x1_1(self.conv_1x1_1(feature_map)))
        out_3x3_1 = F.relu(self.bn_conv_3x3_1(self.conv_3x3_1(feature_map)))
        out_3x3_2 = F.relu(self.bn_conv_3x3_2(self.conv_3x3_2(feature_map)))
        out_3x3_3 = F.relu(self.bn_conv_3x3_3(self.conv_3x3_3(feature_map)))

        add1 = out_1x1
        add2 = add1+out_3x3_1
        add3 = add2+out_3x3_2
        add4 = add3+out_3x3_3
        out = F.relu(self.bn_out(torch.cat([add1, add2, add3, add4], 1))) # (shape: (batch_size, 1280, h/16, w/16))

        return out
        

def conv_dw(in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation=dilation, groups=in_channels, bias=False),
        nn.BatchNorm2d(in_channels),
        nn.ReLU(inplace=True),

        nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )
class Upsample(nn.Module):
    # Custom Upsample layer (nn.Upsample gives deprecated warning message)

    def __init__(self, scale_factor=1, mode='nearest'):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=True)

class AtrousPose(nn.Module):
    def __init__(self, paf_out_channels=38, heat_out_channels=19):
        super(AtrousPose, self).__init__()
        """
        mobile net
        """
        resnet = models.resnet50(pretrained=True)
        self.layer3 = resnet.layer3
        self.resnet = nn.Sequential(*list(resnet.children())[:-4])
        self.smooth_ups2 = self._lateral(1024, 2)
        self.smooth_ups3 = self._lateral(512, 1)
        self.aspp1 = ASPP_ASP(512, out_=16)
        self.h1 = nn.Sequential(
            conv(512, 512, kernel_size=3, padding=1,),
            conv(512, 512, kernel_size=3, padding=1,),
            conv(512, 512, kernel_size=3, padding=1,),
            conv(512, 512, kernel_size=1, padding=0, bn=False),
            conv(512, heat_out_channels, kernel_size=1, padding=0, bn=False, relu=False)
        )
        self.p1 = nn.Sequential(
            conv(512, 512, kernel_size=3, padding=1,),
            conv(512, 512, kernel_size=3, padding=1,),
            conv(512, 512, kernel_size=3, padding=1,),
            conv(512, 512, kernel_size=1, padding=0, bn=False),
            conv(512, paf_out_channels, kernel_size=1, padding=0, bn=False, relu=False)
        )

    def _lateral(self, input_size, factor):
        layers = []
        layers.append(nn.Conv2d(input_size, 256, kernel_size=1, stride=1, bias=False))
        layers.append(nn.BatchNorm2d(256))
        layers.append(nn.ReLU(inplace=True))
        layers.append(Upsample(scale_factor=factor, mode='bilinear'))

        return nn.Sequential(*layers)
        
    def forward(self, x):
        # import time
        # s = time.time()
        feature_map = self.resnet(x)
        _16x = self.layer3(feature_map)
        # _32x = self.smooth_ups1(_32x)
        _16x = self.smooth_ups2(_16x)
        feature_map = self.smooth_ups3(feature_map)
        cat_feat = F.relu(torch.cat([feature_map, _16x], 1))

        out = self.aspp1(cat_feat)
        heatmap = self.h1(out)
        paf = self.p1(out)

        # e = time.time()
        return paf, heatmap

def model_info(model):  # Plots a line-by-line description of a PyTorch model
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    print('\n%5s %50s %9s %12s %20s %12s %12s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
    for i, (name, p) in enumerate(model.named_parameters()):
        name = name.replace('module_list.', '')
        print('%5g %50s %9s %12g %20s %12.3g %12.3g' % (
            i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))
    print('Model Summary: %g layers, %g parameters, %g gradients\n' % (i + 1, n_p, n_g))

if __name__ == "__main__":
    import cv2
    net = AtrousPose().cuda()
    image = cv2.imread('star.jpg')
    image = cv2.resize(image, (256, 256))
    image = torch.from_numpy(image).type(torch.FloatTensor).permute(2, 0, 1).reshape(1, 3, 256, 256).cuda()
    print(net)
    model_info(net)
    for i in range(30):
        vec1, heat1 = net(image)

    print(vec1.shape, heat1.shape)
