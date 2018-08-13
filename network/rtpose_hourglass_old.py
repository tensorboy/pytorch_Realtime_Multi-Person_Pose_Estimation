from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init
from tnn.network.base_model import BaseModel
from tnn.network.net_utils import SeparableConv2d

from tnn.network.bnn.bnn_layer import BNBinaryConv2d, BNRealConv2d
from tnn.network.hourglass import Hourglass, ResidualBlock


class RTPoseModel(BaseModel):

    feat_stride = 4

    def __init__(self, conv_type, n_hourglass=1):
        super(RTPoseModel, self).__init__()

        if conv_type == 'binary':
            conv = BNBinaryConv2d
        elif conv_type == 'real':
            conv = BNRealConv2d
        elif conv_type == 'separable':
            conv = SeparableConv2d
        else:
            assert False, 'unknown conv type: {}'.format(conv_type)

        self.pre_conv = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResidualBlock(conv, 64, 128, 3, same_padding=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            ResidualBlock(conv, 128, 128, 3, same_padding=True),
            ResidualBlock(conv, 128, 256, 3, same_padding=True)
        )

        # stacked hourglass
        self.n_hourglass = n_hourglass
        for i in range(self.n_hourglass):
            setattr(self, 'hourglass_{}'.format(i), Hourglass(conv, 256, 3, n_maxpool=4))
            setattr(self, 'ht_conv_{}'.format(i), nn.Sequential(
                ResidualBlock(conv, 256, 256, 3, same_padding=True),
                conv(256, 256, 1, 1, 0),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 19, 1)
            ))
            setattr(self, 'paf_conv_{}'.format(i), nn.Sequential(
                ResidualBlock(conv, 256, 256, 3, same_padding=True),
                conv(256, 256, 1, 1, 0),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 38, 1)
            ))

            if i != self.n_hourglass - 1:
                setattr(self, 'hour2hourglass_{}'.format(i), conv(256, 256, 1, 1, 0))
                setattr(self, 'ht2hourglass_{}'.format(i), conv(19, 256, 3, 1, 1))
                setattr(self, 'paf2hourglass_{}'.format(i), conv(38, 256, 3, 1, 1))

    def forward(self, image):
        x_in = self.pre_conv(image)

        predicts = []
        for i in range(self.n_hourglass):
            x = getattr(self, 'hourglass_{}'.format(i))(x_in)
            ht_out = getattr(self, 'ht_conv_{}'.format(i))(x)
            paf_out = getattr(self, 'paf_conv_{}'.format(i))(x)

            predicts.append((paf_out, ht_out))

            if i != self.n_hourglass - 1:
                x_in = x_in + \
                       getattr(self, 'hour2hourglass_{}'.format(i))(x) + \
                       getattr(self, 'ht2hourglass_{}'.format(i))(ht_out) + \
                       getattr(self, 'paf2hourglass_{}'.format(i))(paf_out)

        # predict = [self.upsample(predicts[-1][0]), self.upsample(predicts[-1][1])]
        predict = predicts[-1]
        saved_for_loss = [predicts]

        return predict, saved_for_loss
