"""CPM Pytorch Implementation"""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.utils.model_zoo as model_zoo
from tnn.network.base_model import BaseModel
from torch.autograd import Variable
from torch.nn import init


def conv_bn(inp, oup, stride):
    """Implementation of Batch Normalization Layers
    Args:
        inp: int, the number of input channels
        oup: int, the number of output channels
        stride: int, the stride (padding) of the convolution
    Returns:
        a sequential module defining the new layer
    """
    return nn.Sequential(nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                         nn.BatchNorm2d(oup),
                         nn.ReLU(inplace=True))


def conv_dw(inp, oup, stride):
    """Implementation of Separable Convolution Layers
    Args:
        inp: int, the number of input channels
        oup: int, the number of output channels
        stride: int, the stride (padding) of the convolution
    Returns:
        a sequential module defining the new layer
    """
    return nn.Sequential(nn.Conv2d(inp, inp, 3, stride, 1, groups=inp,
                                   bias=False),
                         nn.BatchNorm2d(inp),
                         nn.ReLU(inplace=True),
                         nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                         nn.BatchNorm2d(oup), nn.ReLU(inplace=True))


def make_stages(cfg_dict):
    """Builds CPM stages from a dictionary
    Args:
        cfg_dict: a dictionary
    """
    layers = []
    for i in range(len(cfg_dict) - 1):
        one_ = cfg_dict[i]
        for k, v in one_.iteritems():
            if 'pool' in k:
                layers += [nn.MaxPool2d(kernel_size=v[0], stride=v[1],
                                        padding=v[2])]
            else:
                conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1],
                                   kernel_size=v[2], stride=v[3],
                                   padding=v[4])
                layers += [conv2d, nn.ReLU(inplace=True)]
    one_ = cfg_dict[-1].keys()
    k = one_[0]
    v = cfg_dict[-1][k]
    conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1],
                       kernel_size=v[2], stride=v[3], padding=v[4])
    layers += [conv2d]
    return nn.Sequential(*layers)


def make_vgg19_block(block):
    """Builds a vgg19 block from a dictionary
    Args:
        block: a dictionary
    """
    layers = []
    for i in range(len(block)):
        one_ = block[i]
        for k, v in one_.items():
            if 'pool' in k:
                layers += [nn.MaxPool2d(kernel_size=v[0], stride=v[1],
                                        padding=v[2])]
            else:
                conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1],
                                   kernel_size=v[2], stride=v[3],
                                   padding=v[4])
                layers += [conv2d, nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)


def make_mobilenet_block(block):
    """Builds a mobilenet block from a dictionary
    Args:
        block: a dictionary
    """
    layers = []
    for i in range(len(block)):
        one_ = block[i]
        for k, v in one_.items():
            if 'bn' in k:
                layers += [conv_bn(inp=v[0], oup=v[1], stride=v[2])]
            elif 'dw' in k:
                layers += [conv_dw(inp=v[0], oup=v[1], stride=v[2])]
            else:
                conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1],
                                   kernel_size=v[3], stride=v[2],
                                   padding=v[4])
                layers += [conv2d, nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)


def get_model(trunk='vgg19'):
    """Creates the whole CPM model
    Args:
        trunk: string, 'vgg19' or 'mobilenet'
    Returns: Module, the defined model
    """
    blocks = {}
    # block0 is the preprocessing stage
    if trunk == 'vgg19':
        block0 = [{'conv1_1': [3, 64, 3, 1, 1]},
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

    elif trunk == 'mobilenet':
        block0 = [{'conv_bn': [3, 32, 2]},  # out: 3, 32, 184, 184
                  {'conv_dw1': [32, 64, 1]},  # out: 32, 64, 184, 184
                  {'conv_dw2': [64, 128, 2]},  # out: 64, 128, 92, 92
                  {'conv_dw3': [128, 128, 1]},  # out: 128, 256, 92, 92
                  {'conv_dw4': [128, 256, 2]},  # out: 256, 256, 46, 46
                  {'conv4_3_CPM': [256, 256, 1, 3, 1]},
                  {'conv4_4_CPM': [256, 128, 1, 3, 1]}]

    # Stage 1
    blocks['block1_1'] = [{'conv5_1_CPM_L1': [128, 128, 3, 1, 1]},
                          {'conv5_2_CPM_L1': [128, 128, 3, 1, 1]},
                          {'conv5_3_CPM_L1': [128, 128, 3, 1, 1]},
                          {'conv5_4_CPM_L1': [128, 512, 1, 1, 0]},
                          {'conv5_5_CPM_L1': [512, 38, 1, 1, 0]}]

    blocks['block1_2'] = [{'conv5_1_CPM_L2': [128, 128, 3, 1, 1]},
                          {'conv5_2_CPM_L2': [128, 128, 3, 1, 1]},
                          {'conv5_3_CPM_L2': [128, 128, 3, 1, 1]},
                          {'conv5_4_CPM_L2': [128, 512, 1, 1, 0]},
                          {'conv5_5_CPM_L2': [512, 19, 1, 1, 0]}]

    # Stages 2 - 6
    for i in range(2, 7):
        blocks['block%d_1' % i] = [
            {'Mconv1_stage%d_L1' % i: [185, 128, 7, 1, 3]},
            {'Mconv2_stage%d_L1' % i: [128, 128, 7, 1, 3]},
            {'Mconv3_stage%d_L1' % i: [128, 128, 7, 1, 3]},
            {'Mconv4_stage%d_L1' % i: [128, 128, 7, 1, 3]},
            {'Mconv5_stage%d_L1' % i: [128, 128, 7, 1, 3]},
            {'Mconv6_stage%d_L1' % i: [128, 128, 1, 1, 0]},
            {'Mconv7_stage%d_L1' % i: [128, 38, 1, 1, 0]}
        ]

        blocks['block%d_2' % i] = [
            {'Mconv1_stage%d_L2' % i: [185, 128, 7, 1, 3]},
            {'Mconv2_stage%d_L2' % i: [128, 128, 7, 1, 3]},
            {'Mconv3_stage%d_L2' % i: [128, 128, 7, 1, 3]},
            {'Mconv4_stage%d_L2' % i: [128, 128, 7, 1, 3]},
            {'Mconv5_stage%d_L2' % i: [128, 128, 7, 1, 3]},
            {'Mconv6_stage%d_L2' % i: [128, 128, 1, 1, 0]},
            {'Mconv7_stage%d_L2' % i: [128, 19, 1, 1, 0]}
        ]

    models = {}

    if trunk == 'vgg19':
        print("Bulding VGG19")
        models['block0'] = make_vgg19_block(block0)

    elif trunk == 'mobilenet':
        print("Building mobilenet")
        models['block0'] = make_mobilenet_block(block0)

    for k, v in blocks.iteritems():
        models[k] = make_stages(v)

    class rtpose_model(BaseModel):
        def __init__(self, model_dict):
            super(rtpose_model, self).__init__()
            self.model0 = model_dict['block0']
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

            self._initialize_weights_orthogonal()

        def forward(self, x):

            saved_for_loss = []
            out1 = self.model0(x)

            out1_1 = self.model1_1(out1)
            out1_2 = self.model1_2(out1)
            out2 = torch.cat([out1_1, out1_2, out1], 1)
            saved_for_loss.append(out1_1)
            saved_for_loss.append(out1_2)

            out2_1 = self.model2_1(out2)
            out2_2 = self.model2_2(out2)
            out3 = torch.cat([out2_1, out2_2, out1], 1)
            saved_for_loss.append(out2_1)
            saved_for_loss.append(out2_2)

            out3_1 = self.model3_1(out3)
            out3_2 = self.model3_2(out3)
            out4 = torch.cat([out3_1, out3_2, out1], 1)
            saved_for_loss.append(out3_1)
            saved_for_loss.append(out3_2)

            out4_1 = self.model4_1(out4)
            out4_2 = self.model4_2(out4)
            out5 = torch.cat([out4_1, out4_2, out1], 1)
            saved_for_loss.append(out4_1)
            saved_for_loss.append(out4_2)

            out5_1 = self.model5_1(out5)
            out5_2 = self.model5_2(out5)
            out6 = torch.cat([out5_1, out5_2, out1], 1)
            saved_for_loss.append(out5_1)
            saved_for_loss.append(out5_2)

            out6_1 = self.model6_1(out6)
            out6_2 = self.model6_2(out6)
            saved_for_loss.append(out6_1)
            saved_for_loss.append(out6_2)

            return (out6_1, out6_2), saved_for_loss

        def _initialize_weights_norm(self):

            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    init.normal(m.weight, std=0.01)
                    if m.bias is not None:  # mobilenet conv2d doesn't add bias
                        init.constant(m.bias, 0.0)

            # last layer of these block don't have Relu
            init.normal(self.model1_1[8].weight, std=0.01)
            init.normal(self.model1_2[8].weight, std=0.01)

            init.normal(self.model2_1[12].weight, std=0.01)
            init.normal(self.model3_1[12].weight, std=0.01)
            init.normal(self.model4_1[12].weight, std=0.01)
            init.normal(self.model5_1[12].weight, std=0.01)
            init.normal(self.model6_1[12].weight, std=0.01)

            init.normal(self.model2_2[12].weight, std=0.01)
            init.normal(self.model3_2[12].weight, std=0.01)
            init.normal(self.model4_2[12].weight, std=0.01)
            init.normal(self.model5_2[12].weight, std=0.01)
            init.normal(self.model6_2[12].weight, std=0.01)

        def _initialize_weights_orthogonal(self):

            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    init.orthogonal(m.weight, gain=init.calculate_gain('relu'))
                    if m.bias is not None:  # mobilenet conv2d doesn't add bias
                        init.constant(m.bias, 0.001)

            # last layer of these block don't have Relu
            init.orthogonal(self.model1_1[8].weight,
                            gain=init.calculate_gain('linear'))
            init.orthogonal(self.model1_2[8].weight,
                            gain=init.calculate_gain('linear'))

            init.orthogonal(self.model2_1[12].weight,
                            gain=init.calculate_gain('linear'))
            init.orthogonal(self.model3_1[12].weight,
                            gain=init.calculate_gain('linear'))
            init.orthogonal(self.model4_1[12].weight,
                            gain=init.calculate_gain('linear'))
            init.orthogonal(self.model5_1[12].weight,
                            gain=init.calculate_gain('linear'))
            init.orthogonal(self.model6_1[12].weight,
                            gain=init.calculate_gain('linear'))

            init.orthogonal(self.model2_2[12].weight,
                            gain=init.calculate_gain('linear'))
            init.orthogonal(self.model3_2[12].weight,
                            gain=init.calculate_gain('linear'))
            init.orthogonal(self.model4_2[12].weight,
                            gain=init.calculate_gain('linear'))
            init.orthogonal(self.model5_2[12].weight,
                            gain=init.calculate_gain('linear'))
            init.orthogonal(self.model6_2[12].weight,
                            gain=init.calculate_gain('linear'))

        @staticmethod
        def build_loss(saved_for_loss, heat_temp, heat_weight,
                       vec_temp, vec_weight, batch_size, gpus):

            names = build_names()
            saved_for_log = OrderedDict()
#            criterion = nn.MSELoss(size_average=False).cuda(gpus[0])
            criterion = nn.MSELoss(size_average=True).cuda()
            total_loss = 0
#            div1 = (torch.sum(vec_weight.data > 0) + 1) / 1000.
#            div2 = (torch.sum(heat_weight.data > 0) + 1) / 1000.
            div1 = 1.
            div2 = 1.

            for j in range(6):
                pred1 = saved_for_loss[2 * j] * vec_weight
                """
                print("pred1 sizes")
                print(saved_for_loss[2*j].data.size())
                print(vec_weight.data.size())
                print(vec_temp.data.size())
                """
                gt1 = vec_temp * vec_weight

                pred2 = saved_for_loss[2 * j + 1] * heat_weight
                gt2 = heat_weight * heat_temp
                """
                print("pred2 sizes")
                print(saved_for_loss[2*j+1].data.size())
                print(heat_weight.data.size())
                print(heat_temp.data.size())
                """

                # Compute losses
                loss1 = criterion(pred1, gt1) / div1
                loss2 = criterion(pred2, gt2) / div2

                total_loss += loss1
                total_loss += loss2
                # print(total_loss)

                # Get value from Variable and save for log
                saved_for_log[names[2 * j]] = loss1.data[0]
                saved_for_log[names[2 * j + 1]] = loss2.data[0]

            saved_for_log['max_ht'] = torch.max(
                saved_for_loss[-1].data[:, 0:-1, :, :])
            saved_for_log['min_ht'] = torch.min(
                saved_for_loss[-1].data[:, 0:-1, :, :])
            saved_for_log['max_paf'] = torch.max(saved_for_loss[-2].data)
            saved_for_log['min_paf'] = torch.min(saved_for_loss[-2].data)

            return total_loss, saved_for_log

    model = rtpose_model(models)
    return model


"""Load pretrained model on Imagenet
:param model, the PyTorch nn.Module which will train.
:param model_path, the directory which load the pretrained model, will download one if not have.
:param trunk, the feature extractor network of model.               
"""


def use_vgg(model, model_path, trunk):
    model_urls = {
        'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
        'ssd': 'https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth',
        'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'}

    number_weight = {
        'vgg16': 18,
        'ssd': 18,
        'vgg19': 20}

    url = model_urls[trunk]

    if trunk == 'ssd':
        urllib.urlretrieve('https://s3.amazonaws.com/amdegroot-models/ssd300_mAP_77.43_v2.pth',
                           os.path.join(model_path, 'ssd.pth'))
        vgg_state_dict = torch.load(os.path.join(model_path, 'ssd.pth'))
        print('loading SSD')
    else:
        vgg_state_dict = model_zoo.load_url(url, model_dir=model_path)
    vgg_keys = vgg_state_dict.keys()

    # load weights of vgg
    weights_load = {}
    # weight+bias,weight+bias.....(repeat 10 times)
    for i in range(number_weight[trunk]):
        weights_load[model.state_dict().keys()[i]
                     ] = vgg_state_dict[vgg_keys[i]]

    state = model.state_dict()
    state.update(weights_load)
    model.load_state_dict(state)
    print('load imagenet pretrained model: {}'.format(model_path))


def use_mobilenet(model):
    # get keys for mobilenet model only
    state = model.state_dict()
    model0_keys = []
    for k in state.keys():
        if 'model0' in k:
            model0_keys.append(k)

    total_layers = len(model0_keys) - 6

    # load pretrained model
    checkpoint = torch.load('/home/alberto/mobilenet_sgd_rmsprop_69.526.tar')
    checkpoint_state = checkpoint['state_dict']
    checkpoint_keys = checkpoint_state.keys()[:total_layers]

    # load weights
    weights_load = {}
    for i in range(total_layers):
        weights_load[model0_keys[i]] = checkpoint_state[checkpoint_keys[i]]

    # update model state
    state.update(weights_load)
    model.load_state_dict(state)


def build_names():
    names = []

    for j in range(1, 7):
        for k in range(1, 3):
            names.append('loss_stage%d_L%d' % (j, k))
    return names


def make_variable(tensor, async=False):
    return Variable(tensor).cuda(async=async)
