# Zhiqiang Tang, Feb 2017
import torch
import torch.nn as nn
import math
from collections import OrderedDict
import torch.utils.checkpoint as cp
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn.parameter import Parameter

def _bn_function_factory(norm, relu, conv):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = conv(relu(norm(concated_features)))
        return bottleneck_output

    return bn_function

class _Adapter(nn.Module):
    def __init__(self, num_input_features, num_output_features, efficient):
        super(_Adapter, self).__init__()
        self.add_module('adapter_norm', nn.BatchNorm2d(num_input_features))
        self.add_module('adapter_relu', nn.ReLU(inplace=True))
        self.add_module('adapter_conv', nn.Conv2d(num_input_features, num_output_features,
                                                  kernel_size=1, stride=1, bias=False))
        self.efficient = efficient
    def forward(self, prev_features):
        bn_function = _bn_function_factory(self.adapter_norm, self.adapter_relu,
                                           self.adapter_conv)
        if self.efficient and any(prev_feature.requires_grad for prev_feature in prev_features):
            adapter_output = cp.checkpoint(bn_function, *prev_features)
        else:
            adapter_output = bn_function(*prev_features)

        return adapter_output


class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, efficient=False):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate
        self.efficient = efficient

    def forward(self, prev_features):
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
        # print type(prev_features), type(prev_features[0]), type(prev_features[0][0])
        # for prev_feature in prev_features:
        #     print 'prev_feature type: ', type(prev_feature)
        #     print 'prev_feature size: ', prev_feature.size()
        if self.efficient and any(prev_fea.requires_grad for prev_fea in prev_features):
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features

class _DenseBlock(nn.Module):
    def __init__(self, in_num, growth_rate, neck_size, layer_num, max_link,
                 drop_rate=0, efficient=True, requires_skip=True, is_up=False):

        self.saved_features = []
        self.max_link = max_link
        self.requires_skip = requires_skip
        super(_DenseBlock, self).__init__()
        max_in_num = in_num + max_link * growth_rate
        self.final_num_features = max_in_num
        self.layers = nn.ModuleList()
        print('layer number is %d' % layer_num)
        for i in range(0, layer_num):
            if i < max_link:
                tmp_in_num = in_num + i * growth_rate
            else:
                tmp_in_num = max_in_num
            print('layer %d input channel number is %d' % (i, tmp_in_num))
            self.layers.append(_DenseLayer(tmp_in_num, growth_rate=growth_rate,
                                           bn_size=neck_size, drop_rate=drop_rate,
                                           efficient=efficient))
        # self.layers = nn.ModuleList(self.layers)

        self.adapters_ahead = nn.ModuleList()
        adapter_in_nums = []
        adapter_out_num = in_num
        if is_up:
            adapter_out_num = adapter_out_num / 2
        for i in range(0, layer_num):
            if i < max_link:
                tmp_in_num = in_num + (i+1) * growth_rate
            else:
                tmp_in_num = max_in_num + growth_rate
            adapter_in_nums.append(tmp_in_num)
            print('adapter %d input channel number is %d' % (i, adapter_in_nums[i]))
            self.adapters_ahead.append(_Adapter(adapter_in_nums[i], adapter_out_num,
                                                efficient=efficient))
        # self.adapters_ahead = nn.ModuleList(self.adapters_ahead)
        print('adapter output channel number is %d' % adapter_out_num)

        if requires_skip:
            print('creating skip layers ...')
            self.adapters_skip = nn.ModuleList()
            for i in range(0, layer_num):
                self.adapters_skip.append(_Adapter(adapter_in_nums[i], adapter_out_num,
                                                   efficient=efficient))
            # self.adapters_skip = nn.ModuleList(self.adapters_skip)

    def forward(self, x, i):
        if i == 0:
            self.saved_features = []

        if type(x) is torch.Tensor:
            x = [x]
        if type(x) is not list:
            raise Exception('type(x) should be list, but it is: ', type(x))
        # for t_x in x:
        #     print 't_x type: ', type(t_x)
        #     print 't_x size: ', t_x.size()

        x = x + self.saved_features
        # for t_x in x:
        #     print 't_x type: ', type(t_x)
        #     print 't_x size: ', t_x.size()

        out = self.layers[i](x)
        if i < self.max_link:
            self.saved_features.append(out)
        elif len(self.saved_features) != 0:
            self.saved_features.pop(0)
            self.saved_features.append(out)
        x.append(out)
        out_ahead = self.adapters_ahead[i](x)
        if self.requires_skip:
            out_skip = self.adapters_skip[i](x)
            return out_ahead, out_skip
        else:
            return out_ahead

class _IntermediaBlock(nn.Module):
    def __init__(self, in_num, out_num, layer_num, max_link, efficient=True):

        max_in_num = in_num + max_link * out_num
        self.final_num_features = max_in_num
        self.saved_features = []
        self.max_link = max_link
        super(_IntermediaBlock, self).__init__()
        print('creating intermedia block ...')
        self.adapters = nn.ModuleList()
        for i in range(0, layer_num-1):
            if i < max_link:
                tmp_in_num = in_num + (i+1) * out_num
            else:
                tmp_in_num = max_in_num
            print('intermedia layer %d input channel number is %d' % (i, tmp_in_num))
            self.adapters.append(_Adapter(tmp_in_num, out_num, efficient=efficient))
        # self.adapters = nn.ModuleList(self.adapters)
        print('intermedia layer output channel number is %d' % out_num)

    def forward(self, x, i):
        if i == 0:
            self.saved_features = []
            if type(x) is torch.Tensor:
                if self.max_link != 0:
                    self.saved_features.append(x)
            elif type(x) is list:
                if self.max_link != 0:
                    self.saved_features = self.saved_features + x
            return x

        if type(x) is torch.Tensor:
            x = [x]
        if type(x) is not list:
            raise Exception('type(x) should be list, but it is: ', type(x))

        x = x + self.saved_features
        out = self.adapters[i-1](x)
        if i < self.max_link:
            self.saved_features.append(out)
        elif len(self.saved_features) != 0:
            self.saved_features.pop(0)
            self.saved_features.append(out)
        # print('middle list length is %d' % len(self.saved_features))
        return out

class _Bn_Relu_Conv1x1(nn.Sequential):
    def __init__(self, in_num, out_num):
        super(_Bn_Relu_Conv1x1, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(in_num))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(in_num, out_num, kernel_size=1,
                                          stride=1, bias=False))

# class _TransitionDown(nn.Module):
#     def __init__(self, in_num_list, out_num, num_units):
#         super(_TransitionDown, self).__init__()
#         self.adapters = []
#         for i in range(0, num_units):
#             self.adapters.append(_Bn_Relu_Conv1x1(in_num=in_num_list[i], out_num=out_num))
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
#
#     def forward(self, x, i):
#         x = self.adapters[i](x)
#         out = self.pool(x)
#         return out
#
# class _TransitionUp(nn.Module):
#     def __init__(self, in_num_list, out_num_list, num_units):
#         super(_TransitionUp, self).__init__()
#         self.adapters = []
#         for i in range(0, num_units):
#             self.adapters.append(_Bn_Relu_Conv1x1(in_num=in_num_list[i], out_num=out_num_list[i]))
#         self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
#
#     def forward(self, x, i):
#         x = self.adapters[i](x)
#         out = self.upsample(x)
#         return out


class _CU_Net(nn.Module):
    def __init__(self, in_num, neck_size, growth_rate, layer_num, max_link):
        super(_CU_Net, self).__init__()
        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.num_blocks = 4
        print('creating hg ...')
        for i in range(0, self.num_blocks):
            print('creating down block %d ...' % i)
            self.down_blocks.append(_DenseBlock(in_num=in_num, neck_size=neck_size,
                                    growth_rate=growth_rate, layer_num=layer_num,
                                    max_link=max_link, requires_skip=True))
            print('creating up block %d ...' % i)
            self.up_blocks.append(_DenseBlock(in_num=in_num*2, neck_size=neck_size,
                                  growth_rate=growth_rate, layer_num=layer_num,
                                  max_link=max_link, requires_skip=False, is_up=True))
        # self.down_blocks = nn.ModuleList(self.down_blocks)
        # self.up_blocks = nn.ModuleList(self.up_blocks)
        print('creating neck block ...')
        self.neck_block = _DenseBlock(in_num=in_num, neck_size=neck_size,
                                      growth_rate=growth_rate, layer_num=layer_num,
                                      max_link=max_link, requires_skip=False)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2)

    def forward(self, x, i):
        skip_list = [None] * self.num_blocks
        # print 'input x size is ', x.size()
        for j in range(0, self.num_blocks):
            # print('using down block %d ...' % j)
            x, skip_list[j] = self.down_blocks[j](x, i)
            # print 'output size is ', x.size()
            # print 'skip size is ', skip_list[j].size()
            x = self.maxpool(x)
        # print('using neck block ...')
        x = self.neck_block(x, i)
        # print 'output size is ', x.size()
        for j in list(reversed(range(0, self.num_blocks))):
            x = self.upsample(x)
            # print('using up block %d ...' % j)
            x = self.up_blocks[j]([x, skip_list[j]], i)
            # print 'output size is ', x.size()
        return x

class _CU_Net_Wrapper(nn.Module):
    def __init__(self, init_chan_num, neck_size, growth_rate,
                 class_num, layer_num, order, loss_num):
        assert loss_num <= layer_num and loss_num >= 1
        loss_every = float(layer_num) / float(loss_num)
        self.loss_achors = []
        for i in range(0, loss_num):
            tmp_achor = int(round(loss_every * (i+1)))
            if tmp_achor <= layer_num:
                self.loss_achors.append(tmp_achor)

        assert layer_num in self.loss_achors
        assert loss_num == len(self.loss_achors)

        if order >= layer_num:
            print 'order is larger than the layer number.'
            exit()
        print('layer number is %d' % layer_num)
        print('loss number is %d' % loss_num)
        print('loss achors are: ', self.loss_achors)
        print('order is %d' % order)
        print('growth rate is %d' % growth_rate)
        print('neck size is %d' % neck_size)
        print('class number is %d' % class_num)
        print('initial channel number is %d' % init_chan_num)
        num_chans = init_chan_num
        super(_CU_Net_Wrapper, self).__init__()
        self.layer_num = layer_num
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, init_chan_num, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(init_chan_num)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=2, stride=2)),
        ]))
        # self.denseblock0 = _DenseBlock(layer_num=4, in_num=init_chan_num,
        #                                neck_size=neck_size, growth_rate=growth_rate)
        # hg_in_num = init_chan_num + growth_rate * 4
        print('channel number is %d' % num_chans)
        self.hg = _CU_Net(in_num=num_chans, neck_size=neck_size, growth_rate=growth_rate,
                             layer_num=layer_num, max_link=order)

        self.linears = nn.ModuleList()
        for i in range(0, layer_num):
            self.linears.append(_Bn_Relu_Conv1x1(in_num=num_chans, out_num=class_num))
        # self.linears = nn.ModuleList(self.linears)
        # intermedia_in_nums = []
        # for i in range(0, num_units-1):
        #     intermedia_in_nums.append(num_chans * (i+2))
        self.intermedia = _IntermediaBlock(in_num=num_chans, out_num=num_chans,
                                           layer_num=layer_num, max_link=order)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                stdv = 1/math.sqrt(n)
                m.weight.data.uniform_(-stdv, stdv)
                # m.weight.data.zero_()
                if m.bias is not None:
                    m.bias.data.uniform_(-stdv, stdv)
                    # m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.uniform_()
                # m.weight.data.zero_()
                m.bias.data.zero_()

    def forward(self, x):
        # print(x.size())
        x = self.features(x)
        # print(x.size())
        # x = self.denseblock0(x)
        # print 'x size is', x.size()
        out = []
        # middle = []
        # middle.append(x)
        for i in range(0, self.layer_num):
            # print('using intermedia layer %d ...' % i)
            x = self.intermedia(x, i)
            # print 'x size after intermedia layer is ', x.size()
            # print('using hg %d ...' % i)
            x = self.hg(x, i)
            # print 'x size after hg is ', x.size()
            # middle.append(x)
            if (i+1) in self.loss_achors:
                tmp_out = self.linears[i](x)
                # print 'tmp output size is ', tmp_out.size()
                out.append(tmp_out)
            # if i < self.num_units-1:
        # exit()
        assert len(self.loss_achors) == len(out)
        return out

def create_cu_net(neck_size, growth_rate, init_chan_num,
                  class_num, layer_num, order, loss_num):

    net = _CU_Net_Wrapper(init_chan_num=init_chan_num, neck_size=neck_size,
                          growth_rate=growth_rate, class_num=class_num,
                          layer_num=layer_num, order=order, loss_num=loss_num)
    return net

