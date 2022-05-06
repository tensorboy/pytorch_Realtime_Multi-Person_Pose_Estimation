import torch
from torch.utils import model_zoo

from lib.network.openpose import OpenPose_Model, use_vgg
from lib.network.rtpose_vgg import get_model

model = OpenPose_Model()
model = torch.nn.DataParallel(model).cuda()
# load pretrained
use_vgg(model)

# Fix the VGG weights first, and then the weights will be released
# for i in range(20):
#     for param in model.module.parameters():
#         print(param)
#         param.requires_grad = False




def use_vgg(model):
    url = 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'
    vgg_state_dict = model_zoo.load_url(url)
    vgg_keys = vgg_state_dict.keys()

    weights_load = {}
    print(list(model.state_dict()))
    print(list(vgg_keys))
    for i in range(20):
        print(list(model.state_dict().keys())[i])
        print(list(vgg_keys)[i])
        weights_load[list(model.state_dict().keys())[i]] = vgg_state_dict[list(vgg_keys)[i]]

    #
    # # load weights of vgg
    # weights_load = {}
    # # weight+bias,weight+bias.....(repeat 10 times)
    # for i in range(20):
    #     weights_load[list(model.state_dict().keys())[i]] = vgg_state_dict[list(vgg_keys)[i]]
    #
    # state = model.state_dict()
    # state.update(weights_load)
    # model.load_state_dict(state)
    # print('load imagenet pretrained model')

model = get_model(dataset='coco', trunk='vgg19')
model = torch.nn.DataParallel(model).cuda()
use_vgg(model)