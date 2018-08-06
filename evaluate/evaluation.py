import unittest
import torch
from evaluate.coco_eval import run_eval
from network.rtpose_vgg import get_model, use_vgg
from torch import load

with torch.autograd.no_grad():
    # this path is with respect to the root of the project
    weight_name = './network/weight/best_pose.pth'
    state_dict = torch.load(weight_name)
    model = get_model(trunk='vgg19')

    model.load_state_dict(state_dict)
    model.eval()
    model.float()
    model = model.cuda()
    
    # The choice of image preprocessing include: 'rtpose', 'inception', 'vgg' and 'ssd'.
    # 'our_post_processing' using the ported Matlab post processing, which got slightly higher score
    # else using the ported C++ post-processing, which has amazing speed.
    run_eval(image_dir= '/data/coco/images/', anno_dir = '/data/coco', vis_dir = '/data/coco/vis',
        image_list_txt='./evaluate/image_info_val2014_1k.txt', 
        model=model, preprocess='vgg')


