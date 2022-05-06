import sys, os

sys.path.append(os.path.abspath("./"))
import argparse
from lib.config import update_config, cfg
import torch
from evaluate import bean_eval, coco_eval
from lib.network.rtpose_vgg import get_model
from lib.network.openpose import OpenPose_Model

SOURCE_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

NOW_WHAT = 'bean'
NEW_OPENPOSE = False

parser = argparse.ArgumentParser()

args = parser.parse_args()

if NOW_WHAT == 'coco':
    parser.add_argument('--cfg', help='experiment configure file name',
                        default='./experiments/vgg19_368x368_sgd.yaml', type=str)
    args = parser.parse_args()
    update_config(cfg, args)
    weight_path = 'network/weight/4.8/best_pose.pth'
    image_dir = 'data/coco/images/val2017'
    anno_file = 'data/coco/annotations/person_keypoints_val2017.json'
    vis_dir = 'data/coco/images/vis_val2017'

elif NOW_WHAT == 'bean':
    parser.add_argument('--cfg', help='experiment configure file name',
                        default=os.path.join(SOURCE_DIR, 'experiments/vgg19_368x368_sgd_bean.yaml'), type=str)
    args = parser.parse_args()
    update_config(cfg, args)
    weight_path = os.path.join(SOURCE_DIR, cfg.MODEL.TRAINED)
    image_dir = os.path.join(SOURCE_DIR, cfg.DATASET.TEST_IMAGE_DIR)
    vis_dir = os.path.join(SOURCE_DIR, cfg.OUTPUT_DIR)
    anno_file = None

# update config file
update_config(cfg, args)

# Notice, if you using the
with torch.autograd.no_grad():

    if NEW_OPENPOSE:
        model = OpenPose_Model(l2_stages=4, l1_stages=2, paf_out_channels=8, heat_out_channels=6)
        model = torch.nn.DataParallel(model).cuda()
        wts_dict = torch.load(weight_path)

    else:
        model = get_model(trunk='vgg19', dataset=NOW_WHAT)
        model = torch.nn.DataParallel(model).cuda()
        wts_dict = torch.load(weight_path)
        # wts_dict = {k.replace('module.', ''): v for k, v in wts_dict.items()}

    model.load_state_dict(wts_dict)
    model.eval()
    model.float()
    model = model.cuda()

    # The choice of image preprocessing include: 'rtpose', 'inception', 'vgg19' and 'ssd'.
    # If you use the converted model from caffe, it is 'rtpose' preprocess, the model trained in
    # this repo used 'vgg19' preprocess
    if NOW_WHAT == 'coco':
        avg_precision = coco_eval.run_eval(image_dir=image_dir, anno_file=anno_file, vis_dir=vis_dir, model=model, preprocess='vgg19')
        print("Average precision:", avg_precision)
    elif NOW_WHAT == 'bean':
        bean_eval.run_eval(image_dir=image_dir, model=model, vis_dir=vis_dir, preprocess='vgg19')