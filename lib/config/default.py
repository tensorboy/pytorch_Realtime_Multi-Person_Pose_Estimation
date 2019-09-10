from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from yacs.config import CfgNode as CN


_C = CN()

_C.OUTPUT_DIR = ''
_C.LOG_DIR = ''
_C.EXPERIMENT_NAME = ''
_C.DATA_DIR = ''
_C.GPUS = [0,1,2,3]
_C.WORKERS = 4
_C.PRINT_FREQ = 20
_C.PIN_MEMORY = True
_C.RANK = 0

# Cudnn related params
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

# common params for NETWORK
_C.MODEL = CN()
_C.MODEL.NAME = 'pose_hrnet'
_C.MODEL.INIT_WEIGHTS = True
_C.MODEL.PRETRAINED = ''
_C.MODEL.NUM_JOINTS = 17
_C.MODEL.TAG_PER_JOINT = True
_C.MODEL.TARGET_TYPE = 'gaussian'
_C.MODEL.IMAGE_SIZE = [256, 256]  # width * height, ex: 192 * 256
_C.MODEL.HEATMAP_SIZE = [64, 64]  # width * height, ex: 24 * 32
_C.MODEL.SIGMA = 2
_C.MODEL.EXTRA = CN(new_allowed=True)
_C.MODEL.NUM_KEYPOINTS = 18
_C.MODEL.DOWNSAMPLE = 8

_C.LOSS = CN()
_C.LOSS.USE_OHKM = False
_C.LOSS.TOPK = 8
_C.LOSS.USE_TARGET_WEIGHT = True
_C.LOSS.USE_DIFFERENT_JOINTS_WEIGHT = False

# DATASET related params
_C.DATASET = CN()
_C.DATASET.ROOT = ''
_C.DATASET.DATASET = 'mpii'
_C.DATASET.TRAIN_SET = 'train'
_C.DATASET.TEST_SET = 'valid'
_C.DATASET.DATA_FORMAT = 'jpg'
_C.DATASET.HYBRID_JOINTS_TYPE = ''
_C.DATASET.TRAIN_IMAGE_DIR = ''
_C.DATASET.TRAIN_ANNOTATIONS = []
_C.DATASET.VAL_IMAGE_DIR = ''
_C.DATASET.VAL_ANNOTATIONS = ''

# training data augmentation
_C.DATASET.FLIP = True
_C.DATASET.SCALE_FACTOR = 0.25
_C.DATASET.ROT_FACTOR = 30
_C.DATASET.SCALE_MIN = 0.5
_C.DATASET.SCALE_MAX =  1.1   
_C.DATASET.COLOR_RGB = False
_C.DATASET.IMAGE_SIZE = 368  

# train
_C.PRE_TRAIN = CN()
_C.PRE_TRAIN.LR = 1.0
_C.PRE_TRAIN.OPTIMIZER = 'adam'
_C.PRE_TRAIN.MOMENTUM = 0.9
_C.PRE_TRAIN.WD = 0.0001
_C.PRE_TRAIN.NESTEROV = False
_C.PRE_TRAIN.FREEZE_BASE_EPOCHS = 5
# train
_C.TRAIN = CN()

_C.TRAIN.LR_FACTOR = 0.1
_C.TRAIN.LR_STEP = [90, 110]
_C.TRAIN.LR = 0.001

_C.TRAIN.OPTIMIZER = 'adam'
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.WD = 0.0001
_C.TRAIN.NESTEROV = False
_C.TRAIN.GAMMA1 = 0.99
_C.TRAIN.GAMMA2 = 0.0

_C.TRAIN.EPOCHS = 140

#'apply and reset gradients every n batches'
_C.TRAIN.STRIDE_APPLY = 1

_C.TRAIN.RESUME = False
_C.TRAIN.CHECKPOINT = ''

_C.TRAIN.BATCH_SIZE_PER_GPU = 32
_C.TRAIN.SHUFFLE = True

# testing
_C.TEST = CN()

# size of images for each device
_C.TEST.BATCH_SIZE_PER_GPU = 32
# Test Model Epoch
_C.TEST.FLIP_TEST = False
_C.TEST.POST_PROCESS = False
_C.TEST.SHIFT_HEATMAP = False

_C.TEST.USE_GT_BBOX = False

# nms
_C.TEST.IMAGE_THRE = 0.1
_C.TEST.NMS_THRE = 0.6
_C.TEST.SOFT_NMS = False
_C.TEST.OKS_THRE = 0.5
_C.TEST.IN_VIS_THRE = 0.0
_C.TEST.COCO_BBOX_FILE = ''
_C.TEST.BBOX_THRE = 1.0
_C.TEST.MODEL_FILE = ''

_C.TEST.THRESH_HEATMAP =  0.1
_C.TEST.THRESH_PAF= 0.05
_C.TEST.NUM_INTERMED_PTS_BETWEEN_KEYPOINTS= 10 
  
# debug
_C.DEBUG = CN()
_C.DEBUG.DEBUG = False
_C.DEBUG.SAVE_BATCH_IMAGES_GT = False
_C.DEBUG.SAVE_BATCH_IMAGES_PRED = False
_C.DEBUG.SAVE_HEATMAPS_GT = False
_C.DEBUG.SAVE_HEATMAPS_PRED = False


def update_config(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    cfg.DATASET.ROOT = os.path.join(
        cfg.DATA_DIR, cfg.DATASET.DATASET
    )
    cfg.DATASET.TRAIN_IMAGE_DIR= os.path.join(
        cfg.DATASET.ROOT, cfg.DATASET.TRAIN_IMAGE_DIR
    )
    cfg.DATASET.VAL_IMAGE_DIR = os.path.join(
        cfg.DATASET.ROOT, cfg.DATASET.VAL_IMAGE_DIR
    )
    cfg.DATASET.TRAIN_ANNOTATIONS= [os.path.join(
        cfg.DATASET.ROOT, item) for item in cfg.DATASET.TRAIN_ANNOTATIONS]
        
    cfg.DATASET.VAL_ANNOTATIONS = os.path.join(
        cfg.DATASET.ROOT, cfg.DATASET.VAL_ANNOTATIONS
    )  
    cfg.MODEL.PRETRAINED = os.path.join(
        cfg.DATA_DIR, cfg.MODEL.PRETRAINED
    )

    if cfg.TEST.MODEL_FILE:
        cfg.TEST.MODEL_FILE = os.path.join(
            cfg.DATA_DIR, cfg.TEST.MODEL_FILE
        )

    cfg.freeze()


if __name__ == '__main__':
    import sys
    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)

