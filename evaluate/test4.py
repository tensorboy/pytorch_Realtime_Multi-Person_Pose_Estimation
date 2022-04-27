import random
from collections import defaultdict

import numpy as np
from pycocotools.coco import COCO
from scipy import ndimage, misc
import matplotlib.pyplot as plt
from scipy.ndimage import generate_binary_structure

annFile = 'D:\zjlab\pytorch_Realtime_Multi-Person_Pose_Estimation\data\coco\\annotations\person_keypoints_val2017.json'
cocoGt = COCO(annFile)
catIds = cocoGt.getCatIds()
print('catIds:', catIds)
imgIds = sorted(cocoGt.getImgIds())
gts = cocoGt.loadAnns(cocoGt.getAnnIds(imgIds=imgIds, catIds=catIds))


print(gts[0])