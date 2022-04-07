import copy
import json
import os
import numpy as np
import torch

from PIL import Image
import cv2

# with open('D:\zjlab\pytorch_Realtime_Multi-Person_Pose_Estimation\data\coco\images\\test2017\\000000000183.jpg', 'rb') as f:
#     # original images of different sizes, ex: (480, 640), (500, 375), (501, 640)
#     image = Image.open(f)
#     img1 = image.crop((30 ,100 ,123 ,345))
#     img1.size
#
#
# img = np.array([1 ,2 ,3])
# print(img.shape)


from lib.datasets import datasets
from lib.datasets import transforms

# keypoints = np.array([[[553.95604396, 484.61538462],
#                        [486.26373626, 493.40659341],
#                        [0., 0.],
#                        [0., 0.],
#                        [0., 0.]],
#                       [[443.95604396, 484.61538462],
#                        [486.26373626, 493.40659341],
#                        [0., 0.],
#                        [0., 0.],
#                        [0., 0.]]]
#                      )
#
# MAGIC_CONSTANT = (-1, -1)
# mask = np.logical_or.reduce((keypoints[:, :, 0] >= 500,
#                              keypoints[:, :, 0] < 0,
#                              keypoints[:, :, 1] >= 500,
#                              keypoints[:, :, 1] < 0))
# print(keypoints.shape)
# print(keypoints[mask])
# keypoints[mask] = MAGIC_CONSTANT
# print(keypoints[mask])

path = "D:\zjlab\pytorch_Realtime_Multi-Person_Pose_Estimation\data\\bean\\2055_1\Basler_acA4112-20uc__40059801__20201013_143217765_316.json"
f = open(path)
anns = json.load(f)
del anns['imageData']

shapes = anns['shapes']
dic = {}
for sh in shapes:
    item = copy.deepcopy(sh)
    del item['group_id']
    del item['shape_type']
    del item['flags']
    dic.setdefault(sh['group_id'], []).append(item)

for k, v in dic.items():
    v.sort(key=lambda i: i['label'])

res = []
for k, v in dic.items():
    bean = {'group_id': k,
            'keypoints': np.asarray([item['points'][0] for item in v], dtype=np.float32),
            'unknown_count': v[0]['label'] == '0-0'}
    res.append(bean)
anns = res
print(anns)
keypoints = []
for ann in anns:
    single_keypoints = ann['keypoints']
    if len(single_keypoints) < 5:
        single_keypoints += [[0, 0] for _ in range(5 - len(single_keypoints))]
    keypoints.append(single_keypoints)

keypoints = np.array(keypoints)


print(anns)