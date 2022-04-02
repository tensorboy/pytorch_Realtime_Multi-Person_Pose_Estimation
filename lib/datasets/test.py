
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

path = 'D:\zjlab\pytorch_Realtime_Multi-Person_Pose_Estimation\data\\bean\\2055_1'

preprocess = transforms.Compose([
        transforms.Normalize(),
        transforms.RescaleRelative(),
        transforms.Crop(368),  # 368
        transforms.CenterPad(368),  # 368
    ])

train_data = [datasets.SoybeanKeypoints(
        root=path,
        preprocess=preprocess,
        image_transform=transforms.image_transform_train,
        target_transforms=None
    )]

train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=20, shuffle=True,
        pin_memory=True, num_workers=8, drop_last=True
    )

val_data = datasets.SoybeanKeypoints(
    root=path,
    preprocess=preprocess,
    image_transform=transforms.image_transform_train,
    target_transforms=None
)

val_loader = torch.utils.data.DataLoader(
    val_data, batch_size=20, shuffle=False,
    pin_memory=True, num_workers=8, drop_last=True)

