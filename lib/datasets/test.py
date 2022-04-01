
import os
import numpy as np

from PIL import Image
import cv2

with open('D:\zjlab\pytorch_Realtime_Multi-Person_Pose_Estimation\data\coco\images\\test2017\\000000000183.jpg', 'rb') as f:
    # original images of different sizes, ex: (480, 640), (500, 375), (501, 640)
    image = Image.open(f)
    img1 = image.crop((30 ,100 ,123 ,345))
    img1.size


img = np.array([1 ,2 ,3])
print(img.shape)