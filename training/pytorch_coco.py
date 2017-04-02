import os
import os.path
import torch.utils.data as data
from PIL import Image
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.models as models


class CocoKepoints(data.Dataset):

    def __init__(self, root, annFile, transform=None, target_transform=None):
        from pycocotools.coco import COCO
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)[0]
        
        path = coco.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.ids)
        

cap = CocoKepoints(root = '/media/data2/lab/construct/pytorch_pose/training/dataset/COCO/images/val2014',
                        annFile = '/media/data2/lab/construct/pytorch_pose/training/dataset/COCO/annotations/person_keypoints_val2014.json',
                        transform=transforms.ToTensor())
                        
                        

print('Number of samples: ', len(cap))
img, target = cap[3] # load 4th sample

print("Image Size: ", img.size())
print(target.keys())

vgg19 = models.vgg19()
