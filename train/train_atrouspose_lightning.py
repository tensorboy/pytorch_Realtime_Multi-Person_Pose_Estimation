import argparse
import time
import os
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim.lr_scheduler as lr_scheduler
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from test_tube import Experiment

from lib.network.atrouspose import AtrousPose
from lib.datasets import coco, transforms, datasets
from lib.config import update_config

DATA_DIR = '/home/tensorboy/data/coco'

ANNOTATIONS_TRAIN = [os.path.join(DATA_DIR, 'annotations', item) for item in ['person_keypoints_train2017.json']]
ANNOTATIONS_VAL = os.path.join(DATA_DIR, 'annotations', 'person_keypoints_val2017.json')
IMAGE_DIR_TRAIN = os.path.join(DATA_DIR, 'images/train2017')
IMAGE_DIR_VAL = os.path.join(DATA_DIR, 'images/val2017')


def train_cli(parser):
    group = parser.add_argument_group('dataset and loader')
    group.add_argument('--train-annotations', default=ANNOTATIONS_TRAIN)
    group.add_argument('--train-image-dir', default=IMAGE_DIR_TRAIN)
    group.add_argument('--val-annotations', default=ANNOTATIONS_VAL)
    group.add_argument('--val-image-dir', default=IMAGE_DIR_VAL)
    group.add_argument('--pre-n-images', default=8000, type=int,
                       help='number of images to sampe for pretraining')
    group.add_argument('--n-images', default=None, type=int,
                       help='number of images to sample')
    group.add_argument('--duplicate-data', default=None, type=int,
                       help='duplicate data')
    group.add_argument('--loader-workers', default=8, type=int,
                       help='number of workers for data loading')
    group.add_argument('--batch-size', default=72, type=int,
                       help='batch size')
    group.add_argument('--lr', '--learning-rate', default=1., type=float,
                    metavar='LR', help='initial learning rate')
    group.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
    group.add_argument('--weight-decay', '--wd', default=0.000, type=float,
                    metavar='W', help='weight decay (default: 1e-4)') 
    group.add_argument('--nesterov', dest='nesterov', default=True, type=bool)     
    group.add_argument('--print_freq', default=20, type=int, metavar='N',
                    help='number of iterations to print the training statistics')    
   

def cli():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    train_cli(parser)
    parser.add_argument('-o', '--output', default=None,
                        help='output file')
    parser.add_argument('--stride-apply', default=1, type=int,
                        help='apply and reset gradients every n batches')
    parser.add_argument('--epochs', default=75, type=int,
                        help='number of epochs to train')
    parser.add_argument('--freeze-base', default=0, type=int,
                        help='number of epochs to train with frozen base')
    parser.add_argument('--pre-lr', type=float, default=1e-4,
                        help='pre learning rate')
    parser.add_argument('--update-batchnorm-runningstatistics',
                        default=False, action='store_true',
                        help='update batch norm running statistics')
    parser.add_argument('--square-edge', default=368, type=int,
                        help='square edge of input images')
    parser.add_argument('--ema', default=1e-3, type=float,
                        help='ema decay constant')
    parser.add_argument('--debug-without-plots', default=False, action='store_true',
                        help='enable debug but dont plot')
    parser.add_argument('--disable-cuda', action='store_true',
                        help='disable CUDA')    
    parser.add_argument('--log_dir', default='/home/tensorboy/data/rtpose/', type=str, metavar='DIR',
                    help='path to where the model saved')                                              
    parser.add_argument('--model_path', default='./network/weight/', type=str, metavar='DIR',
                    help='path to where the model saved')                         
    args = parser.parse_args()

    # add args.device
    args.device = torch.device('cpu')
    args.pin_memory = False
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        args.pin_memory = True
        
    return args


args = cli()

print("Loading dataset...")
# load train data
preprocess = transforms.Compose([
        transforms.Normalize(),
        transforms.RandomApply(transforms.HFlip(), 0.5),
        transforms.RescaleRelative(),
        transforms.Crop(args.square_edge),
        transforms.CenterPad(args.square_edge),
    ])

class rtpose_lightning(pl.LightningModule):

    def __init__(self, args, preprocess, target_transforms, model):
        super(rtpose_lightning, self).__init__()
        
        self.args = args
        self.preprocess = preprocess
        self.model = model
        self.target_transforms = target_transforms

    def forward(self, x):
        paf_output, heatmap_output = self.model.forward(x)
        return paf_output, heatmap_output

    def l2_loss(self, heatmap_output, heatmap_target, paf_output, paf_target):
    
        loss_dict = OrderedDict()
        
        total_loss = 0
      
        loss1 = F.mse_loss(paf_output, paf_target, reduction='mean')
        total_loss += loss1
        
        loss2 = F.mse_loss(heatmap_output, heatmap_target, reduction='mean')
        total_loss += loss2    
                                
        loss_dict['loss'] = total_loss.unsqueeze(0)            
        loss_dict['max_heatmap'] = torch.max(heatmap_output.data[:, :-1, :, :]).unsqueeze(0)
        loss_dict['min_heatmap'] = torch.min(heatmap_output.data[:, :-1, :, :]).unsqueeze(0)
        loss_dict['max_paf'] = torch.max(paf_output.data).unsqueeze(0)
        loss_dict['min_paf'] = torch.min(paf_output.data).unsqueeze(0)                       
 
        return loss_dict
        
    def training_step(self, batch, batch_nb):
    
        img, heatmap_target, paf_target = batch
        paf_output, heatmap_output = self.forward(img)
        loss_dict = self.l2_loss(heatmap_output, heatmap_target, paf_output, paf_target)
           
        output = {
            'loss': loss_dict['loss'], # required
            'prog': loss_dict # optional
        }        
        return output

    def validation_step(self, batch, batch_nb):
        img, heatmap_target, paf_target = batch
        paf_output, heatmap_output = self.forward(img)
        
        loss_dict = self.l2_loss(heatmap_output, heatmap_target, paf_output, paf_target)
        loss_dict['val_loss'] = loss_dict['loss'] 
        return loss_dict
        
    def validation_end(self, outputs):
        output_dict = OrderedDict()       
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        output_dict['avg_val_loss'] = avg_loss
      
        return output_dict

    def configure_optimizers(self):
    
        optimizer = torch.optim.SGD(self.parameters(), lr=self.args.lr,
                           momentum=self.args.momentum,
                           weight_decay=self.args.weight_decay,
                           nesterov=self.args.nesterov)    
                             
        #scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5, \
        #            verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=3,\
        #            min_lr=0, eps=1e-08)   
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)        
        return [[optimizer], [scheduler]]

    @pl.data_loader
    def tng_dataloader(self):
        train_datas = [datasets.CocoKeypoints(
            root=self.args.train_image_dir,
            annFile=item,
            preprocess=preprocess,
            image_transform=transforms.image_transform_train,
            target_transforms=self.target_transforms,
            n_images=args.n_images,
            ) for item in self.args.train_annotations]

        train_data = torch.utils.data.ConcatDataset(train_datas)
        
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=self.args.batch_size, shuffle=True,
            pin_memory=self.args.pin_memory, num_workers=self.args.loader_workers, drop_last=True)
            
        return train_loader

    @pl.data_loader
    def val_dataloader(self):
        val_data = datasets.CocoKeypoints(
            root=self.args.val_image_dir,
            annFile=self.args.val_annotations,
            preprocess=preprocess,
            image_transform=transforms.image_transform_train,
            target_transforms=self.target_transforms,
            n_images=self.args.n_images,
        )
        val_loader = torch.utils.data.DataLoader(
            val_data, batch_size=self.args.batch_size, shuffle=False,
            pin_memory=self.args.pin_memory, num_workers=self.args.loader_workers, drop_last=True)
        
        return val_loader

    @pl.data_loader
    def test_dataloader(self):
        val_data = datasets.CocoKeypoints(
            root=self.args.val_image_dir,
            annFile=self.args.val_annotations,
            preprocess=preprocess,
            image_transform=transforms.image_transform_train,
            target_transforms=self.target_transforms,
            n_images=self.args.n_images,
        )
        val_loader = torch.utils.data.DataLoader(
            val_data, batch_size=self.args.batch_size, shuffle=False,
            pin_memory=self.args.pin_memory, num_workers=self.args.loader_workers, drop_last=True)
        
        return val_loader

# model
atrouspose = AtrousPose(paf_out_channels=38, heat_out_channels=19)
# load pretrained
model = rtpose_lightning(args, preprocess, target_transforms=None, model = atrouspose)
exp = Experiment(name='atrouspose', save_dir=args.log_dir)

# callbacks
early_stop = EarlyStopping(
    monitor='avg_val_loss',
    patience=20,
    verbose=True,
    mode='min'
)

model_save_path = '{}/{}/{}'.format(args.log_dir, exp.name, exp.version)
checkpoint = ModelCheckpoint(
    filepath=model_save_path,
    save_best_only=True,
    verbose=True,
    monitor='avg_val_loss',
    mode='min'
)

trainer = Trainer(experiment=exp, \
                  max_nb_epochs=100, \
                  gpus=[0,1,2,3], \
                  checkpoint_callback=checkpoint,
                  early_stop_callback=early_stop)

trainer.fit(model)
          
