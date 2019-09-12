import argparse
import time
import os
import re
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

import _init_paths
from lib.network.rtpose_vgg import get_model, use_vgg
from lib.datasets import coco, transforms, datasets
from lib.config import cfg, update_config


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default='../experiments/vgg19_368x368_sgd.yaml',
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('-o', '--output', default=None,
                        help='output file')
    parser.add_argument('--disable-cuda', action='store_true',
                        help='disable CUDA')
    args = parser.parse_args()

    # add args.device
    args.device = torch.device('cpu')
    args.pin_memory = False
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        args.pin_memory = True

    return args

def load_pretrained_model(trainer, rtpose_model):
    last_epoch = -1
    last_ckpt_name = None
    checkpoints = os.listdir(trainer.checkpoint_callback.filepath)
    for name in checkpoints:
        # ignore hpc ckpts
        if 'hpc_' in name:
            continue

        if '.ckpt' in name:
            epoch = name.split('epoch_')[1]
            epoch = int(re.sub('[^0-9]', '', epoch))

            if epoch > last_epoch:
                last_epoch = epoch
                last_ckpt_name = name

    # restore last checkpoint

    last_ckpt_path = os.path.join(trainer.checkpoint_callback.filepath, last_ckpt_name)
    state_dict = torch.load(last_ckpt_path, map_location=lambda storage, loc: storage)['state_dict']
    os.remove(last_ckpt_path)
    state_dict = {k.replace('model.',''):v for k,v in state_dict.items()}    
    rtpose_model.load_state_dict(state_dict)
 
    return rtpose_model
    
args = parse_args()
update_config(cfg, args)
print("Loading dataset...")
# load train data
preprocess = transforms.Compose([
    transforms.Normalize(),
    transforms.RandomApply(transforms.HFlip(), 0.5),
    transforms.RescaleRelative(scale_range=(cfg.DATASET.SCALE_MIN, cfg.DATASET.SCALE_MAX)),
    transforms.Crop(cfg.DATASET.IMAGE_SIZE),
    transforms.CenterPad(cfg.DATASET.IMAGE_SIZE),
])

# model
rtpose_vgg = get_model(trunk='vgg19')
# load pretrained
use_vgg(rtpose_vgg)
               
class rtpose_lightning(pl.LightningModule):

    def __init__(self, preprocess, target_transforms, model, optimizer):
        super(rtpose_lightning, self).__init__()

        self.preprocess = preprocess
        self.model = model
        self.opt = optimizer
        self.target_transforms = target_transforms

    def forward(self, x):
        _, saved_for_loss = self.model.forward(x)
        return saved_for_loss

    def l2_loss(self, saved_for_loss, heatmap_target, paf_target):
        loss_dict = OrderedDict()
        total_loss = 0
        for j in range(6):
            pred1 = saved_for_loss[2 * j]
            pred2 = saved_for_loss[2 * j + 1]

            # Compute losses
            loss1 = F.mse_loss(pred1, paf_target, reduction='mean')
            loss2 = F.mse_loss(pred2, heatmap_target, reduction='mean')

            total_loss += loss1
            total_loss += loss2

            # Get value from Variable and save for log
            loss_dict['paf'] = loss1.unsqueeze(0)
            loss_dict['heatmap'] = loss2.unsqueeze(0)

        loss_dict['loss'] = total_loss.unsqueeze(0)

        loss_dict['max_heatmap'] = torch.max(
            pred2.data[:, :-1, :, :]).unsqueeze(0)
        loss_dict['min_heatmap'] = torch.min(
            pred2.data[:, :-1, :, :]).unsqueeze(0)
        loss_dict['max_paf'] = torch.max(pred1.data).unsqueeze(0)
        loss_dict['min_paf'] = torch.min(pred1.data).unsqueeze(0)

        return loss_dict

    def training_step(self, batch, batch_nb):
        img, heatmap_target, paf_target = batch
        saved_for_loss = self.forward(img)
        loss_dict = self.l2_loss(saved_for_loss, heatmap_target, paf_target)
        output = {
            'loss': loss_dict['loss'],  # required
            'prog': loss_dict  # optional
        }
        return output

    def validation_step(self, batch, batch_nb):
        img, heatmap_target, paf_target = batch
        saved_for_loss = self.forward(img)
        loss_dict = self.l2_loss(saved_for_loss, heatmap_target, paf_target)
        loss_dict['val_loss'] = loss_dict['loss']
        return loss_dict

    def validation_end(self, outputs):
        output_dict = OrderedDict()
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        output_dict['avg_val_loss'] = avg_loss

        return output_dict

    def configure_optimizers(self):
        #scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5, \
        #            verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=3,\
        #            min_lr=0, eps=1e-08)
        scheduler = lr_scheduler.MultiStepLR(
            optimizer, milestones=cfg.TRAIN.LR_STEP, gamma=0.1)
        return [[self.opt], [scheduler]]

    @pl.data_loader
    def tng_dataloader(self):
        train_datas = [datasets.CocoKeypoints(
            root=cfg.DATASET.TRAIN_IMAGE_DIR,
            annFile=item,
            preprocess=preprocess,
            image_transform=transforms.image_transform_train,
            target_transforms=self.target_transforms,
            n_images=None,
        ) for item in cfg.DATASET.TRAIN_ANNOTATIONS]

        train_data = torch.utils.data.ConcatDataset(train_datas)

        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU*len(cfg.GPUS), shuffle=True,
            pin_memory=cfg.PIN_MEMORY, num_workers=cfg.WORKERS, drop_last=True)

        return train_loader

    @pl.data_loader
    def val_dataloader(self):
        val_data = datasets.CocoKeypoints(
            root=cfg.DATASET.VAL_IMAGE_DIR,
            annFile=cfg.DATASET.VAL_ANNOTATIONS,
            preprocess=preprocess,
            image_transform=transforms.image_transform_train,
            target_transforms=self.target_transforms,
            n_images=None,
        )
        val_loader = torch.utils.data.DataLoader(
            val_data, batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS), shuffle=False,
            pin_memory=cfg.PIN_MEMORY, num_workers=cfg.WORKERS, drop_last=True)

        return val_loader

    @pl.data_loader
    def test_dataloader(self):
        val_data = datasets.CocoKeypoints(
            root=cfg.DATASET.VAL_IMAGE_DIR,
            annFile=cfg.DATASET.VAL_ANNOTATIONS,
            preprocess=preprocess,
            image_transform=transforms.image_transform_train,
            target_transforms=self.target_transforms,
            n_images=None,
        )
        val_loader = torch.utils.data.DataLoader(
            val_data, batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS), shuffle=False,
            pin_memory=cfg.PIN_MEMORY, num_workers=cfg.WORKERS, drop_last=True)

        return val_loader

exp = Experiment(name=cfg.EXPERIMENT_NAME, save_dir=cfg.OUTPUT_DIR)

# callbacks
early_stop = EarlyStopping(
    monitor='avg_val_loss',
    patience=50,
    verbose=True,
    mode='min'
)

model_save_path = '{}/{}/{}'.format(cfg.OUTPUT_DIR, exp.name, exp.version)
checkpoint = ModelCheckpoint(
    filepath=model_save_path,
    save_best_only=True,
    verbose=True,
    monitor='avg_val_loss',
    mode='min'
)

if cfg.PRE_TRAIN.FREEZE_BASE_EPOCHS:
    # Fix the VGG weights first, and then the weights will be released
    for i in range(20):
        for param in rtpose_vgg.model0[i].parameters():
            param.requires_grad = False

    trainable_vars = [param for param in rtpose_vgg.parameters() if param.requires_grad]

    optimizer = torch.optim.SGD(trainable_vars, lr=cfg.PRE_TRAIN.LR,
                                momentum=cfg.PRE_TRAIN.MOMENTUM,
                                weight_decay=cfg.PRE_TRAIN.WD,
                                nesterov=cfg.PRE_TRAIN.NESTEROV)

    model = rtpose_lightning(preprocess, target_transforms=None, model=rtpose_vgg, optimizer = optimizer)



    trainer = Trainer(experiment=exp,
                      max_nb_epochs=cfg.PRE_TRAIN.FREEZE_BASE_EPOCHS,
                      gpus=cfg.GPUS,
                      checkpoint_callback=checkpoint,
                      early_stop_callback=early_stop)

    trainer.fit(model)

# Release all weights                                   
for param in rtpose_vgg.parameters():
    param.requires_grad = True

trainable_vars = [param for param in rtpose_vgg.parameters() if param.requires_grad]

optimizer = torch.optim.SGD(trainable_vars, lr=cfg.TRAIN.LR,
                            momentum=cfg.TRAIN.MOMENTUM,
                            weight_decay=cfg.TRAIN.WD,
                            nesterov=cfg.TRAIN.NESTEROV)

load_pretrained_model(trainer, rtpose_vgg)
model = rtpose_lightning(preprocess, target_transforms=None, model=rtpose_vgg, optimizer = optimizer)

trainer = Trainer(experiment=exp,
                  max_nb_epochs=cfg.TRAIN.EPOCHS,
                  gpus=cfg.GPUS,
                  checkpoint_callback=checkpoint,
                  early_stop_callback=early_stop)

trainer.fit(model)



