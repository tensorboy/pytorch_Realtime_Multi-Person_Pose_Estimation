import argparse
import os
import sys
import time
from collections import OrderedDict

import numpy as np
from matplotlib import pyplot as plt
sys.path.append(os.path.abspath("./"))

from lib.config import update_config, cfg


import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from lib.network.rtpose_vgg import get_model, use_vgg
from lib.datasets import transforms, datasets

SOURCE_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

NOW_WHAT = 'bean'

if NOW_WHAT == 'coco':
    # For coco dataset training
    DATA_DIR = os.path.join(SOURCE_DIR, 'data/coco')
    ANNOTATIONS_TRAIN = [os.path.join(DATA_DIR, 'annotations', item) for item in ['person_keypoints_train2017.json']]
    ANNOTATIONS_VAL = os.path.join(DATA_DIR, 'annotations', 'person_keypoints_val2017.json')
    IMAGE_DIR_TRAIN = os.path.join(DATA_DIR, 'images/train2017')
    IMAGE_DIR_VAL = os.path.join(DATA_DIR, 'images/val2017')
    BATCH_SIZE = 5


def train_factory(args, preprocess, target_transforms):
    train_datas = [datasets.CocoKeypoints(
        root=os.path.join(SOURCE_DIR,cfg.DATASET.TRAIN_IMAGE_DIR),
        annFile=item,
        preprocess=preprocess,
        image_transform=transforms.image_transform_train,
        target_transforms=target_transforms,
        n_images=args.n_images,
    ) for item in cfg.DATASET.TRAIN_ANNOTATIONS]

    train_data = torch.utils.data.ConcatDataset(train_datas)  # shape: (56599, 3, 3, 368, 368)
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        pin_memory=args.pin_memory, num_workers=args.loader_workers, drop_last=True
    )

    val_data = datasets.CocoKeypoints(
        root=os.path.join(SOURCE_DIR, cfg.DATASET.VAL_IMAGE_DIR),
        annFile=cfg.DATASET.VAL_ANNOTATIONS,
        preprocess=preprocess,
        image_transform=transforms.image_transform_train,
        target_transforms=target_transforms,
        n_images=args.n_images,
    )  # shape: (2346, 3, 3, 368, 368)

    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU, shuffle=False,
        pin_memory=args.pin_memory, num_workers=cfg.WORKERS, drop_last=True)

    return train_loader, val_loader, train_data, val_data


def bean_train_factory(cfg, preprocess, target_transforms):
    print(SOURCE_DIR)
    print(cfg.DATASET.TRAIN_IMAGE_DIR)
    print(os.path.join(SOURCE_DIR, cfg.DATASET.TRAIN_IMAGE_DIR))

    train_data = datasets.SoybeanKeypoints(
        root=os.path.join(SOURCE_DIR, cfg.DATASET.TRAIN_IMAGE_DIR),
        preprocess=preprocess,
        image_transform=transforms.image_transform_train,
        target_transforms=target_transforms,
        stride=8
    )
    print(len(train_data))
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU, shuffle=True,
        pin_memory=args.pin_memory, num_workers=cfg.WORKERS, drop_last=True
    )
    print('train length:', len(train_loader.dataset))

    val_data = datasets.SoybeanKeypoints(
        root=os.path.join(SOURCE_DIR, cfg.DATASET.VAL_IMAGE_DIR),
        preprocess=preprocess,
        image_transform=transforms.image_transform_train,
        target_transforms=target_transforms,
        stride=8
    )
    print(len(val_data))

    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU, shuffle=False,
        pin_memory=args.pin_memory, num_workers=cfg.WORKERS, drop_last=True)
    print('val length:', len(val_loader.dataset))

    return train_loader, val_loader, train_data, val_data


def cli(now_what):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    if now_what == 'coco':
        parser.add_argument('--cfg', help='experiment configure file name',
                            default='./experiments/vgg19_368x368_sgd.yaml', type=str)
    elif now_what == 'bean':
        parser.add_argument('--cfg', help='experiment configure file name',
                            default='./experiments/vgg19_368x368_sgd_bean.yaml', type=str)

    args = parser.parse_args()
    # update config file
    update_config(cfg, args)
    # add args.device
    args.device = torch.device('cpu')
    args.pin_memory = False
    if cfg.CUDNN.ENABLED and torch.cuda.is_available():
        args.device = torch.device('cuda')
        args.pin_memory = True

    return args


def build_names():
    names = []

    for j in range(1, 7):
        for k in range(1, 3):
            names.append('loss_stage%d_L%d' % (j, k))
    return names


def get_loss(saved_for_loss, heat_temp, vec_temp):
    names = build_names()
    saved_for_log = OrderedDict()
    criterion = nn.MSELoss(reduction='mean').cuda()
    total_loss = 0
    # print(vec_temp.shape)       # torch.Size([batch_size, 38, 46, 46])
    for j in range(6):
        pred1 = saved_for_loss[2 * j]
        pred2 = saved_for_loss[2 * j + 1]

        # Compute losses
        loss1 = criterion(pred1, vec_temp)
        loss2 = criterion(pred2, heat_temp)

        fig = plt.figure()
        ax1 = fig.add_subplot(121)  # left side
        ax2 = fig.add_subplot(122)  # right side
        print('loss %d:' % (j+1))
        ax1.imshow(pred2[0][0].cpu().detach().numpy())
        ax2.imshow(heat_temp[0][0].cpu().detach().numpy())
        plt.show()

        total_loss += loss1
        total_loss += loss2

        # Get value from Variable and save for log
        saved_for_log[names[2 * j]] = loss1.item()
        saved_for_log[names[2 * j + 1]] = loss2.item()

    saved_for_log['max_ht'] = torch.max(
        saved_for_loss[-1].data[:, 0:-1, :, :]).item()
    saved_for_log['min_ht'] = torch.min(
        saved_for_loss[-1].data[:, 0:-1, :, :]).item()
    saved_for_log['max_paf'] = torch.max(saved_for_loss[-2].data).item()
    saved_for_log['min_paf'] = torch.min(saved_for_loss[-2].data).item()

    return total_loss, saved_for_log


def train(train_loader, model, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    meter_dict = {}
    for name in build_names():
        meter_dict[name] = AverageMeter()
    meter_dict['max_ht'] = AverageMeter()
    meter_dict['min_ht'] = AverageMeter()
    meter_dict['max_paf'] = AverageMeter()
    meter_dict['min_paf'] = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (img, heatmap_target, paf_target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        img = img.cuda()
        heatmap_target = heatmap_target.cuda()
        paf_target = paf_target.cuda()
        # compute output
        _, saved_for_loss = model(img)
        total_loss, saved_for_log = get_loss(saved_for_loss, heatmap_target, paf_target)

        for name, _ in meter_dict.items():
            meter_dict[name].update(saved_for_log[name], img.size(0))
        losses.update(total_loss, img.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % cfg.TRAIN.PRINT_FREQ == 0:
            print_string = 'Epoch: [{0}][{1}/{2}]\t'.format(epoch, i, len(train_loader))
            print_string += 'Data time {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(data_time=data_time)
            print_string += 'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(loss=losses)

            for name, value in meter_dict.items():
                print_string += '{name}: {loss.val:.4f} ({loss.avg:.4f})\t'.format(name=name, loss=value)
            print(print_string)
    return losses.avg


def validate(val_loader, model, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    meter_dict = {}
    for name in build_names():
        meter_dict[name] = AverageMeter()
    meter_dict['max_ht'] = AverageMeter()
    meter_dict['min_ht'] = AverageMeter()
    meter_dict['max_paf'] = AverageMeter()
    meter_dict['min_paf'] = AverageMeter()
    # switch to train mode
    model.eval()

    end = time.time()
    for i, (img, heatmap_target, paf_target) in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        img = img.cuda()
        heatmap_target = heatmap_target.cuda()
        paf_target = paf_target.cuda()

        # compute output
        _, saved_for_loss = model(img)

        total_loss, saved_for_log = get_loss(saved_for_loss, heatmap_target, paf_target)

        # for name,_ in meter_dict.items():
        #    meter_dict[name].update(saved_for_log[name], img.size(0))

        losses.update(total_loss.item(), img.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % cfg.TRAIN.PRINT_FREQ == 0:
            print_string = 'Epoch: [{0}][{1}/{2}]\t'.format(epoch, i, len(val_loader))
            print_string += 'Data time {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(data_time=data_time)
            print_string += 'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(loss=losses)

            for name, value in meter_dict.items():
                print_string += '{name}: {loss.val:.4f} ({loss.avg:.4f})\t'.format(name=name, loss=value)
            print(print_string)

    return losses.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_coco():
    print("Loading dataset...")
    # load train data
    preprocess = transforms.Compose([
        transforms.Normalize(),
        transforms.RandomApply(transforms.HFlip(), 0.5),  # Is it necessary ?
        transforms.RescaleRelative(),
        transforms.Crop(cfg.DATASET.IMAGE_SIZE),  # 368
        transforms.CenterPad(cfg.DATASET.IMAGE_SIZE),  # 368
    ])
    train_loader, val_loader, train_data, val_data = train_factory(cfg, preprocess, target_transforms=None)

    # model
    model = get_model(dataset=NOW_WHAT, trunk='vgg19')
    model = torch.nn.DataParallel(model).cuda()
    # load pretrained
    use_vgg(model)

    # Fix the VGG weights first, and then the weights will be released
    for i in range(20):
        for param in model.module.model0[i].parameters():
            param.requires_grad = False

    trainable_vars = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.SGD(trainable_vars, lr=cfg.TRAIN.LR,
                                momentum=cfg.TRAIN.MOMENTUM,
                                weight_decay=cfg.TRAIN.WD,
                                nesterov=cfg.TRAIN.NESTEROV)

    for epoch in range(5):
        # train for one epoch

        train_loss = train(train_loader, model, optimizer, epoch)
        # evaluate on validation set
        val_loss = validate(val_loader, model, epoch)

        # Release all weights
    for param in model.module.parameters():
        param.requires_grad = True

    trainable_vars = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.SGD(trainable_vars, lr=cfg.TRAIN.LR,
                                momentum=cfg.TRAIN.MOMENTUM,
                                weight_decay=cfg.TRAIN.WD,
                                nesterov=cfg.TRAIN.NESTEROV)

    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5, verbose=True, threshold=0.0001,
                                     threshold_mode='rel', cooldown=3, min_lr=0, eps=1e-08)

    best_val_loss = np.inf

    model_save_filename = f'network/weight/best_pose.pth'
    for epoch in range(5, cfg.TRAIN.EPOCHS):

        # train for one epoch
        train_loss = train(train_loader, model, optimizer, epoch)

        # evaluate on validation set
        val_loss = validate(val_loader, model, epoch)

        lr_scheduler.step(val_loss)

        is_best = val_loss < best_val_loss
        best_val_loss = min(val_loss, best_val_loss)
        if is_best:
            torch.save(model.state_dict(), model_save_filename)


def train_soybean():
    print("Loading dataset...")
    preprocess = transforms.Compose([
        transforms.NormalizeBean(),
        transforms.RescaleRelativeBean(),
        transforms.CropBean(cfg.DATASET.IMAGE_SIZE),  # 368
        transforms.CenterPadBean(cfg.DATASET.IMAGE_SIZE),  # 368
    ])

    train_loader, val_loader, train_data, val_data = bean_train_factory(cfg, preprocess=preprocess,
                                                                        target_transforms=None)

    # model
    model = get_model(dataset=NOW_WHAT, trunk='vgg19')
    model = torch.nn.DataParallel(model).cuda()
    # load pretrained
    use_vgg(model)

    # Fix the VGG weights first, and then the weights will be released
    for i in range(20):
        for param in model.module.model0[i].parameters():
            param.requires_grad = False

    trainable_vars = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.SGD(trainable_vars, lr=cfg.TRAIN.LR,
                                momentum=cfg.TRAIN.MOMENTUM,
                                weight_decay=cfg.TRAIN.WD,
                                nesterov=cfg.TRAIN.NESTEROV)

    for epoch in range(5):
        # train for one epoch

        train_loss = train(train_loader, model, optimizer, epoch)
        # evaluate on validation set
        val_loss = validate(val_loader, model, epoch)

        # Release all weights
    for param in model.module.parameters():
        param.requires_grad = True

    trainable_vars = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.SGD(trainable_vars, lr=cfg.TRAIN.LR,
                                momentum=cfg.TRAIN.MOMENTUM,
                                weight_decay=cfg.TRAIN.WD,
                                nesterov=cfg.TRAIN.NESTEROV)

    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5, verbose=True, threshold=0.0001,
                                     threshold_mode='rel', cooldown=3, min_lr=0, eps=1e-08)

    best_val_loss = np.inf

    model_save_filename = 'network/weight/best_bean.pth'
    for epoch in range(5, cfg.TRAIN.EPOCHS):

        # train for one epoch
        train_loss = train(train_loader, model, optimizer, epoch)

        # evaluate on validation set
        val_loss = validate(val_loader, model, epoch)

        lr_scheduler.step(val_loss)

        is_best = val_loss < best_val_loss
        best_val_loss = min(val_loss, best_val_loss)
        if is_best:
            torch.save(model.state_dict(), model_save_filename)
            print('better model has been saved: ', model_save_filename)
            print()


if __name__ == '__main__':
    args = cli(NOW_WHAT)
    if NOW_WHAT == 'coco':
        train_coco()
    elif NOW_WHAT == 'bean':
        train_soybean()
