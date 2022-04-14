import argparse
import os
import sys
import time
from collections import OrderedDict

import numpy as np


sys.path.append(os.path.abspath("./"))

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from lib.network.openpose import OpenPose_Model, use_vgg
from lib.datasets import transforms, datasets
SOURCE_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

NOW_WHAT = 'coco'

if NOW_WHAT == 'coco':
    # For coco dataset training
    DATA_DIR = os.path.join(SOURCE_DIR, 'data/coco')
    ANNOTATIONS_TRAIN = [os.path.join(DATA_DIR, 'annotations', item) for item in ['person_keypoints_train2017.json']]
    ANNOTATIONS_VAL = os.path.join(DATA_DIR, 'annotations', 'person_keypoints_val2017.json')
    IMAGE_DIR_TRAIN = os.path.join(DATA_DIR, 'images/train2017')
    IMAGE_DIR_VAL = os.path.join(DATA_DIR, 'images/val2017')
    BATCH_SIZE = 72

elif NOW_WHAT == 'bean':
    # For soybean dataset training
    DATA_DIR = os.path.join(SOURCE_DIR, 'data/bean/images')
    ANNOTATIONS_TRAIN = None
    ANNOTATIONS_VAL = None
    IMAGE_DIR_TRAIN = DATA_DIR
    IMAGE_DIR_VAL = DATA_DIR
    BATCH_SIZE = 5


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
    group.add_argument('--batch-size', default=BATCH_SIZE, type=int,
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


def train_factory(args, preprocess, target_transforms):
    train_datas = [datasets.CocoKeypoints(
        root=args.train_image_dir,
        annFile=item,
        preprocess=preprocess,
        image_transform=transforms.image_transform_train,
        target_transforms=target_transforms,
        n_images=args.n_images,
    ) for item in args.train_annotations]

    train_data = torch.utils.data.ConcatDataset(train_datas)  # shape: (56599, 3, 3, 368, 368)
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        pin_memory=args.pin_memory, num_workers=args.loader_workers, drop_last=True
    )

    val_data = datasets.CocoKeypoints(
        root=args.val_image_dir,
        annFile=args.val_annotations,
        preprocess=preprocess,
        image_transform=transforms.image_transform_train,
        target_transforms=target_transforms,
        n_images=args.n_images,
    )  # shape: (2346, 3, 3, 368, 368)

    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False,
        pin_memory=args.pin_memory, num_workers=args.loader_workers, drop_last=True)

    return train_loader, val_loader, train_data, val_data


def bean_train_factory(args, preprocess, target_transforms):
    train_data = [datasets.SoybeanKeypoints(
        root=args.train_image_dir,
        preprocess=preprocess,
        image_transform=transforms.image_transform_train,
        target_transforms=target_transforms,
    )]

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        pin_memory=args.pin_memory, num_workers=args.loader_workers, drop_last=True
    )

    val_data = datasets.SoybeanKeypoints(
        root=args.val_image_dir,
        preprocess=preprocess,
        image_transform=transforms.image_transform_train,
        target_transforms=target_transforms
    )

    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False,
        pin_memory=args.pin_memory, num_workers=args.loader_workers, drop_last=True)

    return train_loader, val_loader, train_data, val_data


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


def build_names():
    names = []
    for j in range(1, 7):
        names.append('loss_stage%d' % j)       # 4 + 2 = 6 stages in total
    return names


def get_loss(saved_for_loss, heat_temp, vec_temp):
    # print("heat_temp", heat_temp.shape)    # torch.Size([batch_size, 19, 46, 46])
    # print("vec_temp", vec_temp.shape)      # torch.Size([batch_size, 38, 46, 46])
    names = build_names()
    saved_for_log = OrderedDict()
    criterion = nn.MSELoss(reduction='mean').cuda()
    total_loss = 0
    paf_preds, cm_preds = saved_for_loss[0], saved_for_loss[1]

    for i in range(4):
        pred = paf_preds[i]

        # Compute losses
        loss = criterion(pred, vec_temp)
        total_loss += loss

        # Get value from Variable and save for log
        saved_for_log[names[i]] = loss.item()

    for i in range(2):
        pred = cm_preds[i]

        # Compute losses
        loss = criterion(pred, heat_temp)
        total_loss += loss

        # Get value from Variable and save for log
        saved_for_log[names[i+4]] = loss.item()

    return total_loss, saved_for_log


def train(train_loader, model, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    meter_dict = {}
    for name in build_names():
        meter_dict[name] = AverageMeter()
    # meter_dict['max_ht'] = AverageMeter()
    # meter_dict['min_ht'] = AverageMeter()
    # meter_dict['max_paf'] = AverageMeter()
    # meter_dict['min_paf'] = AverageMeter()

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
        if i % args.print_freq == 0:
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
    # meter_dict['max_ht'] = AverageMeter()
    # meter_dict['min_ht'] = AverageMeter()
    # meter_dict['max_paf'] = AverageMeter()
    # meter_dict['min_paf'] = AverageMeter()
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
        if i % args.print_freq == 0:
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
        transforms.Crop(args.square_edge),  # 368
        transforms.CenterPad(args.square_edge),  # 368
    ])
    train_loader, val_loader, train_data, val_data = train_factory(args, preprocess, target_transforms=None)

    # model
    model = OpenPose_Model()
    model = torch.nn.DataParallel(model).cuda()
    # load pretrained
    use_vgg(model)

    # Fix the VGG weights first, and then the weights will be released
    for i in range(20):
        for param in model.module.feature_extractor[i].parameters():
            param.requires_grad = False

    trainable_vars = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.SGD(trainable_vars, lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=args.nesterov)

    for epoch in range(5):
        # train for one epoch

        train_loss = train(train_loader, model, optimizer, epoch)
        # evaluate on validation set
        val_loss = validate(val_loader, model, epoch)

        # Release all weights
    for param in model.module.parameters():
        param.requires_grad = True

    trainable_vars = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.SGD(trainable_vars, lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=args.nesterov)

    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5, verbose=True, threshold=0.0001,
                                     threshold_mode='rel', cooldown=3, min_lr=0, eps=1e-08)

    best_val_loss = np.inf

    model_save_filename = f'network/weight/best_coco_openpose.pth'
    for epoch in range(5, args.epochs):

        # train for one epoch
        train_loss = train(train_loader, model, optimizer, epoch)

        # evaluate on validation set
        val_loss = validate(val_loader, model, epoch)

        lr_scheduler.step(val_loss)

        is_best = val_loss < best_val_loss
        best_val_loss = min(val_loss, best_val_loss)
        if is_best:
            torch.save(model.state_dict(), model_save_filename)


if __name__ == '__main__':
    args = cli()
    train_coco()
