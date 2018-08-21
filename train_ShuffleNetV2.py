import argparse
import time
import os
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

#import encoding
from network.rtpose_vgg import get_model, use_vgg
from network import rtpose_shufflenetV2
from training.datasets.coco import get_loader

# Hyper-params
parser = argparse.ArgumentParser(description='PyTorch rtpose Training')
parser.add_argument('--data_dir', default='/data/coco/images', type=str, metavar='DIR',
                    help='path to where coco images stored') 
parser.add_argument('--mask_dir', default='/data/coco/', type=str, metavar='DIR',
                    help='path to where coco images stored')    
parser.add_argument('--logdir', default='/extra/tensorboy', type=str, metavar='DIR',
                    help='path to where tensorboard log restore')                                       
parser.add_argument('--json_path', default='/data/coco/COCO.json', type=str, metavar='PATH',
                    help='path to where coco images stored')                                      

parser.add_argument('--model_path', default='./network/weight/', type=str, metavar='DIR',
                    help='path to where the model saved') 
                    
parser.add_argument('--lr', '--learning-rate', default=1., type=float,
                    metavar='LR', help='initial learning rate')

parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
 
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
                    
parser.add_argument('--weight-decay', '--wd', default=0.000, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')  
parser.add_argument('--nesterov', dest='nesterov', action='store_true')     
                                                   
parser.add_argument('-o', '--optim', default='sgd', type=str)
#Device options
parser.add_argument('--gpu_ids', dest='gpu_ids', help='which gpu to use', nargs="+",
                    default=[0,1,2,3], type=int)
                    
parser.add_argument('-b', '--batch_size', default=80, type=int,
                    metavar='N', help='mini-batch size (default: 256)')

parser.add_argument('--print_freq', default=20, type=int, metavar='N',
                    help='number of iterations to print the training statistics')
from tensorboardX import SummaryWriter      
args = parser.parse_args()  
               
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in args.gpu_ids)

params_transform = dict()
params_transform['mode'] = 5
# === aug_scale ===
params_transform['scale_min'] = 0.5
params_transform['scale_max'] = 1.1
params_transform['scale_prob'] = 1
params_transform['target_dist'] = 0.6
# === aug_rotate ===
params_transform['max_rotate_degree'] = 40

# ===
params_transform['center_perterb_max'] = 40

# === aug_flip ===
params_transform['flip_prob'] = 0.5

params_transform['np'] = 56
params_transform['sigma'] = 7.0

def get_loss(saved_for_loss, heat_temp, heat_weight,
               vec_temp, vec_weight):

    saved_for_log = OrderedDict()
    criterion = nn.MSELoss(size_average=True).cuda()
    #criterion = encoding.nn.DataParallelCriterion(criterion, device_ids=args.gpu_ids)
    total_loss = 0


    pred1 = saved_for_loss[0] * vec_weight
    """
    print("pred1 sizes")
    print(saved_for_loss[2*j].data.size())
    print(vec_weight.data.size())
    print(vec_temp.data.size())
    """
    gt1 = vec_temp * vec_weight

    pred2 = saved_for_loss[1] * heat_weight
    gt2 = heat_weight * heat_temp
    """
    print("pred2 sizes")
    print(saved_for_loss[2*j+1].data.size())
    print(heat_weight.data.size())
    print(heat_temp.data.size())
    """

    # Compute losses
    loss1 = criterion(pred1, gt1)
    loss2 = criterion(pred2, gt2) 

    total_loss += loss1
    total_loss += loss2
    # print(total_loss)

    # Get value from Variable and save for log
    saved_for_log['paf'] = loss1.item()
    saved_for_log['heatmap'] = loss2.item()

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

    meter_dict['paf'] = AverageMeter()
    meter_dict['heatmap'] = AverageMeter()     
    meter_dict['max_ht'] = AverageMeter()
    meter_dict['min_ht'] = AverageMeter()    
    meter_dict['max_paf'] = AverageMeter()    
    meter_dict['min_paf'] = AverageMeter()
    
    
    # switch to train mode
    model.train()

    end = time.time()
    for i, (img, heatmap_target, heat_mask, paf_target, paf_mask) in enumerate(train_loader):
        # measure data loading time
        #writer.add_text('Text', 'text logged at step:' + str(i), i)
        
        #for name, param in model.named_parameters():
        #    writer.add_histogram(name, param.clone().cpu().data.numpy(),i)        
        data_time.update(time.time() - end)

        img = img.cuda()
        heatmap_target = heatmap_target.cuda()
        heat_mask = heat_mask.cuda()
        paf_target = paf_target.cuda()
        paf_mask = paf_mask.cuda()
        
        # compute output
        _,saved_for_loss = model(img)
        
        total_loss, saved_for_log = get_loss(saved_for_loss, heatmap_target, heat_mask,
               paf_target, paf_mask)
        
        for name,_ in meter_dict.items():
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
            print_string +='Data time {data_time.val:.3f} ({data_time.avg:.3f})\t'.format( data_time=data_time)
            print_string += 'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(loss=losses)

            for name, value in meter_dict.items():
                print_string+='{name}: {loss.val:.4f} ({loss.avg:.4f})\t'.format(name=name, loss=value)
            print(print_string)
    return losses.avg  
        
def validate(val_loader, model, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    
    meter_dict = {}
    meter_dict['paf'] = AverageMeter()
    meter_dict['heatmap'] = AverageMeter()    
    meter_dict['max_ht'] = AverageMeter()
    meter_dict['min_ht'] = AverageMeter()    
    meter_dict['max_paf'] = AverageMeter()    
    meter_dict['min_paf'] = AverageMeter()
    # switch to train mode
    model.train()

    end = time.time()
    for i, (img, heatmap_target, heat_mask, paf_target, paf_mask) in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        img = img.cuda()
        heatmap_target = heatmap_target.cuda()
        heat_mask = heat_mask.cuda()
        paf_target = paf_target.cuda()
        paf_mask = paf_mask.cuda()
        
        # compute output
        _,saved_for_loss = model(img)
        
        total_loss, saved_for_log = get_loss(saved_for_loss, heatmap_target, heat_mask,
               paf_target, paf_mask)
               
        #for name,_ in meter_dict.items():
        #    meter_dict[name].update(saved_for_log[name], img.size(0))
            
        losses.update(total_loss.item(), img.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()  
        if i % args.print_freq == 0:
            print_string = 'Epoch: [{0}][{1}/{2}]\t'.format(epoch, i, len(val_loader))
            print_string +='Data time {data_time.val:.3f} ({data_time.avg:.3f})\t'.format( data_time=data_time)
            print_string += 'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(loss=losses)

            for name, value in meter_dict.items():
                print_string+='{name}: {loss.val:.4f} ({loss.avg:.4f})\t'.format(name=name, loss=value)
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


print("Loading dataset...")
# load data
train_data = get_loader(args.json_path, args.data_dir,
                        args.mask_dir, 368, 8,
                        'rtpose', args.batch_size, params_transform = params_transform, 
                        shuffle=True, training=True, num_workers=16)
print('train dataset len: {}'.format(len(train_data.dataset)))

# validation data
valid_data = get_loader(args.json_path, args.data_dir, args.mask_dir, 368,
                            8, preprocess='rtpose', training=False,
                            batch_size=args.batch_size,  params_transform = params_transform, 
                            shuffle=False, num_workers=4)
print('val dataset len: {}'.format(len(valid_data.dataset)))

# model
model = rtpose_shufflenetV2.Network(width_multiplier=1.0)
#model = encoding.nn.DataParallelModel(model, device_ids=args.gpu_ids)
model = torch.nn.DataParallel(model).cuda()

 
writer = SummaryWriter(log_dir=args.logdir)       
                                                                                          

trainable_vars = [param for param in model.parameters() if param.requires_grad]
optimizer = torch.optim.SGD(trainable_vars, lr=args.lr,
                           momentum=args.momentum,
                           weight_decay=args.weight_decay,
                           nesterov=args.nesterov)          
                                                    
lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=3, min_lr=0, eps=1e-08)

best_val_loss = np.inf


model_save_filename = './network/weight/best_pose_ShuffleNetV2.pth'
for epoch in range(args.epochs):

    # train for one epoch
    train_loss = train(train_data, model, optimizer, epoch)

    # evaluate on validation set
    val_loss = validate(valid_data, model, epoch)   
    
    writer.add_scalars('data/scalar_group', {'train loss': train_loss,
                                             'val loss': val_loss}, epoch)
    lr_scheduler.step(val_loss)                        
    
    is_best = val_loss<best_val_loss
    best_val_loss = max(val_loss, best_val_loss)
    if is_best:
        torch.save(model.state_dict(), model_save_filename)      
        
writer.export_scalars_to_json(os.path.join(args.model_path,"tensorboard/all_scalars.json"))
writer.close()    
