import time
from collections import OrderedDict
import torch
import torch.nn as nn
from training.datasets.coco import get_loader
from network.rtpose_vgg import get_model, use_vgg

# Hyper-params
data_dir = '/data/coco/images'
mask_dir = '/data/coco/'
json_path = '/data/coco/COCO.json'
opt = 'sgd'
momentum = 0.9
weight_decay = 0.000
nesterov = True
inp_size = 368
feat_stride = 8

model_path = './models/'

# Set Training parameters
exp_name = 'original_rtpose'
save_dir = '/extra/tensorboy/models/{}'.format(exp_name)

max_epoch = 30
lr_decay_epoch = {30, 60, 90, 120, 150, 180}
init_lr = 0.1
lr_decay = 0.8

gpus = [0,1, 2, 3]
batch_size = 20 * len(gpus)
print_freq = 20

def build_names():
    names = []

    for j in range(1, 7):
        for k in range(1, 3):
            names.append('loss_stage%d_L%d' % (j, k))
    return names


def get_loss(saved_for_loss, heat_temp, heat_weight,
               vec_temp, vec_weight):

    names = build_names()
    saved_for_log = OrderedDict()
    criterion = nn.MSELoss(size_average=True).cuda()
    total_loss = 0

    for j in range(6):
        pred1 = saved_for_loss[2 * j] * vec_weight
        """
        print("pred1 sizes")
        print(saved_for_loss[2*j].data.size())
        print(vec_weight.data.size())
        print(vec_temp.data.size())
        """
        gt1 = vec_temp * vec_weight

        pred2 = saved_for_loss[2 * j + 1] * heat_weight
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
        saved_for_log[names[2 * j]] = loss1.data[0]
        saved_for_log[names[2 * j + 1]] = loss2.data[0]

    saved_for_log['max_ht'] = torch.max(
        saved_for_loss[-1].data[:, 0:-1, :, :])
    saved_for_log['min_ht'] = torch.min(
        saved_for_loss[-1].data[:, 0:-1, :, :])
    saved_for_log['max_paf'] = torch.max(saved_for_loss[-2].data)
    saved_for_log['min_paf'] = torch.min(saved_for_loss[-2].data)

    return total_loss, saved_for_log
         

def train(train_loader, model, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (img, heatmap_target, heat_mask, paf_target, paf_mask) in enumerate(train_loader):
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
        
        losses.update(total_loss.data[0], img.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))

    return losses.avg  
        
def validate(val_loader, model, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

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

        losses.update(total_loss.data[0], img.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()  
    return losses.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


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


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

print("Loading dataset...")
# load data
train_data = get_loader(json_path, data_dir,
                        mask_dir, inp_size, feat_stride,
                        'vgg', batch_size,
                        shuffle=True, training=True)
print('train dataset len: {}'.format(len(train_data.dataset)))

# validation data
valid_data = get_loader(json_path, data_dir, mask_dir, inp_size,
                            feat_stride, preprocess='vgg', training=False,
                            batch_size=batch_size, shuffle=True)
print('val dataset len: {}'.format(len(valid_data.dataset)))

# model
model = get_model(trunk='vgg19')
model = torch.nn.DataParallel(model).cuda()
# load pretrained
use_vgg(model, model_path, 'vgg19')


# Fix the VGG weights first, and then the weights will be released
for param in model.module.model0[i].parameters():
    param.requires_grad = False

trainable_vars = [param for param in model.parameters() if param.requires_grad]
optimizer = torch.optim.SGD(trainable_vars, lr=init_lr,
                           momentum=momentum,
                           weight_decay=weight_decay,
                           nesterov=nesterov)
                                   
                                                               
for epoch in range(10):
    #adjust_learning_rate(optimizer, epoch)

    # train for one epoch
    train_loss = train(train_data, model, optimizer, epoch)

    # evaluate on validation set
    val_loss = validate(val_data, model, epoch)                               
                                   
for param in model.module.parameters():
    param.requires_grad = True

trainable_vars = [param for param in model.parameters() if param.requires_grad]
optimizer = torch.optim.SGD(trainable_vars, lr=init_lr,
                           momentum=momentum,
                           weight_decay=weight_decay,
                           nesterov=nesterov)                                   

for epoch in range(300):
    #adjust_learning_rate(optimizer, epoch)

    # train for one epoch
    train_loss = train(train_data, model, optimizer, epoch)

    # evaluate on validation set
    val_loss = validate(val_data, model, epoch)                                     
