import time
from pathlib import Path
import numpy as np
#import matplotlib.pyplot as plt
import os, sys
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from models import tiramisu
from datasets import camvid
from datasets import joint_transforms
import utils.imgs
import utils.training as train_utils

from swag.posteriors import SWAG
from swag.utils import bn_update, adjust_learning_rate

parser = argparse.ArgumentParser(description='SGD/SWA training')

parser.add_argument('--data_path', type=str, default='/home/wesley/Documents/Code/SegNet-Tutorial/CamVid/', metavar='PATH',
                    help='path to datasets location (default: None)')
parser.add_argument('--dir', type=str, default=None, required=True, help='training directory (default: None)')

parser.add_argument('--epochs', type=int, default=670, metavar='N', help='number of epochs to train (default: 200)')
parser.add_argument('--save_freq', type=int, default=10, metavar='N', help='save frequency (default: 10)')
parser.add_argument('--eval_freq', type=int, default=5, metavar='N', help='evaluation frequency (default: 5)')

parser.add_argument('--batch_size', type=int, default=2, metavar='N', help='input batch size (default: 2)')
parser.add_argument('--lr_init', type=float, default=1e-4, metavar='LR', help='initial learning rate (default: 0.01)')
parser.add_argument('--wd', type=float, default=1e-4, help='weight decay (default: 1e-4)')
parser.add_argument('--optimizer', type=str, choices=['RMSProp', 'SGD'], default='RMSProp')

parser.add_argument('--ft_start', type=int, default=670, help='begin fine-tuning with full sized images (default: 670)')

parser.add_argument('--swa', action='store_true', help='swa usage flag (default: off)')
parser.add_argument('--swa_start', type=float, default=600, metavar='N', help='SWA start epoch number (default: 161)')
parser.add_argument('--swa_lr', type=float, default=0.02, metavar='LR', help='SWA LR (default: 0.02)')
parser.add_argument('--swa_c_epochs', type=int, default=1, metavar='N',
                    help='SWA model collection frequency/cycle length in epochs (default: 1)')

parser.add_argument('--resume', type=str, default=None, metavar='CKPT',
                    help='checkpoint to resume training from (default: None)')
parser.add_argument('--swa_resume', type=str, default=None, metavar='CKPT',
                    help='checkpoint to restor SWA from (default: None)')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

args = parser.parse_args()

CAMVID_PATH = Path(args.data_path)

torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

print('Preparing directory %s' % args.dir)
os.makedirs(args.dir, exist_ok=True)
with open(os.path.join(args.dir, 'command.sh'), 'w') as f:
    f.write(' '.join(sys.argv))
    f.write('\n')

normalize = transforms.Normalize(mean=camvid.mean, std=camvid.std)

# beginning train joint transformer includes random crops
train_joint_transformer = transforms.Compose([
    joint_transforms.JointRandomCrop(224), # commented for fine-tuning
    joint_transforms.JointRandomHorizontalFlip()
    ])

hc_train_dset = camvid.CamVid(CAMVID_PATH, 'train',
      joint_transform=train_joint_transformer,
      transform=transforms.Compose([
          transforms.ToTensor(),
          normalize,
    ]))
hc_train_loader = torch.utils.data.DataLoader(
    hc_train_dset, batch_size=args.batch_size, shuffle=True)

print('Beginning with cropped images')
train_loader = hc_train_loader

# fine-tuning does not include random crops
ft_train_joint_transformer = transforms.Compose([
    joint_transforms.JointRandomHorizontalFlip()
    ])
ft_train_dset = camvid.CamVid(CAMVID_PATH, 'train',
      joint_transform=ft_train_joint_transformer,
      transform=transforms.Compose([
          transforms.ToTensor(),
          normalize,
    ]))

ft_train_loader = torch.utils.data.DataLoader(
    ft_train_dset, batch_size=args.batch_size, shuffle=True)



val_dset = camvid.CamVid(
    CAMVID_PATH, 'val', joint_transform=None,
    transform=transforms.Compose([
        transforms.ToTensor(),
        normalize
    ]))
val_loader = torch.utils.data.DataLoader(
    val_dset, batch_size=args.batch_size, shuffle=False)

test_dset = camvid.CamVid(
    CAMVID_PATH, 'test', joint_transform=None,
    transform=transforms.Compose([
        transforms.ToTensor(),
        normalize
    ]))
test_loader = torch.utils.data.DataLoader(
    test_dset, batch_size=args.batch_size, shuffle=False)

LR = args.lr_init
LR_DECAY = 0.995
DECAY_EVERY_N_EPOCHS = 1

model = tiramisu.FCDenseNet67(n_classes=11).cuda()
model.apply(train_utils.weights_init)
if args.optimizer == 'RMSProp':
    optimizer = torch.optim.RMSprop(model.parameters(), lr=LR, weight_decay=args.wd)
else:
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_init, weight_decay = args.wd, momentum = 0.9)

criterion = nn.NLLLoss2d(weight=camvid.class_weight.cuda()).cuda()
start_epoch = 1

def schedule(epoch):
    t = (epoch) / (args.swa_start if args.swa else args.epochs)
    lr_ratio = args.swa_lr / args.lr_init if args.swa else 0.01
    if t <= 0.5:
        factor = 1.0
    elif t <= 0.9:
        factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
    else:
        factor = lr_ratio
    return args.lr_init * factor

if args.resume is not None:
    print('Resume training from %s' % args.resume)
    checkpoint = torch.load(args.resume)
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    del checkpoint

if args.swa:
    print('SWAG training')
    swag_model = SWAG(tiramisu.FCDenseNet67, no_cov_mat=False, n_classes = 11)
    swag_model.cuda()

if args.swa and args.swa_resume is not None:
    checkpoint = torch.load(args.swa_resume)
    swag_model = SWAG(tiramisu.FCDenseNet67, no_cov_mat=False, max_num_models=20, loading=True, num_classes=11)
    swag_model.cuda()
    swag_model.load_state_dict(checkpoint['state_dict'])

for epoch in range(start_epoch, args.epochs+1):
    since = time.time()

    ### Train ###
    if epoch == args.ft_start:
        print('Now replacing data loader with fine-tuned data loader.')
        train_loader = ft_train_loader

    trn_loss, trn_err = train_utils.train(
        model, train_loader, optimizer, criterion, epoch)
    print('Epoch {:d}\nTrain - Loss: {:.4f}, Acc: {:.4f}'.format(
        epoch, trn_loss, 1-trn_err))    
    time_elapsed = time.time() - since  
    print('Train Time {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    
    if epoch % args.eval_freq is 0:
        ### Test ###
        val_loss, val_err, val_iou = train_utils.test(model, val_loader, criterion, epoch)
        print('Val - Loss: {:.4f} | Acc: {:.4f} | IOU: {:.4f}'.format(val_loss, 1-val_err, val_iou))
    
    time_elapsed = time.time() - since 
    print('Total Time {:.0f}m {:.0f}s\n'.format(
        time_elapsed // 60, time_elapsed % 60)) 

    if args.swa and (epoch + 1) > args.swa_start and (epoch + 1 - args.swa_start) % args.swa_c_epochs == 0:
        print('Saving SWA model at epoch: ', epoch)
        swag_model.collect_model(model)
        
        if epoch % args.eval_freq is 0:
            swag_model.sample(0.0)
            bn_update(train_loader, swag_model)
            val_loss, val_err, val_iou = train_utils.test(swag_model, val_loader, criterion, epoch)
            print('SWA Val - Loss: {:.4f} | Acc: {:.4f} | IOU: {:.4f}'.format(val_loss, 1-val_err, val_iou))
    
    ### Checkpoint ###
    if epoch % args.save_freq is 0:
        print('Saving model at Epoch: ', epoch)
        train_utils.save_checkpoint(dir=args.dir, 
                            epoch=epoch, 
                            state_dict=model.state_dict(), 
                            optimizer=optimizer.state_dict()
                        )
        if args.swa and (epoch + 1) > args.swa_start:
            train_utils.save_checkpoint(
                dir=args.dir,
                epoch=epoch,
                name='swag',
                state_dict=swag_model.state_dict(),
            )
        #train_utils.save_weights(model, epoch, val_loss, val_err)

    if args.optimizer=='RMSProp':
        ### Adjust Lr ###
        if epoch < args.ft_start:
            train_utils.adjust_learning_rate(LR, LR_DECAY, optimizer, 
                                            epoch, DECAY_EVERY_N_EPOCHS)
        else:
            # TODO: make this an option?
            adjust_learning_rate(optimizer, 1e-4)
            
        if args.swa and (epoch + 1) > args.swa_start:
            LR_DECAY = 1.
        
    elif args.optimizer=='SGD':
        lr = schedule(epoch)
        adjust_learning_rate(optimizer, lr)
    
### Test set ###
if args.swa:
    swag_model.sample(0.0)
    bn_update(train_loader, swag_model)
    test_loss, test_err, test_iou = train_utils.test(swag_model, test_loader, criterion, epoch=1)  
    print('SWA Test - Loss: {:.4f} | Acc: {:.4f} | IOU: {:.4f}'.format(test_loss, 1-test_err, test_iou))

test_loss, test_err, test_iou = train_utils.test(model, test_loader, criterion, epoch=1)  
print('SGD Test - Loss: {:.4f} | Acc: {:.4f} | IOU: {:.4f}'.format(test_loss, 1-test_err, test_iou))
