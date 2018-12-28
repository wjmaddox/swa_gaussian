import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
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
from swag.utils import bn_update

parser = argparse.ArgumentParser(description='SGD/SWA training')
parser.add_argument('--swa', action='store_true', help='swa usage flag (default: off)')
parser.add_argument('--swa_start', type=float, default=161, metavar='N', help='SWA start epoch number (default: 161)')
parser.add_argument('--swa_lr', type=float, default=0.02, metavar='LR', help='SWA LR (default: 0.02)')
parser.add_argument('--swa_c_epochs', type=int, default=1, metavar='N',
                    help='SWA model collection frequency/cycle length in epochs (default: 1)')

parser.add_argument('--resume', type=str, default=None, metavar='CKPT',
                    help='checkpoint to resume training from (default: None)')
parser.add_argument('--swa_resume', type=str, default=None, metavar='CKPT',
                    help='checkpoint to restor SWA from (default: None)')
args = parser.parse_args()

CAMVID_PATH = Path('/home/wesley/Documents/Code/SegNet-Tutorial/CamVid/')
RESULTS_PATH = Path('.results/')
WEIGHTS_PATH = Path('.weights/')
RESULTS_PATH.mkdir(exist_ok=True)
WEIGHTS_PATH.mkdir(exist_ok=True)
batch_size = 2

normalize = transforms.Normalize(mean=camvid.mean, std=camvid.std)
train_joint_transformer = transforms.Compose([
    joint_transforms.JointRandomCrop(224), # commented for fine-tuning
    joint_transforms.JointRandomHorizontalFlip()
    ])
train_dset = camvid.CamVid(CAMVID_PATH, 'train',
      joint_transform=train_joint_transformer,
      transform=transforms.Compose([
          transforms.ToTensor(),
          normalize,
    ]))
train_loader = torch.utils.data.DataLoader(
    train_dset, batch_size=batch_size, shuffle=True)

val_dset = camvid.CamVid(
    CAMVID_PATH, 'val', joint_transform=None,
    transform=transforms.Compose([
        transforms.ToTensor(),
        normalize
    ]))
val_loader = torch.utils.data.DataLoader(
    val_dset, batch_size=batch_size, shuffle=False)

test_dset = camvid.CamVid(
    CAMVID_PATH, 'test', joint_transform=None,
    transform=transforms.Compose([
        transforms.ToTensor(),
        normalize
    ]))
test_loader = torch.utils.data.DataLoader(
    test_dset, batch_size=batch_size, shuffle=False)

LR = 1e-4
LR_DECAY = 0.995
DECAY_EVERY_N_EPOCHS = 1
N_EPOCHS = 900
torch.cuda.manual_seed(0)

model = tiramisu.FCDenseNet67(n_classes=12).cuda()
model.apply(train_utils.weights_init)
optimizer = torch.optim.RMSprop(model.parameters(), lr=LR, weight_decay=1e-4)
criterion = nn.NLLLoss2d(weight=camvid.class_weight.cuda()).cuda()
start_epoch = 1

if args.resume is not None:
    print('Resume training from %s' % args.resume)
    checkpoint = torch.load(args.resume)
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    del checkpoint

if args.swa:
    print('SWAG training')
    swag_model = SWAG(tiramisu.FCDenseNet67, no_cov_mat=True, n_classes = 12)
    swag_model.cuda()

for epoch in range(start_epoch, N_EPOCHS+1):
    since = time.time()

    ### Train ###
    trn_loss, trn_err = train_utils.train(
        model, train_loader, optimizer, criterion, epoch)
    print('Epoch {:d}\nTrain - Loss: {:.4f}, Acc: {:.4f}'.format(
        epoch, trn_loss, 1-trn_err))    
    time_elapsed = time.time() - since  
    print('Train Time {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    
    ### Test ###
    val_loss, val_err = train_utils.test(model, val_loader, criterion, epoch)
    print('Val - Loss: {:.4f} | Acc: {:.4f}'.format(val_loss, 1-val_err))
    
    time_elapsed = time.time() - since 
    print('Total Time {:.0f}m {:.0f}s\n'.format(
        time_elapsed // 60, time_elapsed % 60)) 

    if args.swa and (epoch + 1) > args.swa_start and (epoch + 1 - args.swa_start) % args.swa_c_epochs == 0:
        swag_model.collect_model(model)
        
        swag_model.sample(0.0)
        bn_update(train_loader, swag_model)
        val_loss, val_err = train_utils.test(swag_model, val_loader, criterion, epoch)
        print('SWAG Val - Loss: {:.4f} | Acc: {:.4f}'.format(val_loss, 1-val_err))
    
    ### Checkpoint ###
    if epoch % 25 is 0:
        print('Saving model at Epoch: ', epoch)
        train_utils.save_checkpoint(dir='swag_weights', 
                            epoch=epoch, 
                            state_dict=model.state_dict(), 
                            optimizer=optimizer.state_dict()
                        )
        if args.swa and (epoch + 1) > args.swa_start:
            train_utils.save_checkpoint(
                dir='swag_weights',
                epoch=epoch,
                name='swag',
                state_dict=swag_model.state_dict(),
            )
        #train_utils.save_weights(model, epoch, val_loss, val_err)

    if args.swa and (epoch + 1) > args.swa_start:
        train_utils.adjust_learning_rate(args.swa_lr, LR_DECAY, optimizer, epoch, DECAY_EVERY_N_EPOCHS)
    else:
        ### Adjust Lr ###
        train_utils.adjust_learning_rate(LR, LR_DECAY, optimizer, 
                                        epoch, DECAY_EVERY_N_EPOCHS)

test_loss, test_err = train_utils.test(model, test_loader, criterion, epoch=1)  
print('Test - Loss: {:.4f} | Acc: {:.4f}'.format(test_loss, 1-test_err))