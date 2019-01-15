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
from utils.training import test

from swag.posteriors import SWAG
from swag.utils import adjust_learning_rate

parser = argparse.ArgumentParser(description='SGD/SWA training')

parser.add_argument('--data_path', type=str, default='/home/wesley/Documents/Code/SegNet-Tutorial/CamVid/', metavar='PATH',
                    help='path to datasets location (default: None)')

parser.add_argument('--batch_size', type=int, default=4, metavar='N', help='input batch size (default: 4)')

parser.add_argument('--resume', type=str, default=None, metavar='CKPT',
                    help='checkpoint to resume training from (default: None)')
parser.add_argument('--swa_resume', type=str, default=None, metavar='CKPT',
                    help='checkpoint to restor SWA from (default: None)')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--use_test', action='store_true', help='whether to use test or validation dataset (default: val)')
args = parser.parse_args()

CAMVID_PATH = Path(args.data_path)

torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

normalize = transforms.Normalize(mean=camvid.mean, std=camvid.std)

if args.use_test:
    print('Warning: using test dataset')
    test_dset = camvid.CamVid(
        CAMVID_PATH, 'test', joint_transform=None,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize
        ]))
else:
    print('Using validation dataset')
    test_dset = camvid.CamVid(
        CAMVID_PATH, 'val', joint_transform=None,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize
        ]))

test_loader = torch.utils.data.DataLoader(
    test_dset, batch_size=args.batch_size, shuffle=False)

# construct and load model
model = tiramisu.FCDenseNet67(n_classes=11).cuda()
checkpoint = torch.load(args.resume)
start_epoch = checkpoint['epoch']
print(start_epoch)
model.load_state_dict(checkpoint['state_dict'])

criterion = nn.NLLLoss(weight=camvid.class_weight[:-1].cuda(), reduction='none').cuda()

loss, err, mIOU = test(model, test_loader, criterion)
print(loss, 1-err, mIOU)