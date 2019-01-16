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
from swag.utils import adjust_learning_rate, bn_update

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
parser.add_argument('--output', type=str, required=True, help='output file to save model predictions and targets')
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

criterion = nn.NLLLoss(weight=camvid.class_weight[:-1].cuda(), reduction='none').cuda()

# construct and load model
if args.swa_resume is not None:
    train_joint_transformer = transforms.Compose([
    joint_transforms.JointRandomHorizontalFlip()
    ])
    train_dset = camvid.CamVid(CAMVID_PATH, 'train',
        joint_transform=train_joint_transformer,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))
    train_loader = torch.utils.data.DataLoader(
        train_dset, batch_size=3, shuffle=True)

    checkpoint = torch.load(args.swa_resume)
    model = SWAG(tiramisu.FCDenseNet67, no_cov_mat=False, max_num_models=0, loading=True, n_classes=11)
    model.cuda()
    model.load_state_dict(checkpoint['state_dict'])

    model.sample(0.0)
    bn_update(train_loader, model)
else:
    model = tiramisu.FCDenseNet67(n_classes=11).cuda()
    checkpoint = torch.load(args.resume)
    start_epoch = checkpoint['epoch']
    print(start_epoch)
    model.load_state_dict(checkpoint['state_dict'])

print(len(test_loader))
loss, err, mIOU, model_output_targets = test(model, test_loader, criterion, return_outputs = True)
print(loss, 1-err, mIOU)

outputs = np.concatenate(model_output_targets['outputs'])
targets = np.concatenate(model_output_targets['targets'])
np.savez(args.output, preds=outputs, targets=targets)
