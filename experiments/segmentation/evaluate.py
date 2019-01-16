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


def numpy_metrics(y_pred, y_true, n_classes = 11, void_labels=11):
    """
    Similar to theano_metrics to metrics but instead y_pred and y_true are now numpy arrays
    from: https://github.com/SimJeg/FC-DenseNet/blob/master/metrics.py
    void label is 11 by default
    """

    # Put y_pred and y_true under the same shape
    y_pred = np.argmax(y_pred, axis=1)
    #y_true = y_true.flatten()

    # We use not_void in case the prediction falls in the void class of the groundtruth
    not_void = ~ np.any([y_true == label for label in void_labels], axis=0)

    I = np.zeros(n_classes)
    U = np.zeros(n_classes)

    for i in range(n_classes):
        y_true_i = y_true == i
        y_pred_i = y_pred == i

        I[i] = np.sum(y_true_i & y_pred_i)
        U[i] = np.sum((y_true_i | y_pred_i) & not_void)

    accuracy = np.sum(I) / np.sum(not_void)
    return I, U, accuracy

def test(model, test_loader, criterion, num_classes = 11):
    model.eval()
    with torch.no_grad():
        test_loss = 0
        test_error = 0
        I_tot = np.zeros(num_classes)
        U_tot = np.zeros(num_classes)
        
        for data, target in test_loader:
            data = data.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            output = model(data)

            I, U, acc = numpy_metrics(output.cpu().numpy(), target.cpu().numpy(), n_classes=11, void_labels=[11])
            print(1-batch_error, acc) #should be the same
            I_tot += I
            U_tot += U
            test_error += (1 - acc)

        test_loss /= len(test_loader)
        test_error /= len(test_loader)
        m_jacc = np.mean(I_tot / U_tot)
        return test_loss, test_error, m_jacc

# construct and load model
model = tiramisu.FCDenseNet67(n_classes=11).cuda()
checkpoint = torch.load(args.resume)
start_epoch = checkpoint['epoch']
print(start_epoch)
model.load_state_dict(checkpoint['state_dict'])

criterion = nn.NLLLoss(weight=camvid.class_weight[:-1].cuda()).cuda()

loss, err, mIOU = test(model, test_loader, criterion)
print(loss, 1-err, mIOU)