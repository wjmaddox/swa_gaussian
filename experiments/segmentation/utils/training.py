import os
import sys
import math
import string
import random
import shutil
import numpy as np

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import torch.nn.functional as F

from . import imgs as img_utils

RESULTS_PATH = '.results/'
WEIGHTS_PATH = '.weights/'

def save_checkpoint(dir, epoch, name='checkpoint', **kwargs):
    state = {
        'epoch': epoch,
    }
    state.update(kwargs)
    filepath = os.path.join(dir, '%s-%d.pt' % (name, epoch))
    torch.save(state, filepath)

def save_weights(model, epoch, loss, err):
    weights_fname = 'weights-%d-%.3f-%.3f.pth' % (epoch, loss, err)
    weights_fpath = os.path.join(WEIGHTS_PATH, weights_fname)
    torch.save({
            'startEpoch': epoch,
            'loss':loss,
            'error': err,
            'state_dict': model.state_dict()
        }, weights_fpath)
    shutil.copyfile(weights_fpath, WEIGHTS_PATH+'latest.th')

def load_weights(model, fpath):
    print("loading weights '{}'".format(fpath))
    weights = torch.load(fpath)
    startEpoch = weights['startEpoch']
    model.load_state_dict(weights['state_dict'])
    print("loaded weights (lastEpoch {}, loss {}, error {})"
          .format(startEpoch-1, weights['loss'], weights['error']))
    return startEpoch

def get_predictions(output_batch):
    bs,c,h,w = output_batch.size()
    tensor = output_batch.data
    values, indices = tensor.cpu().max(1)
    indices = indices.view(bs,h,w)
    return indices

def error(preds, targets, weights=None, num_classes = 12):
    assert preds.size() == targets.size()
    bs,h,w = preds.size()
    n_pixels = bs*h*w

    # ignore background class
    incorrect = (preds.ne(targets) & targets.ne(torch.ones_like(targets) * (num_classes -1))).long().cpu().sum().item()
    #incorrect = preds.ne(targets).cpu().sum().item()
    err = incorrect/n_pixels

    return round(err,5)

def train(model, trn_loader, optimizer, criterion, epoch):
    model.train()
    trn_loss = 0
    trn_error = 0
    for idx, data in enumerate(trn_loader):
        inputs = Variable(data[0].cuda())
        targets = Variable(data[1].cuda())
        #print(inputs.size(), targets.size())

        optimizer.zero_grad()
        output = model(inputs)
        output_padded = torch.cat([output, torch.zeros_like(targets).float().unsqueeze(1)],dim=1)

        loss = criterion(output_padded, targets)
        loss.backward()
        optimizer.step()

        trn_loss += loss.item()
        pred = get_predictions(output)
        trn_error_curr = error(pred, targets.data.cpu())
        trn_error += trn_error_curr

    trn_loss /= len(trn_loader)
    trn_error /= len(trn_loader)
    return trn_loss, trn_error

def test(model, test_loader, criterion, epoch=1):
    model.eval()
    with torch.no_grad():
        test_loss = 0
        test_error = 0
        test_iou = []
        for data, target in test_loader:
            data = Variable(data.cuda(), volatile=True)
            target = Variable(target.cuda())
            output = model(data)
            output_padded = torch.cat([output, torch.zeros_like(target).float().unsqueeze(1)],dim=1)
            test_loss += criterion(output_padded, target).item()

            target = target.data.cpu()
            pred = get_predictions(output)
            test_error += error(pred, target)

            test_iou.append( iou(pred, target) )

        test_iou = np.array(test_iou).transpose()
        test_iou = np.nanmean(test_iou, axis=-1)
        print(test_iou)

        test_loss /= len(test_loader)
        test_error /= len(test_loader)
        #test_iou /= len(test_loader)
        return test_loss, test_error, test_iou.mean()

def adjust_learning_rate(lr, decay, optimizer, cur_epoch, n_epochs):
    """Sets the learning rate to the initially
        configured `lr` decayed by `decay` every `n_epochs`"""
    new_lr = lr * (decay ** (cur_epoch // n_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform(m.weight)
        m.bias.data.zero_()

def predict(model, input_loader, n_batches=1):
    input_loader.batch_size = 1
    predictions = []
    model.eval()
    for input, target in input_loader:
        data = Variable(input.cuda(), volatile=True)
        label = Variable(target.cuda())
        output = model(data)
        pred = get_predictions(output)
        predictions.append([input,target,pred])
    return predictions

def view_sample_predictions(model, loader, n):
    inputs, targets = next(iter(loader))
    data = Variable(inputs.cuda(), volatile=True)
    label = Variable(targets.cuda())
    output = model(data)
    pred = get_predictions(output)
    batch_size = inputs.size(0)
    for i in range(min(n, batch_size)):
        img_utils.view_image(inputs[i])
        img_utils.view_annotated(targets[i])
        img_utils.view_annotated(pred[i])


# https://github.com/Kaixhin/FCN-semantic-segmentation/blob/405f57c91894ed0dbbfc992d7f12b352cfbd6a8e/main.py#L78
def iou(pred, target, num_classes = 12):
    ious = []
    # Ignore IoU for background class
    for cls in range(num_classes - 1):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds[target_inds]).long().sum().cpu().item()  # Cast to long to prevent overflows
        union = pred_inds.long().sum().cpu().item() + target_inds.long().sum().cpu().item() - intersection
        if union == 0:
            ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
        else:
            ious.append(intersection / max(union, 1))
    return ious
