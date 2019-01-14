import torch
import os
import copy
from datetime import datetime
import math
import numpy as np

import torch.nn.functional as F

def flatten(lst):
    tmp = [i.contiguous().view(-1,1) for i in lst]
    return torch.cat(tmp).view(-1)

def LogSumExp(x,dim=0):
    m,_ = torch.max(x,dim=dim,keepdim=True)
    return m + torch.log((x - m).exp().sum(dim=dim,keepdim=True))

def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def save_checkpoint(dir, epoch, name='checkpoint', **kwargs):
    state = {
        'epoch': epoch,
    }
    state.update(kwargs)
    filepath = os.path.join(dir, '%s-%d.pt' % (name, epoch))
    torch.save(state, filepath)


def train_epoch(loader, model, criterion, optimizer, cuda=True, regression=False):
    loss_sum = 0.0
    correct = 0.0

    model.train()

    for i, (input, target) in enumerate(loader):
        if cuda:
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

        loss, output = criterion(model, input, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
            
        loss_sum += loss.data.item() * input.size(0)
        
        if not regression:
            pred = output.data.argmax(1, keepdim=True)
            correct += pred.eq(target.data.view_as(pred)).sum().item()
    
    return {
        'loss': loss_sum / len(loader.dataset),
        'accuracy': None if regression else correct / len(loader.dataset) * 100.0
    }


def eval(loader, model, criterion, cuda=True, regression=False):
    loss_sum = 0.0
    correct = 0.0

    model.eval()

    with torch.no_grad():
        for i, (input, target) in enumerate(loader):
            if cuda:
                input = input.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

            loss, output = criterion(model, input, target)

            loss_sum += loss.item() * input.size(0)

            if not regression:
                pred = output.data.argmax(1, keepdim=True)
                correct += pred.eq(target.data.view_as(pred)).sum().item()

    return {
        'loss': loss_sum / len(loader.dataset),
        'accuracy': None if regression else correct / len(loader.dataset) * 100.0,
    }

def moving_average(net1, net2, alpha=1):
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        param1.data *= (1.0 - alpha)
        param1.data += param2.data * alpha


def _check_bn(module, flag):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        flag[0] = True


def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]


def reset_bn(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)


def _get_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]


def bn_update(loader, model, **kwargs):
    """
        BatchNorm buffers update (if any).
        Performs 1 epochs to estimate buffers average using train dataset.

        :param loader: train dataset loader for buffers average estimation.
        :param model: model being update
        :return: None
    """
    if not check_bn(model):
        return
    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    n = 0
    for input, _ in loader:
        input = input.cuda(non_blocking=True)
        input_var = torch.autograd.Variable(input)
        b = input_var.data.size(0)

        momentum = b / (n + b)
        for module in momenta.keys():
            module.momentum = momentum

        model(input_var, **kwargs)
        n += b

    model.apply(lambda module: _set_momenta(module, momenta))

def inv_softmax(x, eps = 1e-10):
    return torch.log(x/(1.0 - x + eps))

def predictions(test_loader, model, seed=None, cuda=True, regression=False, **kwargs):
    #will assume that model is already in eval mode
    #model.eval()
    preds = []
    targets = []
    for input, target in test_loader:
        if seed is not None:
            torch.manual_seed(seed)
        if cuda:
            input = input.cuda(async=True)
        output = model(input, **kwargs)
        if regression:
            preds.append(output.cpu().data.numpy())
        else:
            probs = F.softmax(output, dim=1)
            preds.append(probs.cpu().data.numpy())
        targets.append(target.numpy())
    return np.vstack(preds), np.concatenate(targets)