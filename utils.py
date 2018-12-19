import torch
import os
import copy
from datetime import datetime
import math

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


def train_epoch(loader, model, criterion, optimizer):
    loss_sum = 0.0
    correct = 0.0

    model.train()

    for i, (input, target) in enumerate(loader):
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        loss, output = criterion(model, input, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
            
        loss_sum += loss.data.item() * input.size(0)
        
        #if criterion.__name__ == 'cross_entropy':
        pred = output.data.argmax(1, keepdim=True)
        correct += pred.eq(target.data.view_as(pred)).sum().item()
        """if criterion.__name__ == 'mse_loss':
            correct = (target.data.view_as(output) - output).pow(2).mean().sqrt().item()"""
    
    return {
        'loss': loss_sum / len(loader.dataset),
        'accuracy': correct / len(loader.dataset) * 100.0,
    }


def eval(loader, model, criterion):
    loss_sum = 0.0
    correct = 0.0

    model.eval()

    with torch.no_grad():
        for i, (input, target) in enumerate(loader):
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            loss, output = criterion(model, input, target)

            loss_sum += loss.item() * input.size(0)

            #if criterion.__name__ == 'cross_entropy':
            pred = output.data.argmax(1, keepdim=True)
            correct += pred.eq(target.data.view_as(pred)).sum().item()
            #if criterion.__name__ == 'mse_loss':
            #    correct = (target.data.view_as(output) - output).pow(2).mean().sqrt().item()

    return {
        'loss': loss_sum / len(loader.dataset),
        'accuracy': correct / len(loader.dataset) * 100.0,
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

def fast_ensembling(loaders, swa_model, criterion, samples = 10, cov=True, scale = 1.0):
    r"""loader: dataset loader
    swa_model: stochastic weight averaging model
    criterion: loss function
    samples: number of samples to draw from laplace approximation
    scale: multiple to scale the variance of laplace approximation by
    cov: whether to use the estimated covariance matrix """

    loss_sum = 0.0
    correct = 0.0

    #set seed by time + range(samples)
    #this is to ensure that we get different random numbers each time
    seed_base = int(datetime.now().timestamp())

    for (input, target) in loaders['test']:
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        full_output_prob = 0.0
        for i in range(samples):
            #randomly sample from N(swa, swa_var) with seed i
            swa_model.sample(scale=scale, cov=cov, seed=i+seed_base)

            #perform batch norm update with training data
            bn_update(loaders['train'], swa_model)

            #forwards pass through network
            output = swa_model(input)

            #averaging is slightly different based on loss
            if criterion.__name__ == 'cross_entropy':
                full_output_prob += torch.nn.Softmax(dim=1)(output) #avg of probabilities
            else:
                full_output_prob += output #avg for mse
        
        full_output_prob /= samples
        #print(full_output.size())
        eps = 1e-20
        #full_output_logit = torch.log(full_output_prob/(1.0 - full_output_prob + eps))
        full_output_logit = full_output_prob.log()
        #print((full_output_logit - output).sum())
        loss = criterion(full_output_logit, target)
        #print('avg model loss: ', loss.data.item())

        loss_sum += loss.data.item() * input.size(0)

        if criterion.__name__ == 'cross_entropy':
            pred = full_output_logit.data.argmax(1, keepdim=True)
            correct += pred.eq(target.data.view_as(pred)).sum().item()
        else:
            correct = (target.data.view_as(output) - output).pow(2).mean().sqrt().item()
            #sample_output = torch.cat((sample_output, output))
        
            ##average output to full output
            #full_output += 1/samples * sample_output

    return {
        'loss': loss_sum / len(loaders['test'].dataset),
        'accuracy': correct / len(loaders['test'].dataset) * 100.0
    }

def fast_importance_sampling(loaders, swa_model, criterion, samples = 10, cov=True, scale = 1.0):
    r"""loader: dataset loader
    swa_model: stochastic weight averaging model
    criterion: loss function
    samples: number of samples to draw from laplace approximation
    scale: multiple to scale the variance of laplace approximation by
    cov: whether to use the estimated covariance matrix """

    loss_sum = 0.0
    correct = 0.0
    #set seed by time + range(samples) basically
    seed_base = int(datetime.now().timestamp())

    log_weights = torch.zeros(samples) #log weights = 0
    if samples > 1:
        for i in range(samples):
            #randomly sample from N(swa, swa_var) with seed i+base
            swa_model.sample(scale=scale, cov=cov, seed=i+seed_base)
            #perform batch norm update with training
            bn_update(loaders['train'], swa_model)

            #now iterate through train to calculate weight
            train_res = eval(loaders['train'], swa_model, criterion)
            train_loss = -train_res['loss'] * len(loaders['train'].dataset)

            approx_ll = swa_model.compute_logprob()
            approx_ll_prior = swa_model.compute_logprob(use_pars=False)
            #print(train_loss, approx_ll, approx_ll_prior)
            log_weights[i] = train_loss + (approx_ll_prior - approx_ll)/(math.pow(len(loaders['train'].dataset),2))
        #print(log_weights)
    log_weights = log_weights.cuda()
    
    #print(log_weights)
    #print(LogSumExp(log_weights))

    for (input, target) in loaders['test']:
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        
        full_output_logprob = torch.zeros(samples, input.size(0), target.max() + 1).cuda()
        for i in range(samples):
            #randomly sample from N(swa, swa_var) with seed i
            swa_model.sample(scale=scale, cov=cov, seed=i+seed_base)
            #perform batch norm update with training
            bn_update(loaders['train'], swa_model)

            output = swa_model(input)
            if criterion.__name__ == 'cross_entropy':
                loss_output = torch.nn.LogSoftmax(dim=1)(output)
            else:
                loss_output = output

            if i==0:
                full_output_logprob = loss_output + log_weights[i]
                full_output_logprob = full_output_logprob.unsqueeze(0)
            else:
                res = loss_output + log_weights[i]
                full_output_logprob = torch.cat((full_output_logprob, res.unsqueeze(0)),dim=0)
        
        output_logprob = LogSumExp(full_output_logprob) - LogSumExp(log_weights)
        output_logprob = output_logprob.squeeze(0)
        loss = criterion(output_logprob, target)

        loss_sum += loss.data.item() * input.size(0)

        if criterion.__name__ == 'cross_entropy':
            pred = output_logprob.data.argmax(1, keepdim=True)
            correct += pred.eq(target.data.view_as(pred)).sum().item()
        else:
            correct = (target.data.view_as(output) - output).pow(2).mean().sqrt().item()

    return {
        'loss': loss_sum / len(loaders['test'].dataset),
        'accuracy': correct / len(loaders['test'].dataset) * 100.0
    }