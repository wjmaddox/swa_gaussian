import torch
import os
import copy
from datetime import datetime

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


def train_epoch(loader, model, criterion, optimizer, batch_means=False):
    loss_sum = 0.0
    correct = 0.0

    model.train()

    #zero means before batch begins
    if batch_means:
        #curr_mem_usage = torch.cuda.memory_allocated()
        avg_params = list()
        for param in model.parameters():
            avg_params.append(copy.deepcopy(param))
        #new_mem_usage = torch.cuda.memory_allocated() - curr_mem_usage
        #print(new_mem_usage/(1024.0 ** 3))

    for i, (input, target) in enumerate(loader):
        input = input.cuda(async=True)
        target = target.cuda(async=True)

        output = model(input)
        loss = criterion(output, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_means:
            for j, (avg, param) in enumerate(zip(avg_params, model.parameters())):
                avg_params[j] = i/(i+1) * avg.data + 1/(i+1) * param.data
            
        loss_sum += loss.data.item() * input.size(0)
        
        if criterion.__name__ == 'cross_entropy':
            pred = output.data.argmax(1, keepdim=True)
            correct += pred.eq(target.data.view_as(pred)).sum().item()
        if criterion.__name__ == 'mse_loss':
            correct = (target.data.view_as(output) - output).pow(2).mean().sqrt().item()
    
    if batch_means:
        return avg_params, {
            'loss': loss_sum / len(loader.dataset),
            'accuracy': correct / len(loader.dataset) * 100.0,
        }
    else:
        return {
            'loss': loss_sum / len(loader.dataset),
            'accuracy': correct / len(loader.dataset) * 100.0,
        }


def eval(loader, model, criterion):
    loss_sum = 0.0
    correct = 0.0

    model.eval()

    for i, (input, target) in enumerate(loader):
        input = input.cuda(async=True)
        target = target.cuda(async=True)

        output = model(input)
        loss = criterion(output, target)

        loss_sum += loss.item() * input.size(0)

        if criterion.__name__ == 'cross_entropy':
            pred = output.data.argmax(1, keepdim=True)
            correct += pred.eq(target.data.view_as(pred)).sum().item()
        if criterion.__name__ == 'mse_loss':
            correct = (target.data.view_as(output) - output).pow(2).mean().sqrt().item()

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
        input = input.cuda(async=True)
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
    #set seed by time + range(samples) basically
    seed_base = int(datetime.now().timestamp())

    for (input, target) in loaders['test']:
        input = input.cuda(async=True)
        target = target.cuda(async=True)

        full_output_prob = 0.0
        for i in range(samples):

            #randomly sample from N(swa, swa_var) with seed i
            swa_model.sample(scale=scale, cov=cov, seed=i+seed_base)
            #perform batch norm update with training
            bn_update(loaders['train'], swa_model)

            """#now iterate through dataset
            sample_output = torch.zeros(0, num_classes)
            for j, (input, target) in enumerate(loader):
                input = input.cuda(async=True)
                target = target.cuda(async=True)"""

            output = swa_model(input)
            #print('indiv model loss:', criterion(output,target).data.item())
            if criterion.__name__ == 'cross_entropy':
                #print(output.size())
                #print(torch.nn.Softmax(dim=1)(output).sum(dim=1).size())
                full_output_prob += torch.nn.Softmax(dim=1)(output) #avg of probabilities
            else:
                full_output_prob += output #avg for mse?
        
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