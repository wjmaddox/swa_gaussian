import os
import torch
import tqdm
from datetime import datetime
import torch.nn.functional as F
import numpy as np

from ..utils import bn_update, LogSumExp, predictions
from .laplace import KFACLaplace

def eval_dropout(model, loaders, criterion, path, num_classes, samples = 10, **kwargs):
    r"""loader: dataset loader
    model: stochastic weight averaging model
    criterion: loss function
    samples: number of samples to draw from dropout approximation"""

    checkpoint = torch.load(path)
    if 'model_state' in checkpoint.keys():
        model.load_state_dict(checkpoint['model_state'])
    else:
        model.load_state_dict(checkpoint['state_dict'])

    model.eval()

    #num_classes = max(loaders['train'].dataset.labels)
    predictions_sum = np.zeros((len(loaders['test'].dataset), num_classes))

    #function that sets dropout to train mode
    def train_dropout(m):
        if type(m)==torch.nn.modules.dropout.Dropout:
            m.train()
    model.apply(train_dropout)

    #set seed by time + range(samples)
    #this is to ensure that we get different random numbers each time
    seed_base = int(datetime.now().timestamp())
    with torch.no_grad():
        for i in range(samples):
            torch.manual_seed(i + seed_base)

            pred_probs, targets = predictions(loaders['test'], model)
            acc = 100.0 * np.mean(np.argmax(pred_probs, axis=1) == targets)

            predictions_sum += pred_probs
            ens_acc = 100.0 * np.mean(np.argmax(predictions_sum, axis=1) == targets)

            #test loss
            targets_tensor = torch.tensor(targets)
            log_predictions_tensor = torch.tensor(pred_probs).log()
            log_ens_predictions_tensor = torch.tensor(predictions_sum/(i+1)).log()
            
            loss = criterion(log_predictions_tensor, targets_tensor)
            ens_loss = criterion(log_ens_predictions_tensor, targets_tensor)

            print('Model accuracy: %8.4f. Ensemble accuracy: %8.4f' % (acc, ens_acc))
            print('Model loss: %8.4f. Ensemble loss: %8.4f' % (loss, ens_loss))

    return {
        'loss': ens_loss,
        'accuracy': ens_acc
    }

def eval_laplace(model, loaders, criterion, path, num_classes, samples = 10, **kwargs):
    t_input, t_target = next(iter(loaders['train']))
    t_input, t_target = t_input.cuda(non_blocking = True), t_target.cuda(non_blocking = True)
    
    checkpoint = torch.load(path)
    if 'model_state' in checkpoint.keys():
        model.load_state_dict(checkpoint['model_state'])
        weight_decay = checkpoint['optimizer_state']['param_groups'][0]['weight_decay']
    else:
        model.load_state_dict(checkpoint['state_dict'])
        weight_decay = checkpoint['optimizer']['param_groups'][0]['weight_decay']
    model.eval()

    print('Preparing Laplace model')
    laplace_model = KFACLaplace(model, eps = weight_decay, data_size = len(loaders['train'].dataset))

    predictions_sum = np.zeros((len(loaders['test'].dataset), num_classes))

    for i in range(samples):
        #reload original state dict
        laplace_model.net.load_state_dict(laplace_model.mean_state)

        #set up the sampling procedure first
        #requires one backwards call to get gradients
        laplace_model.net.train()

        output = laplace_model.net(t_input)

        loss = criterion(output, t_target)
        loss.backward()
        laplace_model.step(update_params = False)

        #now resample and update batch norm parameters
        laplace_model.sample()
        bn_update(loaders['train'], laplace_model.net)

        #eval mode for testing
        laplace_model.net.eval()

        pred_probs, targets = predictions(loaders['test'], model)
        acc = 100.0 * np.mean(np.argmax(pred_probs, axis=1) == targets)

        predictions_sum += pred_probs
        ens_acc = 100.0 * np.mean(np.argmax(predictions_sum, axis=1) == targets)

        #test loss
        targets_tensor = torch.tensor(targets)
        log_predictions_tensor = torch.tensor(pred_probs).log()
        log_ens_predictions_tensor = torch.tensor(predictions_sum/(i+1)).log()
        
        loss = criterion(log_predictions_tensor, targets_tensor)
        ens_loss = criterion(log_ens_predictions_tensor, targets_tensor)

        print('Model accuracy: %8.4f. Ensemble accuracy: %8.4f' % (acc, ens_acc))
        print('Model loss: %8.4f. Ensemble loss: %8.4f' % (loss, ens_loss))
    
    return {
        'loss': ens_loss,
        'accuracy': ens_acc
    }

def eval_ecdf(model, loaders, criterion, dir, num_classes, **kwargs):
    predictions_sum = np.zeros((len(loaders['test'].dataset), num_classes))

    ckpt_models = find_models(dir)
    ckpt_models.reverse() #start ensembles from epoch 300 and go back
    #print(ckpt_models)
    for i, path in enumerate(ckpt_models):
        
        print(path)
        checkpoint = torch.load(path)
        if 'model_state' in checkpoint.keys():
            model.load_state_dict(checkpoint['model_state'])
        else:
            model.load_state_dict(checkpoint['state_dict'])

        model.eval()

        pred_probs, targets = predictions(loaders['test'], model)
        acc = 100.0 * np.mean(np.argmax(pred_probs, axis=1) == targets)

        predictions_sum += pred_probs
        ens_acc = 100.0 * np.mean(np.argmax(predictions_sum, axis=1) == targets)

        #test loss
        targets_tensor = torch.tensor(targets)
        log_predictions_tensor = torch.tensor(pred_probs).log()
        log_ens_predictions_tensor = torch.tensor(predictions_sum/(i+1)).log()
        
        loss = criterion(log_predictions_tensor, targets_tensor)
        ens_loss = criterion(log_ens_predictions_tensor, targets_tensor)

        print('Model accuracy: %8.4f. Ensemble accuracy: %8.4f' % (acc, ens_acc))
        print('Model loss: %8.4f. Ensemble loss: %8.4f' % (loss, ens_loss))

    return {
        'loss': ens_loss,
        'accuracy': ens_acc
    }

def eval_swag(swag_model, loaders, criterion, path, num_classes, samples = 10, cov = True, fullrank = True, **kwargs):
    print(path)
    checkpoint = torch.load(path)
    if 'model_state' in checkpoint.keys():
        swag_model.load_state_dict(checkpoint['model_state'])
    else:
        swag_model.load_state_dict(checkpoint['state_dict'])
    swag_model.eval()

    predictions_sum = np.zeros((len(loaders['test'].dataset), num_classes))

    for i in range(samples):

        #now resample and update batch norm parameters
        swag_model.sample(scale=0.0, cov=cov, fullrank=fullrank, block = False)
        bn_update(loaders['train'], swag_model)

        #eval mode for testing
        swag_model.eval()

        pred_probs, targets = predictions(loaders['test'], swag_model)
        acc = 100.0 * np.mean(np.argmax(pred_probs, axis=1) == targets)

        predictions_sum += pred_probs
        ens_acc = 100.0 * np.mean(np.argmax(predictions_sum, axis=1) == targets)

        #test loss
        targets_tensor = torch.tensor(targets)
        log_predictions_tensor = torch.tensor(pred_probs).log()
        log_ens_predictions_tensor = torch.tensor(predictions_sum/(i+1)).log()
        
        loss = criterion(log_predictions_tensor, targets_tensor)
        ens_loss = criterion(log_ens_predictions_tensor, targets_tensor)

        print('Model accuracy: %8.4f. Ensemble accuracy: %8.4f' % (acc, ens_acc))
        print('Model loss: %8.4f. Ensemble loss: %8.4f' % (loss, ens_loss))
    
    return {
        'loss': ens_loss,
        'accuracy': ens_acc
    }

def find_models(dir, start_epoch = None):
    #this is to generate the list of models we'll be searching for
    if type(dir) is str:
        all_models = os.popen('ls ' + dir + '/checkpoint*.pt').read().split('\n')
    else:
        all_models = [os.popen('ls ' + d + '/checkpoint*.pt').read().split('\n') for d in dir]

    #print('all_models: ',all_models[0][:-1])
    if start_epoch is not None:
        model_epochs = [int(t.replace('.', '-').split('-')[1]) for t in all_models[:-1]]
        models_to_use = [t >= start_epoch for t in model_epochs]

        model_names = list()
        for model_name, use in zip(all_models, models_to_use):
            if use is True:
                model_names.append(model_name)

        return model_names
    
    else:
        return all_models[0][:-1]