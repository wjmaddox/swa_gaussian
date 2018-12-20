import os
import torch
import tqdm
from datetime import datetime
import torch.nn.functional as F

from ..utils import bn_update, LogSumExp

def eval_dropout(loaders, model, criterion, samples = 10):
    r"""loader: dataset loader
    model: stochastic weight averaging model
    criterion: loss function
    samples: number of samples to draw from dropout approximation"""

    correct = 0.0
    model.eval()

    #function that sets dropout to train mode
    def train_dropout(m):
        if type(m)==torch.nn.modules.dropout.Dropout:
            m.train()
    model.apply(train_dropout)

    #set seed by time + range(samples)
    #this is to ensure that we get different random numbers each time
    seed_base = int(datetime.now().timestamp())
    target_list = []
    with torch.no_grad():
        epoch_logprob_list = []
        for i in range(samples):
            torch.manual_seed(i + seed_base)

            batch_prob_list = []

            for j, (input, target) in enumerate(loaders['test']):
                #set seed so dropout should be deterministic by seed
                torch.manual_seed(i + seed_base)

                input = input.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)
                if i==0:
                    target_list.append(target)

                #standard forwards pass but we don't reduce
                output = model(input)
                #loss = criterion(output, target, reduction='none')
                prob = torch.nn.Softmax(dim=1)(output).unsqueeze(1)
                batch_prob_list.append(prob)

            #stack all results
            batch_logprobs = torch.cat(batch_prob_list)
            epoch_logprob_list.append(batch_logprobs)

    epoch_prob = torch.cat(epoch_logprob_list, dim=1).permute(0, 2, 1)

    epoch_logprobs = (epoch_prob.mean(dim=2)+1e-6).log()

    target_stack = torch.cat(target_list)
    pred = epoch_logprobs.data.argmax(1, keepdim=True)
    correct = pred.eq(target_stack.data.view_as(pred)).sum().item()

    loss = criterion(epoch_logprobs, target_stack)

    return {
        'loss': loss.data.item(),
        'accuracy': (correct / len(loaders['test'].dataset)) * 100.0,
    } 

def find_models(dir, start_epoch):
    #this is to generate the list of models we'll be searching for

    all_models = os.popen('ls ' + dir + '/checkpoint*.pt').read().split('\n')
    model_epochs = [int(t.replace('.', '-').split('-')[1]) for t in all_models[:-1]]
    models_to_use = [t >= start_epoch for t in model_epochs]

    model_names = list()
    for model_name, use in zip(all_models, models_to_use):
        if use is True:
            model_names.append(model_name)

    return model_names

def eval_ecdf(loader, model, locs):
    loss_sum = 0.0
    correct = 0.0

    with torch.no_grad():
        for (input, target) in tqdm.tqdm(loader):

            input = input.cuda(async=True)
            target = target.cuda(async=True)

            full_output_prob = 0.0
            for loc in locs:
                #randomly sample from N(swa, swa_var) with seed i
                #swa_model.sample(scale=scale, cov=cov, seed=i+seed_base)
                model.load_state_dict(torch.load(loc)['state_dict'])

                output = model(input)
                full_output_prob += torch.nn.Softmax(dim=1)(output)
                #print('indiv model loss:', criterion(output,target).data.item())
                #if criterion.__name__ == 'cross_entropy':
                #    full_output_prob += torch.nn.Softmax(dim=1)(output) #avg of probabilities
                #else:
                #    full_output_prob += output #avg for mse?
            
            full_output_prob /= len(locs)

            full_output_logit = full_output_prob.log()
            loss = F.cross_entropy(full_output_logit, target)

            loss_sum += loss.data.item() * input.size(0)

            #if criterion.__name__ == 'cross_entropy':
            pred = full_output_logit.data.argmax(1, keepdim=True)
            correct += pred.eq(target.data.view_as(pred)).sum().item()
            #else:
            #    correct = (target.data.view_as(output) - output).pow(2).mean().sqrt().item()

        return {
            'loss': loss_sum / len(loader.dataset),
            'accuracy': correct / len(loader.dataset) * 100.0
        }