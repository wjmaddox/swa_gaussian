import argparse
import torch
import os
import models, swag, data, utils
import torch.nn.functional as F
import numpy as np
from itertools import chain, product
import tabulate
import itertools
import tqdm

parser = argparse.ArgumentParser(description='SGD/SWA/SWAG ensembling')
#parser.add_argument('--replications', type=int, default=10, help='number of passes through testing set')
parser.add_argument('--dir', type=str, nargs='+', default=None, required=True, help='loading directories (default: None)')

parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset name (default: CIFAR10)')
parser.add_argument('--data_path', type=str, default='/scratch/datasets/', metavar='PATH',
                    help='path to datasets location (default: None)')
parser.add_argument('--use_test', dest='use_test', action='store_true', help='use test dataset instead of validation (default: False)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='input batch size (default: 128)')
parser.add_argument('--num_workers', type=int, default=4, metavar='N', help='number of workers (default: 4)')
parser.add_argument('--model', type=str, default='VGG16', metavar='MODEL',
                    help='model name (default: VGG16)')

parser.add_argument('--epoch', type=int, default=160, metavar='N', help='epoch to begin evaluations at at (default: 200)')

parser.add_argument('--no_ensembles', action='store_true', help='whether to run ensembles or not')
parser.add_argument('--no_swa', action='store_true', help='whether to create swa from checkpoints')
#parser.add_argument('--swa', action='store_true', help='swa usage flag (default: off)')
#parser.add_argument('--cov_mat', action='store_true', help='save sample covariance')

#parser.add_argument('--plot', action='store_true', help='plot replications (default: off)')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--save_path', type=str, required=True, help='output file to save to')
args = parser.parse_args()

torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

print('Using model %s' % args.model)
model_cfg = getattr(models, args.model)

#sorry this is hardcoded for now
if args.dataset == 'CIFAR10.1':
    #from torchvision import transforms
    import sys
    sys.path.append('/home/wm326/CIFAR-10.1/code')
    from cifar10_1_dataset import cifar10_1
    
    dataset = cifar10_1(transform=model_cfg.transform_test)
    loaders = {'test': torch.utils.data.DataLoader(dataset, batch_size = args.batch_size, num_workers = args.num_workers)}

    num_classes = 10
else:
    print('Loading dataset %s from %s' % (args.dataset, args.data_path))
    loaders, num_classes = data.loaders(
        args.dataset,
        args.data_path,
        args.batch_size,
        args.num_workers,
        model_cfg.transform_train,
        model_cfg.transform_test,
        use_validation=not args.use_test,
        split_classes=None
    )

print('Preparing model')
model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
model.cuda()

print('using cross-entropy loss')
criterion = F.cross_entropy

#this is to generate the list of models we'll be searching for
def find_models(dir):
    all_models = os.popen('ls ' + dir + '/*.pt').read().split('\n')
    model_epochs = [int(t.replace('.', '-').split('-')[1]) for t in all_models[:-1]]
    models_to_use = [t >= args.epoch for t in model_epochs]

    model_names = list()
    for model_name, use in zip(all_models, models_to_use):
        if use is True:
            model_names.append(model_name)

    return model_names

def empirical_dist_ensembling(loader, model, locs):
    loss_sum = 0.0
    correct = 0.0

    for (input, target) in tqdm.tqdm(loader):

        input = input.cuda(async=True)
        target = target.cuda(async=True)

        full_output_prob = 0.0
        for loc in locs:
            #randomly sample from N(swa, swa_var) with seed i
            #swa_model.sample(scale=scale, cov=cov, seed=i+seed_base)
            model.load_state_dict(torch.load(loc)['state_dict'])

            output = model(input)
            #print('indiv model loss:', criterion(output,target).data.item())
            if criterion.__name__ == 'cross_entropy':
                full_output_prob += torch.nn.Softmax(dim=1)(output) #avg of probabilities
            else:
                full_output_prob += output #avg for mse?
        
        full_output_prob /= len(locs)

        full_output_logit = full_output_prob.log()
        loss = criterion(full_output_logit, target)

        loss_sum += loss.data.item() * input.size(0)

        if criterion.__name__ == 'cross_entropy':
            pred = full_output_logit.data.argmax(1, keepdim=True)
            correct += pred.eq(target.data.view_as(pred)).sum().item()
        else:
            correct = (target.data.view_as(output) - output).pow(2).mean().sqrt().item()

    return {
        'loss': loss_sum / len(loader.dataset),
        'accuracy': correct / len(loader.dataset) * 100.0
    }

if not args.no_ensembles:
    ensemble_loss = list()
    ensemble_accuracy = list()

    for dir in args.dir:
        #print('now running ' + dir)
        dir_locs = find_models(dir)
        res = empirical_dist_ensembling(loaders['test'], model, dir_locs)
        ensemble_loss.append(res['loss'])
        ensemble_accuracy.append(res['accuracy'])

    columns = ['model', 'starting_epoch', 'acc', 'acc_var', 'loss', 'loss_var']
    values = [args.model, args.epoch, np.mean(ensemble_accuracy), np.var(ensemble_accuracy), np.mean(ensemble_loss), np.var(ensemble_loss)]
    table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='8.4f')
    print(table)

    np.savez(args.save_path, result=values)

if not args.no_swa:
    for dir in args.dir:
        print('now running ' + dir)
        dir_locs = find_models(dir)

        print('SWAG training')
        swag_model = swag.SWAG(model_cfg.base, no_cov_mat=False, max_num_models=29, *model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
        swag_model.cuda()

        for loc in dir_locs:
            #one epoch of training for batch means
            #model_batch_means, _ = utils.train_epoch(loaders['train'], model, criterion, optimizer, batch_means=True)

            model.load_state_dict(torch.load(loc)['state_dict'])
            swag_model.collect_model(model, bm=None)

        #save checkpoint of swa model
        utils.save_checkpoint(
            dir,
            300,
            name=args.save_path,
            state_dict=swag_model.state_dict(),
        )




