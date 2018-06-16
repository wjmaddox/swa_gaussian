r"""
This will eventually incorporate the Laplace approximations and use variances as well, but not yet...
"""
import argparse
import torch
import os
import models, swag, data, utils
import torch.nn.functional as F
import numpy as np
from itertools import chain, product
import tabulate
import itertools
#from load_different_state_dict import load_different_state_dict, _load_from_different_state_dict

parser = argparse.ArgumentParser(description='SGD/SWA/SWAG ensembling')
parser.add_argument('--replications', type=int, default=10, help='number of passes through testing set')
parser.add_argument('--dir', type=str, default=None, required=True, help='training directory (default: None)')

parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset name (default: CIFAR10)')
parser.add_argument('--data_path', type=str, default='/scratch/datasets/', metavar='PATH',
                    help='path to datasets location (default: None)')
parser.add_argument('--use_test', dest='use_test', action='store_true', help='use test dataset instead of validation (default: False)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='input batch size (default: 128)')
parser.add_argument('--num_workers', type=int, default=4, metavar='N', help='number of workers (default: 4)')
parser.add_argument('--model', type=str, default='VGG16', metavar='MODEL',
                    help='model name (default: VGG16)')

parser.add_argument('--epoch', type=int, default=200, metavar='N', help='epoch to evaluate at (default: 200)')

parser.add_argument('--swa', action='store_true', help='swa usage flag (default: off)')
parser.add_argument('--cov_mat', action='store_true', help='save sample covariance')

parser.add_argument('--plot', action='store_true', help='plot replications (default: off)')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

args = parser.parse_args()

if args.cov_mat:
    args.no_cov_mat = False
else:
    args.no_cov_mat = True

torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

print('Using model %s' % args.model)
model_cfg = getattr(models, args.model)

print('Loading dataset %s from %s' % (args.dataset, args.data_path))
loaders, num_classes = data.loaders(
    args.dataset,
    args.data_path,
    args.batch_size,
    args.num_workers,
    model_cfg.transform_train,
    model_cfg.transform_test,
    use_validation=False
)

swag_model_location = args.dir + '/swag-' + str(args.epoch) + '.pt'
model_location = args.dir + '/checkpoint-' + str(args.epoch) + '.pt'
print('Loading sgd model at ' + model_location + ' and swag_model at ' + swag_model_location)

print(torch.cuda.memory_allocated()/(1024.0 ** 3))
swag_checkpoint = torch.load(swag_model_location)
print(torch.cuda.memory_allocated()/(1024.0 ** 3))

print('Preparing model')
model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
model.cuda()

print('SWAG training')

swag_model = swag.SWAG(model_cfg.base, no_cov_mat=args.no_cov_mat, max_num_models = 20, loading = True, *model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)

swag_model.cuda()

swag_model.load_state_dict(swag_checkpoint['state_dict'])
#print(torch.cuda.memory_allocated()/(1024.0 ** 3))

#print(torch.cuda.memory_allocated()/(1024.0 ** 3))
model_checkpoint = torch.load(model_location)
model.load_state_dict(model_checkpoint['state_dict'])

def compute_predictions(loader, model, criterion=None):
    x = []
    output = []
    y = []
    for (input, target) in loader:
        input = input.cuda(async=True)
        target = target.cuda(async=True)

        x.append(input.detach().cpu().numpy())
        output.append(model(input).detach().cpu().numpy())
        y.append(target.detach().cpu().numpy())
    return x, y, output

x, y, pred = compute_predictions(loaders['test'], model)

swag_model.sample(0.0)
x, y, swa_pred = compute_predictions(loaders['test'], swag_model)

import matplotlib.pyplot as plt
plt.scatter(x[0], y[0], c='blue')
plt.scatter(x[0], pred[0], c='green')
plt.scatter(x[0], pred[0], c='red')
plt.savefig('plots/toy_problem.png')

