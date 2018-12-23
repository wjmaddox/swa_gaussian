import argparse
import numpy as np
import os
import sys
import torch
import torch.nn.functional as F

from swag import data, models, utils
from swag.posteriors.utils import eval_ecdf, eval_dropout, eval_laplace, eval_swag
from swag.posteriors import SWAG

parser = argparse.ArgumentParser(description='Ensemble evaluation')

parser.add_argument('--dataset', type=str, default='CIFAR10', metavar='DATASET',
                    help='dataset name (default: CIFAR10)')
parser.add_argument('--use_test', action='store_true',
                    help='switches between validation and test set (default: validation)')
parser.add_argument('--transform', type=str, default='VGG', metavar='TRANSFORM',
                    help='transform name (default: VGG)')
parser.add_argument('--data_path', type=str, default=None, metavar='PATH',
                    help='path to datasets location (default: None)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size (default: 128)')
parser.add_argument('--num-workers', type=int, default=4, metavar='N',
                    help='number of workers (default: 4)')

parser.add_argument('--model', type=str, default=None, metavar='MODEL',
                    help='model name (default: None)')

parser.add_argument('--ckpt', type=str, action='append', metavar='CKPT', required=True,
                    help='checkpoint to eval, pass all the models through this parameter')

parser.add_argument('--method', type=str, choices = ['empirical', 'dropout', 'SWAG', 'SWAG-Diagonal', 'SWAG-LR', 'KFACLaplace'], 
                    required = True, help='method to compute ensembles with')

parser.add_argument('--samples', type=int, default=10, help='number of samples to compute MC estimates, if used with method')
args = parser.parse_args()

torch.backends.cudnn.benchmark = True

#loaders, num_classes = data.loaders(
#    args.dataset,
#    args.data_path,
#    args.batch_size,
#    args.num_workers,
#    args.transform,
#    args.use_test
#)

architecture = getattr(models, args.model)

loaders, num_classes = data.loaders(
    args.dataset,
    args.data_path,
    args.batch_size,
    args.num_workers,
    architecture.transform_train,
    architecture.transform_test,
    use_validation=not args.use_test,
    split_classes=None
)

model = architecture.base(num_classes=num_classes, **architecture.kwargs)
criterion = F.cross_entropy

model.cuda()

ensemble_size = 0

if args.method == 'empirical':
    print('Warning: args.ckpt should be a directory')
    eval_ecdf(model, loaders, criterion, args.ckpt, num_classes)
elif args.method == 'dropout':
    eval_dropout(model, loaders, criterion, args.ckpt[0], num_classes, args.samples)
elif args.method == 'KFACLaplace':
    eval_laplace(model, loaders, criterion, args.ckpt[0], num_classes, args.samples)
else:
    if args.method == 'SWAG-Diagonal':
        use_cov_mat = False
        full_rank = None
    else:
        use_cov_mat = True
        
        if args.method == 'SWAG-LR':
            full_rank = False
        else:
            full_rank = True

    swag_model = SWAG(architecture.base, no_cov_mat=False, max_num_models = 20, loading = True, *architecture.args,
                    num_classes=num_classes, **architecture.kwargs)
    swag_model.cuda()

    eval_swag(swag_model, loaders, criterion, args.ckpt[0], num_classes, args.samples, use_cov_mat, full_rank)

    