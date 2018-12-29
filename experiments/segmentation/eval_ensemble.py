import argparse
import numpy as np
import os
import sys

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

from swag import data, models, utils
#from swag.posteriors.utils import eval_ecdf, eval_dropout, eval_laplace, eval_swag
from swag.posteriors import SWAG
from swag.utils import bn_update

parser = argparse.ArgumentParser(description='Ensemble evaluation')

parser.add_argument('--use_test', action='store_true',
                    help='switches between validation and test set (default: validation)')
parser.add_argument('--data_path', type=str, default=None, metavar='PATH',
                    help='path to datasets location (default: None)')
parser.add_argument('--batch_size', type=int, default=2, metavar='N',
                    help='input batch size (default: 2)')
parser.add_argument('--num-workers', type=int, default=4, metavar='N',
                    help='number of workers (default: 4)')

parser.add_argument('--model', type=str, default=None, metavar='MODEL',
                    help='model name (default: None)')

parser.add_argument('--ckpt', type=str, action='append', metavar='CKPT', required=True,
                    help='checkpoint to eval, pass all the models through this parameter')

parser.add_argument('--method', type=str, choices = ['empirical', 'dropout', 'SWAG', 'SWAG-Diagonal', 'SWAG-LR', 'KFACLaplace'],
                    default = 'SWAG-Diagonal', help='method to compute ensembles with')

parser.add_argument('--samples', type=int, default=10, help='number of samples to compute MC estimates, if used with method')
args = parser.parse_args()

torch.backends.cudnn.benchmark = True

normalize = transforms.Normalize(mean=camvid.mean, std=camvid.std)
train_joint_transformer = transforms.Compose([
    #joint_transforms.JointRandomCrop(224), # commented for fine-tuning
    joint_transforms.JointRandomHorizontalFlip()
    ])
train_dset = camvid.CamVid(args.data_path, 'train',
      joint_transform=train_joint_transformer,
      transform=transforms.Compose([
          transforms.ToTensor(),
          normalize,
    ]))
train_loader = torch.utils.data.DataLoader(
    train_dset, batch_size=args.batch_size, shuffle=True)

if args.use_test:
    test_dset = camvid.CamVid(
        args.data_path, 'test', joint_transform=None,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize
        ]))
    test_loader = torch.utils.data.DataLoader(
        test_dset, batch_size=args.batch_size, shuffle=False)
else:
    test_dset = camvid.CamVid(
        args.data_path, 'val', joint_transform=None,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize
        ]))
    test_loader = torch.utils.data.DataLoader(
        test_dset, batch_size=args.batch_size, shuffle=False)

loaders = {'train': train_loader, 'test': test_loader}

#model = tiramisu.FCDenseNet67(n_classes=12).cuda()
criterion = nn.NLLLoss2d(weight=camvid.class_weight)
#model.cuda()

def eval_swag(model, loaders, criterion, samples, use_cov_mat, full_rank):
    for i in range(samples):
        model.sample(scale = 1e-3, cov = False, block = True)
        bn_update(loaders['train'], model)

        model.eval()

        with torch.no_grad():
            preds, targets = [], []
            for data, target in loaders['test']:
                data = data.cuda()
                output = model(data)

                preds.append(output.cpu().data.numpy())
                targets.append(target.numpy())
            predictions = np.vstack(preds)
            targets = np.concatenate(targets)

            if i is 0:
                predictions_sum = predictions
            else:
                predictions_sum += predictions

            acc = 100.0 * np.mean(np.argmax(predictions, axis=1) == targets)
            ens_acc = 100.0 * np.mean(np.argmax(predictions_sum, axis=1) == targets)

            predictions_sum_tensor = torch.tensor(predictions_sum)
            predictions_tensor = torch.tensor(predictions)
            targets_tensor = torch.tensor(targets)
            
            loss = criterion(predictions_tensor, targets_tensor)
            ens_loss = criterion(predictions_sum_tensor/(i+1), targets_tensor)
            print('Model accuracy: %8.4f. Ensemble accuracy: %8.4f' % (acc, ens_acc))
            print('Model loss: %8.4f. Ensemble loss: %8.4f' % (loss, ens_loss))
    return ens_acc, ens_loss

            
if args.method == 'empirical':
    #print('Warning: args.ckpt should be a directory')
    #eval_ecdf(model, loaders, criterion, args.ckpt, num_classes)
    raise NotImplementedError('Ecdf is not implemented yet')
elif args.method == 'dropout':
    raise NotImplementedError('Dropout is not implemented yet')
    #eval_dropout(model, loaders, criterion, args.ckpt[0], num_classes, args.samples)
elif args.method == 'KFACLaplace':
    raise NotImplementedError('KFAC Laplace is not implemented yet')
    #eval_laplace(model, loaders, criterion, args.ckpt[0], num_classes, args.samples)
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

    swag_model = SWAG(tiramisu.FCDenseNet67, no_cov_mat=False, max_num_models = 0, loading = True,
                    n_classes = 12)
    swag_model.cuda()
    checkpoint = torch.load(args.ckpt[0])
    swag_model.load_state_dict(checkpoint['state_dict'])

    test_err, test_acc = eval_swag(swag_model, loaders, criterion, args.samples, use_cov_mat, full_rank)


    