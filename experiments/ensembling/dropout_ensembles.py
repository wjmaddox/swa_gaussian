import argparse
import os, sys
import tabulate
import torch
import numpy as np

import torch.nn.functional as F
from itertools import chain, product

from swag import models, data, utils, losses
from swag.posteriors.utils import eval_dropout

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

parser.add_argument('--plot', action='store_true', help='plot replications (default: off)')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--save_path', type=str, required=True, help='output file to save to')
parser.add_argument('--num_models', type=int, default=20, help='number of swa covariance models in your checkpoint (default: 20)')
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
    test_data_loader = torch.utils.data.DataLoader(dataset, batch_size = args.batch_size, num_workers = args.num_workers)

    loaders, num_classes = data.loaders(
        args.dataset[:-2],
        args.data_path,
        args.batch_size,
        args.num_workers,
        model_cfg.transform_train,
        model_cfg.transform_test,
        use_validation=not args.use_test,
        split_classes=None
    )
    loaders['test'] = test_data_loader
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

model_location = args.dir + '/checkpoint-' + str(args.epoch) + '.pt'
print('Loading sgd model at ' + model_location)

print('Preparing model')
model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
model.cuda()

model_checkpoint = torch.load(model_location)
model.load_state_dict(model_checkpoint['state_dict'])
del model_checkpoint

#criterion = losses.cross_entropy

#sgd_results, swa_results, swag_1sample_results, swag_3samples_results, swag_10samples_results = [],[],[],[], []
columns = ['model', 'samples', 'acc', 'acc_sd', 'te_loss', 'te_loss_sd']

results = []
#these are not random...
#sgd predictions
sgd_results = utils.eval(loaders['test'], model, losses.cross_entropy) 
values = ['sgd', 0, sgd_results['accuracy'], 0, sgd_results['loss'], 0]

table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='8.4f')
print(table)
results.append(values)

# now dropout predictions
table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='8.4f')
table = table.split('\n')[2]
print(table)
results.append(values)

def run_ensembles(samples, reps=args.replications):
    current_list = []
    for _ in range(reps):
        current_list.append([samples, eval_dropout(loaders, model, F.cross_entropy, samples=samples)])
    return current_list

samples_list = [1, 3, 10, 20, 30]

for i, sample in enumerate(samples_list):
    drop_current_list = run_ensembles(sample)

    loss = [j[1]['loss'] for j in drop_current_list]
    accuracy = [j[1]['accuracy'] for j in drop_current_list]

    mean_accuracy = np.mean(accuracy)
    sd_accuracy = np.std(accuracy)

    mean_loss = np.mean(loss)
    sd_loss = np.std(loss)
    
    values = ['dropout', sample, mean_accuracy, sd_accuracy, mean_loss, sd_loss]

    table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='8.4f')
    table = table.split('\n')[2]

    print(table)
    results.append(values)

#finally save all results in numpy file
np.savez(args.save_path, result=results)