import argparse
import torch
import os
import models, swag, data, utils
import torch.nn.functional as F
import numpy as np
from itertools import chain

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
""" parser.add_argument('--save_freq', type=int, default=25, metavar='N', help='save frequency (default: 25)')
parser.add_argument('--eval_freq', type=int, default=5, metavar='N', help='evaluation frequency (default: 5)')
parser.add_argument('--lr_init', type=float, default=0.1, metavar='LR', help='initial learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
parser.add_argument('--wd', type=float, default=1e-4, help='weight decay (default: 1e-4)')
 """

parser.add_argument('--swa', action='store_true', help='swa usage flag (default: off)')
""" parser.add_argument('--swa_start', type=float, default=161, metavar='N', help='SWA start epoch number (default: 161)')
parser.add_argument('--swa_lr', type=float, default=0.05, metavar='LR', help='SWA LR (default: 0.05)')
parser.add_argument('--swa_c_epochs', type=int, default=1, metavar='N',
                    help='SWA model collection frequency/cycle length in epochs (default: 1)')

parser.add_argument('--swa_resume', type=str, default=None, metavar='CKPT',
                    help='checkpoint to restor SWA from (default: None)')
 """

parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

args = parser.parse_args()

#print('Preparing directory %s' % args.dir)
#os.makedirs(args.dir, exist_ok=True)
#with open(os.path.join(args.dir, 'command.sh'), 'w') as f:
#    f.write(' '.join(sys.argv))
#    f.write('\n')

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
    model_cfg.transform_test
)

print('Preparing model')
model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
model.cuda()

print('SWAG training')
swag_model = swag.SWAG(model_cfg.base, *model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
swag_model.cuda()

swag_model_location = args.dir + '/swag-' + str(args.epoch) + '.pt'
model_location = args.dir + '/checkpoint-' + str(args.epoch) + '.pt'
print('Loading model at ' + model_location + ' and swag_model at ' + swag_model_location)

swag_checkpoint = torch.load(swag_model_location)
swag_model.load_state_dict(swag_checkpoint['state_dict'])

model_checkpoint = torch.load(model_location)
model.load_state_dict(model_checkpoint['state_dict'])

#will be trying three different methods 
#sgd, swa, swag, swag + outerproduct (later)

criterion = F.cross_entropy

sgd_results, swa_results, swag_1sample_results, swag_3samples_results, swag_10samples_results = [],[],[],[], []

#these are not random...
#sgd predictions
sgd_results.append( utils.eval(loaders['test'], model, criterion) )

#swa predictions
#swag_model.collect_model(model)
swag_model.sample(0.0)
swa_results.append( utils.eval(loaders['test'], swag_model, criterion) )

for i in range(10):
    #swag predictions (1 sample)
    swag_1sample_results.append( utils.fast_ensembling(loaders['test'], swag_model, criterion, samples=1) )

    #swag-cov1 predictions (not implemented yet)
    """ swag_model.collect_model(model)
    swag_model.sample(1.0, block_cov=True) """

    #now the evaluation fast ensembling
    swag_3samples_results.append( utils.fast_ensembling(loaders['test'], swag_model, criterion, samples=3, scale=1.0) )
    swag_10samples_results.append( utils.fast_ensembling(loaders['test'], swag_model, criterion, samples=10, scale=1.0) )

def compute_mean_var(results_dict_list):
    accuracy = [i['accuracy'] for i in results_dict_list]
    mean_accuracy = np.mean(accuracy)
    sd_accuracy = np.std(accuracy)
    return mean_accuracy, sd_accuracy

sgd_mean = compute_mean_var(sgd_results)
swa_mean = compute_mean_var(swa_results)
swag_1sample_mean = compute_mean_var(swag_1sample_results)
swag_3sample_mean = compute_mean_var(swag_3samples_results)
swag_10sample_mean = compute_mean_var(swag_10samples_results)

print('SGD: ', sgd_mean[0], sgd_mean[1])
print('SWA: ', swa_mean[0], swa_mean[1])
print('SWAG, 1 sample: ', swag_1sample_mean[0], swag_1sample_mean[1])
print('SWAG, 3 samples: ', swag_3sample_mean[0], swag_3sample_mean[1])
print('SWAG, 10 samples: ', swag_10sample_mean[0], swag_10sample_mean[1])

print('Now running accuracy over range')

accuracy = []
accuracy_sd = []
ivec = []
for i in chain(range(1, 10), range(10, 100, 10)):
    print('Using ', i, ' samples')
    ivec.append(i)

    current_accuracy = []
    #replicate everything 10x
    for _ in range(args.replications):
        out = utils.fast_ensembling(loaders['test'], swag_model, criterion, samples=i, scale=1.0)
        current_accuracy.append(out)
    mean, sd = compute_mean_var(current_accuracy)

    accuracy.append(mean)
    accuracy_sd.append(sd)

import matplotlib.pyplot as plt
plt.errorbar(ivec, accuracy, yerr=accuracy_sd)
plt.axhline(swa_mean[0], lw=4, color='r')
plt.axhline(sgd_mean[0], lw=4, color='g')
plt.xlabel('Samples in Ensemble')
plt.ylabel('Accuracy')
plt.savefig(args.dataset + '_ensemble_accuracy.png')
