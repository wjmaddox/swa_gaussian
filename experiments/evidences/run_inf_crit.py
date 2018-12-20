import argparse
import torch
import os

import torch.nn.functional as F
import numpy as np
from itertools import chain, product
import tabulate
import itertools

from swag import data, models, utils
from swag.posteriors import evidences, SWAG


#from load_different_state_dict import load_different_state_dict, _load_from_different_state_dict

parser = argparse.ArgumentParser(description='SGD/SWA/SWAG ensembling')
parser.add_argument('--replications', type=int, default=10, help='number of passes through testing set')
parser.add_argument('--dir', type=str, default=None, required=True, help='training directory (default: None)')

parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset name (default: CIFAR10)')
parser.add_argument('--data_path', type=str, default='/home/wesley/Documents/datasets/', metavar='PATH',
                    help='path to datasets location (default: None)')
parser.add_argument('--use_test', dest='use_test', action='store_true', help='use test dataset instead of validation (default: False)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='input batch size (default: 128)')
parser.add_argument('--num_workers', type=int, default=4, metavar='N', help='number of workers (default: 4)')
parser.add_argument('--model', type=str, default='VGG16', metavar='MODEL',
                    help='model name (default: VGG16)')

parser.add_argument('--epoch', type=int, default=200, metavar='N', help='epoch to evaluate at (default: 200)')

parser.add_argument('--swa', action='store_true', help='swa usage flag (default: off)')
parser.add_argument('--cov_mat', action='store_true', help='save sample covariance')

#parser.add_argument('--plot', action='store_true', help='plot replications (default: off)')
parser.add_argument('--samples', type=int, default=25, help='number of approximate posterior samples to draw')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--save_path', type=str, required=True, help='output file to save to')
parser.add_argument('--num_models', type=int, default=20, help='number of swa covariance models in your checkpoint (default: 20)')
parser.add_argument('--resume', type=str, required=False, help='preload a forwards pass dict')
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
    use_validation=False,
    split_classes=None
)

swag_model_location = args.dir + '/swag-' + str(args.epoch) + '.pt'
model_location = args.dir + '/checkpoint-' + str(args.epoch) + '.pt'
print('Loading sgd model at ' + model_location + ' and swag_model at ' + swag_model_location)

print('Preparing model')
model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
model.cuda()
model_checkpoint = torch.load(model_location)
model.load_state_dict(model_checkpoint['state_dict'])
del model_checkpoint

print('SWAG training')
swag_model = SWAG(model_cfg.base, no_cov_mat=args.no_cov_mat, max_num_models = args.num_models, loading = True, *model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
swag_model.cuda()
swag_checkpoint = torch.load(swag_model_location)
swag_model.load_state_dict(swag_checkpoint['state_dict'], strict = False)

#will be trying four different methods 
#sgd, swa, swag, swag + outerproduct (later)

criterion = F.cross_entropy

#sgd_results, swa_results, swag_1sample_results, swag_3samples_results, swag_10samples_results = [],[],[],[], []
columns = ['model', 'samples', 'cov', 'acc', 'acc_sd', 'te_loss', 'te_loss_sd']

results = []
#these are not random...
#sgd predictions
sgd_results = utils.eval(loaders['test'], model, criterion) 
values = ['sgd', 0, False, sgd_results['accuracy'], 0, sgd_results['loss'], 0]

table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='8.4f')
print(table)
results.append(values)

#swa predictions
#swag_model.collect_model(model)
swag_model.sample(0.0)
utils.bn_update(loaders['train'], swag_model)
swa_results = utils.eval(loaders['test'], swag_model, criterion)

values = ['swa', 0, False, swa_results['accuracy'], 0, swa_results['loss'], 0]

table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='8.4f')
table = table.split('\n')[2]
print(table)
results.append(values)

#compute numparameters
numparams = evidences.compute_numparams(swag_model)


if args.resume is None:
    #compute log probabilities
    log_prob_results_dict = evidences.compute_epoch_logprobs(loaders, swag_model, samples = args.samples, use_test = args.use_test, cov=True, block=False)
    log_swa_results_dict = evidences.compute_epoch_logprobs(loaders, swag_model, samples = 1, scale = 0.0, use_test = args.use_test, cov=False, block=False)

    if args.save_path is not None:
        torch.save([log_prob_results_dict, log_swa_results_dict], f=args.save_path)
else:
    log_prob_results_dict, log_swa_results_dict = torch.load(args.resume)

dic = evidences.compute_dic1(log_prob_results_dict['log_ll'], log_swa_results_dict['log_ll'], num_samples = args.samples)
waic = evidences.compute_waic(log_prob_results_dict['full_log_ll'], num_samples = args.samples)

print('DIC: ', dic)
print('WAIC: ', waic)
    



    