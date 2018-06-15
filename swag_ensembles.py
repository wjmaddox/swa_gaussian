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
        model_cfg.transform_test
    )

#torch.nn.Module.load_different_state_dict = load_different_state_dict
#torch.nn.Module._load_from_different_state_dict = _load_from_different_state_dict

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
print(torch.cuda.memory_allocated()/(1024.0 ** 3))

print(torch.cuda.memory_allocated()/(1024.0 ** 3))
model_checkpoint = torch.load(model_location)
model.load_state_dict(model_checkpoint['state_dict'])
del model_checkpoint

#will be trying four different methods 
#sgd, swa, swag, swag + outerproduct (later)

criterion = F.cross_entropy

#sgd_results, swa_results, swag_1sample_results, swag_3samples_results, swag_10samples_results = [],[],[],[], []
columns = ['model', 'samples', 'cov', 'acc', 'acc_sd', 'te_loss', 'te_loss_sd']

#these are not random...
#sgd predictions
sgd_results = utils.eval(loaders['test'], model, criterion) 
values = ['sgd', 0, False, sgd_results['accuracy'], 0, sgd_results['loss'], 0]

table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='8.4f')
#table = table.split('\n')[2]

print(table)

#swa predictions
#swag_model.collect_model(model)
swag_model.sample(0.0)
swa_results = utils.eval(loaders['test'], swag_model, criterion)

values = ['swa', 0, False, swa_results['accuracy'], 0, swa_results['loss'], 0]

table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='8.4f')
table = table.split('\n')[2]

print(table)

def run_ensembles(samples, cov):
    return utils.fast_ensembling(loaders['test'], swag_model, criterion, samples=samples, cov=cov)

samples_list = [1, 3, 10]
if args.no_cov_mat is True:
    cov_list = [False]
else:
    cov_list = [True, False]

swag_replications = []
for i in range(args.replications):
    swag_current_list = []

    for (sample, cov) in product(samples_list, cov_list):
        swag_current_list.append([sample, cov, run_ensembles(sample, cov)])
    
    swag_replications.append(swag_current_list)

for i, (sample, cov) in enumerate(product(samples_list, cov_list)):
    matched_list = [j[i] for j in swag_replications]

    loss = [j[2]['loss'] for j in matched_list]
    accuracy = [j[2]['accuracy'] for j in matched_list]

    mean_accuracy = np.mean(accuracy)
    sd_accuracy = np.std(accuracy)

    mean_loss = np.mean(loss)
    sd_loss = np.std(loss)

    #warning in case sample, cov don't match
    if sample != matched_list[0][0]:
        print('warning sample does not match list setup')
    if cov != matched_list[0][1]:
        print('warning cov does not match list setup')
    
    values = ['swa', sample, cov, mean_accuracy, sd_accuracy, mean_loss, sd_loss]

    table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='8.4f')
    table = table = table.split('\n')[2]

    print(table)
    


    
def compute_mean_var(results_dict_list):
    accuracy = [i['accuracy'] for i in results_dict_list]
    loss = [i['loss'] for i in results_dict_list]
    mean_accuracy = np.mean(accuracy)
    sd_accuracy = np.std(accuracy)

    mean_loss = np.mean(loss)
    sd_loss = np.std(loss)
    return mean_accuracy, sd_accuracy, mean_loss, sd_loss



if args.plot:
    print('Now generating accuracy plot')

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
