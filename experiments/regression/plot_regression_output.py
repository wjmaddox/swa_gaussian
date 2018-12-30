r"""
This will eventually incorporate the Laplace approximations and use variances as well, but not yet...
"""
import argparse
import torch
import os
import models, swag, data, utils, laplace
import torch.nn.functional as F
import numpy as np
from itertools import chain, product
import tabulate
import itertools
import matplotlib.pyplot as plt
from torch.distributions.normal import Normal 

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
laplace_model_location = args.dir + '/swag-laplace-' + str(args.epoch) + '.pt'
print('Loading sgd model at ' + model_location + ' \n swag_model at ' + swag_model_location + '\n laplace model at ' + laplace_model_location)


print('Preparing model')
model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
model.cuda()

print('SWAG model loading')
swag_checkpoint = torch.load(swag_model_location)
swag_model = swag.SWAG(model_cfg.base, no_cov_mat=args.no_cov_mat, max_num_models = 20, loading = True, *model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
swag_model.cuda()
swag_model.load_state_dict(swag_checkpoint['state_dict'])

print('SGD model loading')
model_checkpoint = torch.load(model_location)
model.load_state_dict(model_checkpoint['state_dict'])

print('laplace model loading')
laplace_checkpoint = torch.load(laplace_model_location)
laplace_model = laplace.Laplace(model_cfg.base, *model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
laplace_model.cuda()
laplace_model.load_state_dict(laplace_checkpoint['state_dict'])

x = loaders['test'].dataset.test_data.cuda()
y = loaders['test'].dataset.test_labels.detach().cpu().numpy()

#x, y, pred = compute_predictions(loaders['test'], model)

def plot_regression_uncertainty(model, cov, output_file, scale=1.0, nsamples=1000):
    input = torch.arange(-6, 6, 0.1).cuda().view(-1,1)
    input_numpy = input.detach().cpu().numpy()
    
    #perform a 1d grid search over optimal scaling of laplace
    if type(model).__name__ == 'Laplace' and scale==None:
        logscale_range = torch.arange(-10, 0, 0.1).cuda()

        all_losses = torch.zeros_like(logscale_range)

        #generate validation dataset
        torch.manual_seed(3)
        xval = (8.0 * torch.rand(20,1) - 4.0).cuda() #[-4,4]
        yval = xval ** 3.0 + 3.0 * torch.randn(20,1).cuda()

        for i, logscale in enumerate(logscale_range):
            current_scale = torch.exp(logscale)
            all_output = torch.zeros(0, xval.size(1)).cuda()
            for _ in range(int(nsamples/20.)):
                model.sample(cov=cov, scale=current_scale)
                output = model(yval)
                all_output = torch.cat((all_output, output.t()), dim= 0)
            all_losses[i] = -Normal(yval, current_scale).log_prob(all_output.t()).mean()
            #all_losses[i] = (-0.5 * (all_output.t() - yval).pow(2)/current_scale - 0.5 * logscale).mean()

        print(all_losses)

        min_index = torch.min(all_losses,dim=0)[1]
        scale = torch.exp(logscale_range[min_index]).item()
        print('best scale is', scale)

    all_output = torch.zeros(0, input.size(1)).cuda()
    for i in range(nsamples):
        model.sample(cov=cov, scale=scale)
        output = model(input)
        all_output = torch.cat((all_output, output.t()), dim= 0)

    swag_means = torch.mean(all_output, dim = 0).view(-1).detach().cpu().numpy()
    swag_std = torch.std(all_output, dim = 0).view(-1).detach().cpu().numpy()

    for alpha, i in zip(range(2, 5), range(3, 0, -1)):
        plt.fill_between(x=np.squeeze(input_numpy), y1=np.squeeze(swag_means - i * swag_std), y2=np.squeeze(swag_means + i * swag_std), color='blue', alpha=alpha/20)
        #plt.plot(input_numpy, swag_means - i * swag_std, c='blue', alpha=alpha)
        #plt.plot(input_numpy, swag_means + i * swag_std, c='blue', alpha=alpha)
    
    #this is swa predictions
    swag_model.sample(0.0)
    output = swag_model(input).detach().cpu().numpy()
    plt.plot(input_numpy, output, c='red')

    #this is E(y^* | y)
    plt.plot(input_numpy, swag_means, c='blue')

    #true regression curve
    plt.plot(input_numpy, input_numpy ** 3, c='black')

    #these are the testing data points
    plt.scatter(x, y, c='black') #these are the true points
    plt.ylim(((-200, 200)))

    plt.savefig(output_file, format='png', dpi=1200)
    plt.close()
    #return input, input_numpy

#this is sgd
#sgd_output = model(input).detach().cpu().numpy()
#plt.plot(input_numpy, sgd_output, c='red')
#sgd_output = model(x).detach().cpu().numpy()
#plt.plot()

#swag - no covariance
plot_regression_uncertainty(swag_model, cov=False, output_file='plots/toyreg_swag_nocov.png')

#swag - covariance
plot_regression_uncertainty(swag_model, cov=True, output_file='plots/toyreg_swag_cov.png')

#laplace
plot_regression_uncertainty(laplace_model, scale=None, cov=False, output_file='plots/toyreg_laplace_nocov.png')

#swag - hessian
plot_regression_uncertainty(laplace_model, scale=1.0, cov=True, output_file='plots/toyreg_slaplace.png')

plot_regression_uncertainty(laplace_model, scale=None, cov=True, output_file='plots/toyreg_slaplace_tuned.png')
