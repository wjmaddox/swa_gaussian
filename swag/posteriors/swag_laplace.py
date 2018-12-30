import argparse
import torch
import models, swag, data, utils, laplace
import torch.nn.functional as F
import numpy as np

parser = argparse.ArgumentParser(description='SGD/SWA training')
parser.add_argument('--file', type=str, default=None, required=True, help='checkpoint')

parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset name (default: CIFAR10)')
parser.add_argument('--data_path', type=str, default='/scratch/datasets/', metavar='PATH',
                    help='path to datasets location (default: None)')
parser.add_argument('--use_test', dest='use_test', action='store_true', help='use test dataset instead of validation (default: False)')
parser.add_argument('--split_classes', type=int, default=None)
parser.add_argument('--num_workers', type=int, default=4, metavar='N', help='number of workers (default: 4)')
parser.add_argument('--model', type=str, default='VGG16', metavar='MODEL',
                    help='model name (default: VGG16)')
parser.add_argument('--save_dir', type=str, default=None, required=True, help='path to npz results file')
parser.add_argument('--cov_mat', action='store_true', help='save sample covariance')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--loss', type=str, default='CE', help='loss to use for training model (default: Cross-entropy)')
parser.add_argument('--wd', type=float, default=1e-4, help='weight decay (default: 1e-4)')

args = parser.parse_args()

torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

print('Using model %s' % args.model)
model_cfg = getattr(models, args.model)

print('Split classes', args.split_classes)
print('Loading dataset %s from %s' % (args.dataset, args.data_path))
loaders, num_classes = data.loaders(
    args.dataset,
    args.data_path,
    1,
    args.num_workers,
    model_cfg.transform_train,
    model_cfg.transform_test,
    use_validation=not args.use_test,
    split_classes=args.split_classes
)

if args.cov_mat:
    args.no_cov_mat = False
else:
    args.no_cov_mat = True

print('Preparing models')
swag_model = swag.SWAG(model_cfg.base, no_cov_mat=args.no_cov_mat, max_num_models=20, loading=True, *model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
swag_model.cuda()
laplace_model = laplace.Laplace(model_cfg.base, *model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
laplace_model.cuda()

print('Loading model %s' % args.file)
checkpoint = torch.load(args.file)
swag_model.load_state_dict(checkpoint['state_dict'])

if args.loss == 'CE':
    criterion = F.cross_entropy
else:
    criterion = F.mse_loss

if not args.cov_mat:
    mean, var = swag_model.export_numpy_params()
    laplace_model.import_numpy_mean(mean)

else:
    mean, var, cov_mat_list = swag_model.export_numpy_params(export_cov_mat=True)
    laplace_model.import_numpy_mean(mean)

    laplace_model.import_numpy_cov_mat_sqrt(cov_mat_list)

laplace_model.import_numpy_mean(mean)

#variance estimation
print('Estimating variance')
#i used 1e-4 for weight decay when training the mlps
laplace_model.estimate_variance(loaders['train'], criterion, tau=args.wd)


if not args.cov_mat:
    print('Estimating Scale')
    scale = laplace_model.scale_grid_search(loaders['train'], criterion)
    print('Best scale found is: ', scale)

    model_file_name = 'laplace'
else:
    scale = 1.0
    model_file_name = 'swag-laplace'


utils.save_checkpoint(
    args.save_dir,
    checkpoint['epoch'],
    model_file_name,
    state_dict=laplace_model.state_dict(),
    scale=scale
)