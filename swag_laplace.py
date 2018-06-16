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


parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')



args = parser.parse_args()

torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

print('Using model %s' % args.model)
model_cfg = getattr(models, args.model)

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


print('Preparing models')
swag_model = swag.SWAG(model_cfg.base, *model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
swag_model.cuda()
laplace_model = laplace.Laplace(model_cfg.base, *model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
laplace_model.cuda()

print('Loading model %s' % args.file)
checkpoint = torch.load(args.file)
swag_model.load_state_dict(checkpoint['state_dict'])

mean, var = swag_model.export_numpy_params()
laplace_model.import_numpy_mean(mean)
print('Estimating variance')
laplace_model.estimate_variance(loaders['train'], F.cross_entropy)

utils.save_checkpoint(
    args.save_dir,
    checkpoint['epoch'],
    'laplace',
    state_dict=laplace_model.state_dict()
)