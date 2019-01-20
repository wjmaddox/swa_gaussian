import argparse
import torch
import torch.nn.functional as F
import numpy as np

from swag import data, models, utils
from swag.posteriors import SWAG

parser = argparse.ArgumentParser(description='SGD/SWA training')
parser.add_argument('--file', type=str, default=None, required=True, help='checkpoint')

parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset name (default: CIFAR10)')
parser.add_argument('--data_path', type=str, default='/scratch/datasets/', metavar='PATH',
                    help='path to datasets location (default: None)')
parser.add_argument('--use_test', dest='use_test', action='store_true', help='use test dataset instead of validation (default: False)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='input batch size (default: 128)')
parser.add_argument('--num_workers', type=int, default=4, metavar='N', help='number of workers (default: 4)')
parser.add_argument('--model', type=str, default='VGG16', metavar='MODEL',
                    help='model name (default: VGG16)')


parser.add_argument('--N', type=int, default=700, metavar='N', help='number of weights to try')
parser.add_argument('--delta', type=float, default=0.03, metavar='DELTA', help='weights delta (default: 0.03)')
parser.add_argument('--save_path', type=str, default=None, required=True, help='path to npz results file')


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
    args.batch_size,
    args.num_workers,
    model_cfg.transform_train,
    model_cfg.transform_test,
    use_validation=not args.use_test
)


print('Preparing SWAG model')
swag_model = SWAG(model_cfg.base, *model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
swag_model.cuda()

print('Loading model %s' % args.file)
checkpoint = torch.load(args.file)
swag_model.load_state_dict(checkpoint['state_dict'])

mean, var = swag_model.export_numpy_params()
w = mean.copy()

ord = np.argsort(var)

criterion = F.cross_entropy

te_acc = np.zeros((args.N, 2))
te_nll = np.zeros((args.N, 2))

K = ord.size // (args.N - 1)
ind = ord[list(range(0, ord.size - ord.size % K, K)) + [ord.size - 1]]

w_mean = mean[ind]
w_std = np.sqrt(var[ind])

for i, w_id in enumerate(ind):
    print('%d/%d. Mean: %f Std: %f' % (i + 1, args.N, w_mean[i], w_std[i]))
    val = w[w_id]

    w[w_id] = val - args.delta
    swag_model.import_numpy_weights(w)
    res = utils.eval(loaders['test'], swag_model, criterion)
    te_acc[i, 0], te_nll[i, 0] = res['accuracy'], res['loss']
    print('w = %f: acc = %.4f nll = %.4f' % (w[w_id], te_acc[i, 0], te_nll[i, 0]))

    w[w_id] = val + args.delta
    swag_model.import_numpy_weights(w)
    res = utils.eval(loaders['test'], swag_model, criterion)
    te_acc[i, 1], te_nll[i, 1] = res['accuracy'], res['loss']
    print('w = %f: acc = %.4f nll = %.4f' % (w[w_id], te_acc[i, 1], te_nll[i, 1]))
    print()
    w[w_id] = val

np.savez(
    args.save_path,
    w_mean=w_mean,
    w_std=w_std,
    te_acc=te_acc,
    te_nll=te_nll,
    delta=args.delta
)