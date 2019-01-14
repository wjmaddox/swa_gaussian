import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
import sklearn.decomposition

os.sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../'))

from swag import data, models, utils, losses
from swag.posteriors import SWAG

parser = argparse.ArgumentParser(description='Width along PCA directions')

parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset name (default: CIFAR10)')
parser.add_argument('--data_path', type=str, default='/scratch/datasets/', metavar='PATH',
                    help='path to datasets location (default: None)')
parser.add_argument('--use_test', dest='use_test', action='store_true', help='use test dataset instead of validation (default: False)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='input batch size (default: 128)')
parser.add_argument('--num_workers', type=int, default=4, metavar='N', help='number of workers (default: 4)')
parser.add_argument('--model', type=str, default='VGG16', metavar='MODEL',
                    help='model name (default: VGG16)')

parser.add_argument('--checkpoint', action='append')
parser.add_argument('--save_path', type=str, default=None, required=True, help='path to npz results file')

parser.add_argument('--dist', type=float, default=30.0, metavar='D', help='dist to travel along a direction (default: 30.0)')
parser.add_argument('--N', type=int, default=31, metavar='N', help='number of points on a grid (default: 31)')

parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

args = parser.parse_args()

torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)

args.device = None
if torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')

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

print('Preparing model')
model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
model.to(args.device)

criterion = losses.cross_entropy

W = []
num_checkpoints = len(args.checkpoint)
for path in args.checkpoint:
    print('Loading %s' % path)
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['state_dict'])
    W.append(np.concatenate([p.detach().cpu().numpy().ravel() for p in model.parameters()]))
W = np.array(W)
print('Shape: %d %d' % (W.shape[0], W.shape[1]))

pca = sklearn.decomposition.PCA(n_components=W.shape[0])
pca.fit(W)
print(pca.explained_variance_ratio_ * 100.0)


pc_idx = [0, 1, num_checkpoints - 2, num_checkpoints - 1] if num_checkpoints > 3 else list(range(num_checkpoints))
K = len(pc_idx)

ts = np.linspace(-args.dist, args.dist, args.N)


train_acc = np.zeros((K, args.N))
train_loss = np.zeros((K, args.N))
test_acc = np.zeros((K, args.N))
test_loss = np.zeros((K, args.N))


for i in pc_idx:
    print('PC %d. Variance ratio: %.2f%%' % (i, pca.explained_variance_ratio_[i] * 100.0))
    mean = pca.mean_
    v = pca.components_[i, :].copy()
    v /= np.linalg.norm(v)
    for j, t in enumerate(ts):
        print('t: %.2f' % t)
        w = mean + t * v

        offset = 0
        for param in model.parameters():
            size = np.prod(param.size())
            param.data.copy_(param.new_tensor(w[offset:offset+size].reshape(param.size())))
            offset += size

        print('BN')
        utils.bn_update(loaders['train'], model)
        print('Train')
        train_res = utils.eval(loaders['train'], model, criterion)
        print(train_res)
        print('Test')
        test_res = utils.eval(loaders['test'], model, criterion)
        print(test_res)

        train_acc[i, j] = train_res['accuracy']
        train_loss[i, j] = train_res['loss']
        test_acc[i, j] = train_res['accuracy']
        test_loss[i, j] = test_res['loss']

np.savez(
    args.save_path,
    N=num_checkpoints,
    dim=W.shape[1],
    explained_variance_ratio=pca.explained_variance_ratio_,
    pc_idx=pc_idx,
    train_acc=train_acc,
    train_err=100.0 - train_acc,
    train_loss=train_loss,
    test_acc=test_acc,
    test_err=100.0 - test_acc,
    test_loss=test_loss
)

