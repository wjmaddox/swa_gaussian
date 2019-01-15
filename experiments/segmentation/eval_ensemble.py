import argparse
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import tqdm

from pathlib import Path

from models import tiramisu
from datasets import camvid
from datasets import joint_transforms
from utils.training import test

from swag import data, losses, models, utils
from swag.posteriors import SWAG, KFACLaplace

parser = argparse.ArgumentParser(description='SGD/SWA training')
parser.add_argument('--file', type=str, default=None, required=True, help='checkpoint')

parser.add_argument('--data_path', type=str, default='/home/wesley/Documents/Code/SegNet-Tutorial/CamVid/', metavar='PATH',
                    help='path to datasets location (default: None)')
parser.add_argument('--use_test', dest='use_test', action='store_true', help='use test dataset instead of validation (default: False)')
parser.add_argument('--batch_size', type=int, default=5, metavar='N', help='input batch size (default: 5)')
parser.add_argument('--split_classes', type=int, default=None)
parser.add_argument('--num_workers', type=int, default=4, metavar='N', help='number of workers (default: 4)')
parser.add_argument('--method', type=str, default='SWAG', choices=['SWAG', 'KFACLaplace', 'SGD', 'HomoNoise', 'Dropout', 'SWAGDrop'], required=True)
parser.add_argument('--save_path', type=str, default=None, required=True, help='path to npz results file')
parser.add_argument('--N', type=int, default=30)
parser.add_argument('--scale', type=float, default=1.0)
parser.add_argument('--cov_mat', action='store_true', help = 'use sample covariance for swag')
parser.add_argument('--use_diag', action='store_true', help = 'use diag cov for swag')

parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

args = parser.parse_args()

eps = 1e-12
if args.cov_mat:
    args.cov_mat = True
else:
    args.cov_mat = False

torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

CAMVID_PATH = Path(args.data_path)
normalize = transforms.Normalize(mean=camvid.mean, std=camvid.std)
# fine-tuning does not include random crops
train_joint_transformer = transforms.Compose([
    joint_transforms.JointRandomHorizontalFlip()
    ])
train_dset = camvid.CamVid(CAMVID_PATH, 'train',
      joint_transform=train_joint_transformer,
      transform=transforms.Compose([
          transforms.ToTensor(),
          normalize,
    ]))
train_loader = torch.utils.data.DataLoader(
    train_dset, batch_size=3, shuffle=True)

if args.use_test:
    print('Warning: using test dataset')
    test_dset = camvid.CamVid(
        CAMVID_PATH, 'test', joint_transform=None,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize
        ]))
else:
    print('Using validation dataset')
    test_dset = camvid.CamVid(
        CAMVID_PATH, 'val', joint_transform=None,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize
        ]))

test_loader = torch.utils.data.DataLoader(
    test_dset, batch_size=args.batch_size, shuffle=False)
loaders = {'train': train_loader, 'test': test_loader}



print('Preparing model')
if args.method in ['SWAG', 'HomoNoise', 'SWAGDrop']:
    model = SWAG(tiramisu.FCDenseNet67, no_cov_mat=not args.cov_mat, max_num_models = 0, loading = True, n_classes=11)
elif args.method in ['SGD', 'Dropout', 'KFACLaplace']:
    # construct and load model
    model = tiramisu.FCDenseNet67(n_classes=11)
else:
    assert False
model.cuda()

def train_dropout(m):
    if m.__module__ == torch.nn.modules.dropout.__name__:
        #print('here')
        m.train()

print('Loading model %s' % args.file)
checkpoint = torch.load(args.file)
model.load_state_dict(checkpoint['state_dict'])

if args.method == 'KFACLaplace':
    print(len(loaders['train'].dataset))
    model = KFACLaplace(model, eps = 5e-4, data_size = len(loaders['train'].dataset)) #eps: weight_decay

    t_input, t_target = next(iter(loaders['train']))
    t_input, t_target = t_input.cuda(non_blocking = True), t_target.cuda(non_blocking = True)

if args.method == 'HomoNoise':
    std = 0.01
    for module, name in model.params:
        mean = module.__getattr__('%s_mean' % name)
        module.__getattr__('%s_sq_mean' % name).copy_(mean**2 + std**2)
                            
predictions = np.zeros((len(test_dset), 11, 360, 480))
targets = np.zeros((len(test_dset), 360, 480))
print(targets.size)

for i in range(args.N):
    print('%d/%d' % (i + 1, args.N))
    if args.method == 'KFACLaplace':
        ## KFAC Laplace needs one forwards pass to load the KFAC model at the beginning
        model.net.load_state_dict(model.mean_state)

        if i==0:
            model.net.train()

            loss, _ = losses.cross_entropy(model.net, t_input, t_target)
            loss.backward(create_graph = True)
            model.step(update_params = False)

    if args.method not in ['SGD', 'Dropout']:
        sample_with_cov = args.cov_mat and not args.use_diag
        model.sample(scale=args.scale, cov=sample_with_cov)

    if 'SWAG' in args.method:
        utils.bn_update(loaders['train'], model)
        
    model.eval()
    if args.method in ['Dropout', 'SWAGDrop']:
        model.apply(train_dropout)

    k = 0
    for input, target in tqdm.tqdm(loaders['test']):
        input = input.cuda(non_blocking=True)
        torch.manual_seed(i)

        if args.method == 'KFACLaplace':
            output = model.net(input)
        else:
            output = model(input)

        with torch.no_grad():
            batch_probs = F.softmax(output, dim=1).cpu().numpy()
            #print( (np.argmax(batch_probs, axis=1) == np.argmax(predictions[k:k+input.size(0),:, :, :], axis=1)).mean() )
            predictions[k:k+input.size(0),:, :, :] += batch_probs
        targets[k:(k+target.size(0)), :, :] = target.numpy()
        k += input.size(0)

    print(np.mean(np.argmax(predictions, axis=1) == targets))
predictions /= args.N

entropies = -np.sum(np.log(predictions + eps) * predictions, axis=1)
np.savez(args.save_path, entropies=entropies, predictions=predictions, targets=targets)






