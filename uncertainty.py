import argparse
import torch
import models, swag, data, utils, laplace
import torch.nn.functional as F
import numpy as np
import tqdm

parser = argparse.ArgumentParser(description='SGD/SWA training')
parser.add_argument('--file', type=str, default=None, required=True, help='checkpoint')

parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset name (default: CIFAR10)')
parser.add_argument('--data_path', type=str, default='/scratch/datasets/', metavar='PATH',
                    help='path to datasets location (default: None)')
parser.add_argument('--use_test', dest='use_test', action='store_true', help='use test dataset instead of validation (default: False)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='input batch size (default: 128)')
parser.add_argument('--split_classes', type=int, default=0, required=True)
parser.add_argument('--num_workers', type=int, default=4, metavar='N', help='number of workers (default: 4)')
parser.add_argument('--model', type=str, default='VGG16', metavar='MODEL',
                    help='model name (default: VGG16)')
parser.add_argument('--method', type=str, default='SWAG', choices=['SWAG', 'Laplace'], required=True)
parser.add_argument('--save_path', type=str, default=None, required=True, help='path to npz results file')
parser.add_argument('--N', type=int, default=20)
parser.add_argument('--scale', type=float, default=1.0)

parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

args = parser.parse_args()

eps = 1e-12

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
    use_validation=not args.use_test,
    split_classes=args.split_classes,
    shuffle_train=False
)


print('Preparing model')
if args.method == 'SWAG':
    model = swag.SWAG(model_cfg.base, *model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
elif args.method == 'Laplace':
    model = laplace.Laplace(model_cfg.base, *model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
else:
    assert False
model.cuda()

print('Loading model %s' % args.file)
checkpoint = torch.load(args.file)
model.load_state_dict(checkpoint['state_dict'])

predictions = np.zeros((len(loaders['train'].dataset), num_classes))
for i in range(args.N):
    print('%d/%d' % (i + 1, args.N))
    model.train()
    model.sample(scale=args.scale)
    k = 0
    for input, target in tqdm.tqdm(loaders['train']):
        input = input.cuda(async=True)
        output = model(input)

        predictions[k:k+input.size()[0]] += F.softmax(output, dim=1).cpu().numpy()
        k += input.size()[0]
predictions /= args.N

entropies = -np.sum(np.log(predictions + eps) * predictions, axis=1)
np.savez(args.save_path, entropies=entropies)






