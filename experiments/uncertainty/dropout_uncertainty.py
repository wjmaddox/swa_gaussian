import argparse
import torch
import torch.nn.functional as F
import numpy as np
import tqdm

from swag import data, utils, models
#from swag.posteriors import Laplace, SWAG

parser = argparse.ArgumentParser(description='SGD/SWA training')
parser.add_argument('--file', type=str, default=None, required=True, help='checkpoint')

parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset name (default: CIFAR10)')
parser.add_argument('--data_path', type=str, default='/scratch/datasets/', metavar='PATH',
                    help='path to datasets location (default: None)')
parser.add_argument('--use_test', dest='use_test', action='store_true', help='use test dataset instead of validation (default: False)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='input batch size (default: 128)')
parser.add_argument('--split_classes', type=int, default=None)
parser.add_argument('--num_workers', type=int, default=4, metavar='N', help='number of workers (default: 4)')
parser.add_argument('--model', type=str, default='VGG16', metavar='MODEL',
                    help='model name (default: VGG16)')
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
model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
model.cuda()

print('Loading model %s' % args.file)
checkpoint = torch.load(args.file)
model.load_state_dict(checkpoint['state_dict'])

predictions = np.zeros((len(loaders['test'].dataset), num_classes))
targets = np.zeros(len(loaders['test'].dataset))
print(targets.size)
with torch.no_grad():
    for i in range(args.N):
        print('%d/%d' % (i + 1, args.N))

        model.eval()

        #function that sets dropout to train mode
        def train_dropout(m):
            if type(m)==torch.nn.modules.dropout.Dropout:
                m.train()
        model.apply(train_dropout)

        k = 0
        for input, target in tqdm.tqdm(loaders['test']):
            input = input.cuda(non_blocking=True)
            output = model(input)

            predictions[k:k+input.size()[0]] += F.softmax(output, dim=1).cpu().numpy()
            targets[k:(k+target.size(0))] = target.numpy()
            k += input.size()[0]
            
    predictions /= args.N

entropies = -np.sum(np.log(predictions + eps) * predictions, axis=1)
np.savez(args.save_path, entropies=entropies, predictions=predictions, targets=targets)