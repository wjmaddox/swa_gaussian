import argparse
import torch
import os
import models, swag, data, utils
import torch.nn.functional as F

parser = argparse.ArgumentParser(description='SGD/SWA training')
parser.add_argument('--file', type=str, default=None, required=True, help='training directory (default: None)')

parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset name (default: CIFAR10)')
parser.add_argument('--data_path', type=str, default='/scratch/datasets/', metavar='PATH',
                    help='path to datasets location (default: None)')
parser.add_argument('--use_test', dest='use_test', action='store_true', help='use test dataset instead of validation (default: False)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='input batch size (default: 128)')
parser.add_argument('--num_workers', type=int, default=4, metavar='N', help='number of workers (default: 4)')
parser.add_argument('--model', type=str, default='VGG16', metavar='MODEL',
                    help='model name (default: VGG16)')

parser.add_argument('--resume', type=str, default=None, metavar='CKPT',
                    help='checkpoint to resume training from (default: None)')

parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of epochs to train (default: 200)')
parser.add_argument('--save_freq', type=int, default=25, metavar='N', help='save frequency (default: 25)')
parser.add_argument('--eval_freq', type=int, default=5, metavar='N', help='evaluation frequency (default: 5)')
parser.add_argument('--lr_init', type=float, default=0.1, metavar='LR', help='initial learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
parser.add_argument('--wd', type=float, default=1e-4, help='weight decay (default: 1e-4)')


parser.add_argument('--swa', action='store_true', help='swa usage flag (default: off)')
parser.add_argument('--swa_start', type=float, default=161, metavar='N', help='SWA start epoch number (default: 161)')
parser.add_argument('--swa_lr', type=float, default=0.05, metavar='LR', help='SWA LR (default: 0.05)')
parser.add_argument('--swa_c_epochs', type=int, default=1, metavar='N',
                    help='SWA model collection frequency/cycle length in epochs (default: 1)')

parser.add_argument('--swa_resume', type=str, default=None, metavar='CKPT',
                    help='checkpoint to restor SWA from (default: None)')


parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

args = parser.parse_args()

#print('Preparing directory %s' % args.dir)
#os.makedirs(args.dir, exist_ok=True)
#with open(os.path.join(args.dir, 'command.sh'), 'w') as f:
#    f.write(' '.join(sys.argv))
#    f.write('\n')

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
    model_cfg.transform_test
)

print('Preparing model')
model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
model.cuda()

print('SWAG training')
swag_model = swag.SWAG(model_cfg.base, *model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
swag_model.cuda()

print('Loading model')
checkpoint = torch.load(args.file)
swag_model.load_state_dict(checkpoint['state_dict'])

#will be trying three different methods 
#swa predictions, swag, swag + outerproduct

criterion = F.cross_entropy

#now fast ensembling through several samples
def eval_fast_ensembling(loader, swa_model, criterion, samples = 10, scale = 1.0):
    correct = 0.0
    for i, (input, target) in enumerate(loader):
        #load data
        input = input.cuda(async=True)
        target = target.cuda(async=True)

        loss_sum = 0.0
        output = 0.0

        for _ in range(samples):
            #sample model
            #swa_model.collect_model(model)
            swa_model.sample(scale=scale)
            swa_model.eval()
            """ for i, (module, name) in enumerate(swa_model.params):
                if i==0:
                    print(module) """
            #now add forwards pass to running average
            output += swa_model(input)

            loss = criterion(output, target)

            loss_sum += loss.item() * input.size(0)

        output /= samples
        loss_sum /= samples

        pred = output.data.argmax(1, keepdim=True)
        correct += pred.eq(target.data.view_as(pred)).sum().item()

    return {
        'loss': loss_sum / len(loader.dataset),
        'accuracy': correct / len(loader.dataset) * 100.0,
    }

#swa predictions
#swag_model.collect_model(model)
swag_model.sample(0.0)
for i, (module, name) in enumerate(swag_model.params):
    if i==0:
        print(module)

swa_res = utils.eval(loaders['test'], swag_model, criterion)

#swag predictions
#swag_model.collect_model(model)
swag_model.sample(1.0)
swag_res = utils.eval(loaders['test'], swag_model, criterion)

#swag-cov1 predictions
""" swag_model.collect_model(model)
swag_model.sample(1.0, block_cov=True) """

#now the evaluation fast ensembling
swagy_res = eval_fast_ensembling(loaders['test'], swag_model, criterion, samples=100, scale=1.0)

print('SWA predictions', swa_res)
print('1 Sample SWAG predictions', swag_res)
print('Fast Ensembled SWAG predictions', swagy_res)