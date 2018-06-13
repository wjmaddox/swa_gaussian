import torch, argparse
import torch.utils.data as tud
import models, swag, data, utils
import torch.nn.functional as F


import sys
sys.path.append('/home/wm326/CIFAR-10.1/code')
import cutils

parser = argparse.ArgumentParser(description='SGD/SWA training')
parser.add_argument('--file', type=str, default=None, required=True, help='training directory (default: None)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='input batch size (default: 128)')
parser.add_argument('--num_workers', type=int, default=4, metavar='N', help='number of workers (default: 4)')
parser.add_argument('--model', type=str, default='VGG16', metavar='MODEL',
                    help='model name (default: VGG16)')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

args = parser.parse_args()

version = 'default'
images, labels = cutils.load_new_test_data(version)
print('\nLoaded version "{}" of the CIFAR-10.1 dataset.'.format(version))
#print('There are {} images in the dataset.'.format(num_images))

cifar10_1 = tud.TensorDataset(torch.from_numpy(images).permute(0, 3, 1, 2).float(), torch.from_numpy(labels).long(), transform = model_cfg.transform_test)
loader = tud.DataLoader(cifar10_1, batch_size = args.batch_size, num_workers = args.num_workers)
num_classes = 10

torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

print('Using model %s' % args.model)
model_cfg = getattr(models, args.model)

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
        print(input.size(), target.size())
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

swa_res = utils.eval(loader, swag_model, criterion)

#swag predictions
#swag_model.collect_model(model)
swag_model.sample(1.0)
swag_res = utils.eval(loader, swag_model, criterion)

#swag-cov1 predictions
""" swag_model.collect_model(model)
swag_model.sample(1.0, block_cov=True) """

#now the evaluation fast ensembling
swag_100 = eval_fast_ensembling(loader, swag_model, criterion, samples=100, scale=1.0)
swag_10 = eval_fast_ensembling(loader, swag_model, criterion, samples = 10, scale=1.0)

print('SWA predictions', swa_res)
print('1 Sample SWAG predictions', swag_res)
print('SWAG predictions (100 samples)', swag_100)
print('SWAG predictions (10 samples)', swag_10)

accuracy = []
ivec = []
for i in range(1, 10):
    ivec.append(i)
    if i%101 is 0:
        print('now on ', i)
    out = eval_fast_ensembling(loader, swag_model, criterion, samples=i, scale=1.0)
    accuracy.append(out['accuracy'])

import matplotlib.pyplot as plt
plt.plot(ivec, accuracy)
plt.savefig('cifar101_accuracy_short.png')
