import argparse
import numpy as np
import os
import torch
import tabulate
import tqdm

import torch.nn.functional as F
from itertools import chain, product

from swag import data, losses, models, utils

parser = argparse.ArgumentParser(description="SGD/SWA/SWAG ensembling")
# parser.add_argument('--replications', type=int, default=10, help='number of passes through testing set')
parser.add_argument(
    "--dir",
    type=str,
    nargs="+",
    default=None,
    required=True,
    help="loading directories (default: None)",
)

parser.add_argument(
    "--dataset", type=str, default="CIFAR10", help="dataset name (default: CIFAR10)"
)
parser.add_argument(
    "--data_path",
    type=str,
    default="/scratch/datasets/",
    metavar="PATH",
    help="path to datasets location (default: None)",
)
parser.add_argument(
    "--use_test",
    dest="use_test",
    action="store_true",
    help="use test dataset instead of validation (default: False)",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=128,
    metavar="N",
    help="input batch size (default: 128)",
)
parser.add_argument(
    "--num_workers",
    type=int,
    default=4,
    metavar="N",
    help="number of workers (default: 4)",
)
parser.add_argument(
    "--model",
    type=str,
    default="VGG16",
    metavar="MODEL",
    help="model name (default: VGG16)",
)

parser.add_argument(
    "--epoch",
    type=int,
    default=160,
    metavar="N",
    help="epoch to begin evaluations at at (default: 200)",
)

parser.add_argument(
    "--no_ensembles", action="store_true", help="whether to run ensembles or not"
)
parser.add_argument(
    "--swa", action="store_true", help="whether to create swa from checkpoints"
)
# parser.add_argument('--swa', action='store_true', help='swa usage flag (default: off)')
# parser.add_argument('--cov_mat', action='store_true', help='save sample covariance')

# parser.add_argument('--plot', action='store_true', help='plot replications (default: off)')
parser.add_argument(
    "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
)
parser.add_argument(
    "--save_path", type=str, required=True, help="output file to save to"
)
args = parser.parse_args()

torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)


def find_models(dir, model_name="checkpoint", start_epoch=161, finish_epoch=325):

    all_models = os.popen("ls " + dir + "/" + model_name + "*.pt").read().split("\n")
    model_epochs = [int(t.replace(".", "-").split("-")[1]) for t in all_models[:-1]]
    models_to_use = [t >= start_epoch and t <= finish_epoch for t in model_epochs]

    model_names = list()
    for model_name, use in zip(all_models, models_to_use):
        if use:
            model_names.append(model_name)

    return model_names


def nll(outputs, labels):
    labels = labels.astype(int)
    idx = (np.arange(labels.size), labels)
    ps = outputs[idx]
    nll = -np.sum(np.log(ps))
    return nll


print("Using model %s" % args.model)
model_cfg = getattr(models, args.model)

print("Loading dataset %s from %s" % (args.dataset, args.data_path))
loaders, num_classes = data.loaders(
    args.dataset,
    args.data_path,
    args.batch_size,
    args.num_workers,
    model_cfg.transform_train,
    model_cfg.transform_test,
    use_validation=not args.use_test,
    split_classes=None,
)

print("Preparing model")
model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
model.cuda()
model.eval()

print("using cross-entropy loss")
# criterion = F.cross_entropy
criterion = losses.cross_entropy

dir_locs = []
for dir in args.dir:
    dir_locs.append(
        find_models(
            dir, model_name="checkpoint", start_epoch=args.epoch, finish_epoch=1000
        )
    )
flatten = lambda l: [item for sublist in l for item in sublist]
dir_locs = flatten(dir_locs)

columns = ["model", "epoch", "acc", "loss", "swa_acc", "swa_loss"]

pt_loss, pt_accuracy = list(), list()

if not args.no_ensembles:
    predictions = np.zeros((len(loaders["test"].dataset), num_classes, len(dir_locs)))
    targets = np.zeros(len(loaders["test"].dataset))


for i, ckpt in enumerate(dir_locs):

    model.load_state_dict(torch.load(ckpt)["state_dict"])
    epoch = int(ckpt.replace(".", "-").split("-")[1])
    model.eval()

    res = utils.eval(loaders["test"], model, criterion)

    pt_loss.append(res["loss"])
    pt_accuracy.append(res["accuracy"])

    if not args.no_ensembles:
        k = 0
        with torch.no_grad():
            for input, target in tqdm.tqdm(loaders["test"]):
                input = input.cuda(non_blocking=True)
                torch.manual_seed(1)

                output = model(input)

                predictions[k : k + input.size(0), :, i] += (
                    F.softmax(output, dim=1).cpu().numpy()
                )
                targets[k : (k + target.size(0))] = target.numpy()
                k += input.size(0)

        current_accuracy = (
            np.mean(np.argmax(np.sum(predictions[:, :, 0 : (i + 1)], 2), 1) == targets)
            * 100
        )
        # torch_mean_preds = torch.Tensor(np.sum(predictions[:,:,0:(i+1)],2)).float()
        mean_preds = np.sum(predictions[:, :, 0 : (i + 1)], 2) / (i + 1)

        current_loss = nll(mean_preds, targets) / targets.shape[0]

        values = [
            args.model,
            epoch,
            pt_accuracy[-1],
            pt_loss[-1],
            current_accuracy,
            current_loss,
        ]

    else:
        values = [args.model, epoch, pt_accuracy[-1], pt_loss[-1], None, None]

    table = tabulate.tabulate([values], columns, tablefmt="simple", floatfmt="8.4f")
    if i % 40 == 0:
        table = table.split("\n")
        table = "\n".join([table[1]] + table)
    else:
        table = table.split("\n")[2]
    print(table)

if not args.no_ensembles:
    np.savez(
        args.save_path,
        predictions=predictions,
        targets=targets,
        sgd_acc=pt_accuracy,
        sgd_loss=pt_loss,
    )
else:
    np.savez(args.save_path, sgd_acc=pt_accuracy, sgd_loss=pt_loss)
