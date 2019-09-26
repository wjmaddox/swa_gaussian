import argparse
import os
import random
import sys
import time
import tabulate

import numpy as np

import torch
import torch.nn.functional as F
import torchvision.models

import data
from swag import utils, losses

parser = argparse.ArgumentParser(description="SGD/SWA training")

parser.add_argument(
    "--data_path",
    type=str,
    default=None,
    required=True,
    metavar="PATH",
    help="path to datasets location (default: None)",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=256,
    metavar="N",
    help="input batch size (default: 256)",
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
    default=None,
    required=True,
    metavar="MODEL",
    help="model name (default: None)",
)
parser.add_argument(
    "--parallel", action="store_true", help="data parallel model switch (default: off)"
)

parser.add_argument(
    "--ckpt",
    type=str,
    required=True,
    default=None,
    metavar="CKPT",
    help="checkpoint to load (default: None)",
)
parser.add_argument(
    "--ckpt_cut_prefix",
    action="store_true",
    help='cut "module." prefix from state dict (default: off)',
)

parser.add_argument(
    "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
)

parser.add_argument(
    "--save_path",
    type=str,
    default=None,
    required=True,
    help="path to npz results file",
)

args = parser.parse_args()

eps = 1e-12

args.device = None
if torch.cuda.is_available():
    args.device = torch.device("cuda")
else:
    args.device = torch.device("cpu")

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

print("Using model %s" % args.model)
model_class = getattr(torchvision.models, args.model)

print("Loading ImageNet from %s" % (args.data_path))
loaders, num_classes = data.loaders(args.data_path, args.batch_size, args.num_workers)

print("Preparing model")
model = model_class(num_classes=num_classes)
model.to(args.device)

criterion = losses.cross_entropy

if args.parallel:
    print("Using Data Parallel model")
    model = torch.nn.parallel.DataParallel(model)

print("Loading checkpoint %s" % args.ckpt)
checkpoint = torch.load(args.ckpt)
state_dict = checkpoint["state_dict"]
if args.ckpt_cut_prefix:
    state_dict = {
        k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()
    }
model.load_state_dict(state_dict)

print("BN update")
utils.bn_update(loaders["train"], model, verbose=True, subset=0.1)
print("EVAL")
res = utils.predict(loaders["test"], model, verbose=True)

predictions = res["predictions"]
targets = res["targets"]


accuracy = np.mean(np.argmax(predictions, axis=1) == targets)
nll = -np.mean(np.log(predictions[np.arange(predictions.shape[0]), targets] + eps))
print("Accuracy: %.2f%% NLL: %.4f" % (accuracy * 100, nll))
entropies = -np.sum(np.log(predictions + eps) * predictions, axis=1)


np.savez(
    args.save_path,
    accuracy=accuracy,
    nll=nll,
    entropies=entropies,
    predictions=predictions,
    targets=targets,
)
