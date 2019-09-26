"""
    ensemble evaluation script for segmentation
    note: only swag and dropout have been tested
"""

import time
from pathlib import Path
import numpy as np

# import matplotlib.pyplot as plt
import os, sys
import argparse
import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

from swag.models import tiramisu
from swag import data, losses, models
from utils.training import test

from swag.utils import adjust_learning_rate, bn_update
from swag.posteriors import SWAG, KFACLaplace

parser = argparse.ArgumentParser(description="SGD/SWA training")
parser.add_argument("--file", type=str, default=None, required=True, help="checkpoint")

parser.add_argument(
    "--data_path",
    type=str,
    default="/home/wesley/Documents/Code/SegNet-Tutorial/CamVid/",
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
    default=1,
    metavar="N",
    help="input batch size (default: 5)",
)
parser.add_argument(
    "--num_workers",
    type=int,
    default=4,
    metavar="N",
    help="number of workers (default: 4)",
)
parser.add_argument(
    "--method",
    type=str,
    default="SWAG",
    choices=["SWAG", "SGD", "HomoNoise", "Dropout", "SWAGDrop"],
    required=True,
)
parser.add_argument(
    "--save_path",
    type=str,
    default=None,
    required=True,
    help="path to npz results file",
)
parser.add_argument("--N", type=int, default=30)
parser.add_argument("--scale", type=float, default=1.0)
parser.add_argument(
    "--cov_mat", action="store_true", help="use sample covariance for swag"
)
parser.add_argument("--use_diag", action="store_true", help="use diag cov for swag")

parser.add_argument(
    "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
)
parser.add_argument(
    "--loss", type=str, choices=["cross_entropy", "aleatoric"], default="cross_entropy"
)

args = parser.parse_args()

eps = 1e-12
if args.cov_mat:
    args.cov_mat = True
else:
    args.cov_mat = False

torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

model_cfg = getattr(models, "FCDenseNet67")
loaders, num_classes = data.loaders(
    "CamVid",
    args.data_path,
    args.batch_size,
    args.num_workers,
    ft_batch_size=1,
    transform_train=model_cfg.transform_train,
    transform_test=model_cfg.transform_test,
    joint_transform=model_cfg.joint_transform,
    ft_joint_transform=model_cfg.ft_joint_transform,
    target_transform=model_cfg.target_transform,
)

if args.loss == "cross_entropy":
    criterion = losses.seg_cross_entropy
else:
    criterion = losses.seg_ale_cross_entropy

print("Preparing model")
if args.method in ["SWAG", "HomoNoise", "SWAGDrop"]:
    model = SWAG(
        model_cfg.base,
        no_cov_mat=False,
        max_num_models=20,
        num_classes=num_classes,
        use_aleatoric=args.loss == "aleatoric",
    )

elif args.method in ["SGD", "Dropout"]:
    # construct and load model
    model = model_cfg.base(
        num_classes=num_classes, use_aleatoric=args.loss == "aleatoric"
    )
else:
    assert False
model.cuda()


def train_dropout(m):
    if m.__module__ == torch.nn.modules.dropout.__name__:
        # print('here')
        m.train()


print("Loading model %s" % args.file)
checkpoint = torch.load(args.file)
model.load_state_dict(checkpoint["state_dict"])

if args.method == "HomoNoise":
    std = 0.01
    for module, name in model.params:
        mean = module.__getattr__("%s_mean" % name)
        module.__getattr__("%s_sq_mean" % name).copy_(mean ** 2 + std ** 2)

predictions = np.zeros((len(loaders["test"].dataset), 11, 360, 480))
targets = np.zeros((len(loaders["test"].dataset), 360, 480))

if args.loss == "aleatoric":
    scales = np.zeros((len(loaders["test"].dataset), 11, 360, 480))
else:
    scales = None

print(targets.size)

for i in range(args.N):
    print("%d/%d" % (i + 1, args.N))

    if args.method not in ["SGD", "Dropout"]:
        sample_with_cov = args.cov_mat and not args.use_diag
        with torch.no_grad():
            model.sample(scale=args.scale, cov=sample_with_cov)

    if "SWAG" in args.method:
        bn_update(loaders["fine_tune"], model)

    model.eval()
    if args.method in ["Dropout", "SWAGDrop"]:
        model.apply(train_dropout)

    k = 0
    current_predictions = np.zeros_like(predictions)
    for input, target in tqdm.tqdm(loaders["test"]):
        input = input.cuda(non_blocking=True)
        torch.manual_seed(i)

        with torch.no_grad():
            output = model(input)

            if args.loss == "aleatoric":
                scale = output[:, 1, :, :, :].abs().cpu().numpy()
                output = output[:, 0, :, :, :]

                scales[k : k + input.size(0), :, :, :] = scale

            batch_probs = F.softmax(output, dim=1).cpu().numpy()

            predictions[k : k + input.size(0), :, :, :] += batch_probs

            current_predictions[k : k + input.size(0), :, :, :] = batch_probs

        targets[k : (k + target.size(0)), :, :] = target.numpy()
        k += input.size(0)

    # np.savez(args.save_path+'pred_'+str(i), predictions = current_predictions)

    print(np.mean(np.argmax(predictions, axis=1) == targets))
predictions /= args.N

np.savez(args.save_path, predictions=predictions, targets=targets, scales=scales)
