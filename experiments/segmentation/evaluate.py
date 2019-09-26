"""
    evaluation script for sgd/swa
"""
import time
from pathlib import Path
import numpy as np

# import matplotlib.pyplot as plt
import os, sys
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from swag.models import tiramisu
from swag import data, losses, models
from utils.training import test

from swag.posteriors import SWAG
from swag.utils import adjust_learning_rate, bn_update

parser = argparse.ArgumentParser(description="SGD/SWA training")

parser.add_argument(
    "--data_path",
    type=str,
    default="/home/wesley/Documents/Code/SegNet-Tutorial/CamVid/",
    metavar="PATH",
    help="path to datasets location (default: None)",
)

parser.add_argument(
    "--batch_size",
    type=int,
    default=4,
    metavar="N",
    help="input batch size (default: 4)",
)

parser.add_argument(
    "--resume",
    type=str,
    default=None,
    metavar="CKPT",
    help="checkpoint to resume training from (default: None)",
)
parser.add_argument(
    "--swa_resume",
    type=str,
    default=None,
    metavar="CKPT",
    help="checkpoint to restor SWA from (default: None)",
)
parser.add_argument(
    "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
)
parser.add_argument(
    "--use_test",
    action="store_true",
    help="whether to use test or validation dataset (default: val)",
)
parser.add_argument(
    "--output",
    type=str,
    required=True,
    help="output file to save model predictions and targets",
)

parser.add_argument(
    "--loss", type=str, choices=["cross_entropy", "aleatoric"], default="cross_entropy"
)

args = parser.parse_args()

torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

model_cfg = getattr(models, "FCDenseNet67")
loaders, num_classes = data.loaders(
    "CamVid",
    args.data_path,
    args.batch_size,
    4,
    ft_batch_size=1,
    transform_train=model_cfg.transform_train,
    transform_test=model_cfg.transform_test,
    joint_transform=model_cfg.joint_transform,
    ft_joint_transform=model_cfg.ft_joint_transform,
    target_transform=model_cfg.target_transform,
)

# criterion = nn.NLLLoss(weight=camvid.class_weight[:-1].cuda(), reduction='none').cuda()
if args.loss == "cross_entropy":
    criterion = losses.seg_cross_entropy
else:
    criterion = losses.seg_ale_cross_entropy

# construct and load model
if args.swa_resume is not None:
    checkpoint = torch.load(args.swa_resume)
    model = SWAG(
        model_cfg.base,
        no_cov_mat=False,
        max_num_models=20,
        num_classes=num_classes,
        use_aleatoric=args.loss == "aleatoric",
    )
    model.cuda()
    model.load_state_dict(checkpoint["state_dict"])

    model.sample(0.0)
    bn_update(loaders["fine_tune"], model)
else:
    model = model_cfg.base(
        num_classes=num_classes, use_aleatoric=args.loss == "aleatoric"
    ).cuda()
    checkpoint = torch.load(args.resume)
    start_epoch = checkpoint["epoch"]
    print(start_epoch)
    model.load_state_dict(checkpoint["state_dict"])

print(len(loaders["test"]))
if args.use_test:
    print("Using test dataset")
    test_loader = "test"
else:
    test_loader = "val"
loss, err, mIOU, model_output_targets = test(
    model,
    loaders[test_loader],
    criterion,
    return_outputs=True,
    return_scale=args.loss == "aleatoric",
)
print(loss, 1 - err, mIOU)

outputs = np.concatenate(model_output_targets["outputs"])
targets = np.concatenate(model_output_targets["targets"])

if args.loss == "aleatoric":
    scales = np.concatenate(model_output_targets["scales"])
else:
    scales = None
np.savez(args.output, preds=outputs, targets=targets, scales=scales)
