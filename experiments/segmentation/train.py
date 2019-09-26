"""
    training script for segmentation models
    partial port of our own train/run_swag.py file
    note: no options to train swag-diag
"""

import time
from pathlib import Path
import numpy as np
import os, sys
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from functools import partial

import utils.training as train_utils

from swag import models, losses, data
from swag.posteriors import SWAG
from swag.utils import bn_update, adjust_learning_rate, schedule, save_checkpoint

parser = argparse.ArgumentParser(description="SGD/SWA training")

parser.add_argument("--dataset", type=str, default="CamVid")
parser.add_argument(
    "--data_path",
    type=str,
    default="/home/wesley/Documents/Code/SegNet-Tutorial/CamVid/",
    metavar="PATH",
    help="path to datasets location (default: None)",
)
parser.add_argument(
    "--dir",
    type=str,
    default=None,
    required=True,
    help="training directory (default: None)",
)

parser.add_argument(
    "--epochs",
    type=int,
    default=850,
    metavar="N",
    help="number of epochs to train (default: 850)",
)
parser.add_argument(
    "--save_freq",
    type=int,
    default=10,
    metavar="N",
    help="save frequency (default: 10)",
)
parser.add_argument(
    "--eval_freq",
    type=int,
    default=5,
    metavar="N",
    help="evaluation frequency (default: 5)",
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
    "--batch_size",
    type=int,
    default=2,
    metavar="N",
    help="input batch size (default: 2)",
)
parser.add_argument(
    "--lr_init",
    type=float,
    default=1e-4,
    metavar="LR",
    help="initial learning rate (default: 0.01)",
)
parser.add_argument(
    "--lr_decay",
    type=float,
    default=0.995,
    help="amount of learning rate decay per epoch (default: 0.995)",
)
parser.add_argument(
    "--wd", type=float, default=1e-4, help="weight decay (default: 1e-4)"
)
parser.add_argument(
    "--optimizer", type=str, choices=["RMSProp", "SGD"], default="RMSProp"
)
parser.add_argument(
    "--num_workers",
    type=int,
    default=4,
    metavar="N",
    help="number of workers (default: 4)",
)

parser.add_argument(
    "--ft_start",
    type=int,
    default=750,
    help="begin fine-tuning with full sized images (default: 750)",
)
parser.add_argument(
    "--ft_batch_size", type=int, default=1, help="fine-tuning batch size (default: 1)"
)

parser.add_argument("--swa", action="store_true", help="swa usage flag (default: off)")
parser.add_argument(
    "--swa_start",
    type=float,
    default=800,
    metavar="N",
    help="SWA start epoch number (default: 161)",
)
parser.add_argument(
    "--swa_lr", type=float, default=0.02, metavar="LR", help="SWA LR (default: 0.02)"
)
parser.add_argument(
    "--swa_c_epochs",
    type=int,
    default=1,
    metavar="N",
    help="SWA model collection frequency/cycle length in epochs (default: 1)",
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
    "--loss", type=str, choices=["cross_entropy", "aleatoric"], default="cross_entropy"
)
parser.add_argument(
    "--use_weights", action="store_true", help="whether to use weighted loss"
)

args = parser.parse_args()
if torch.cuda.is_available():
    args.device = torch.device("cuda")
else:
    args.device = torch.device("cpu")

torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

print("Preparing directory %s" % args.dir)
os.makedirs(args.dir, exist_ok=True)
with open(os.path.join(args.dir, "command.sh"), "w") as f:
    f.write(" ".join(sys.argv))
    f.write("\n")

print("Using model %s" % args.model)
model_cfg = getattr(models, args.model)

loaders, num_classes = data.loaders(
    args.dataset,
    args.data_path,
    args.batch_size,
    args.num_workers,
    ft_batch_size=args.ft_batch_size,
    transform_train=model_cfg.transform_train,
    transform_test=model_cfg.transform_test,
    joint_transform=model_cfg.joint_transform,
    ft_joint_transform=model_cfg.ft_joint_transform,
    target_transform=model_cfg.target_transform,
)
print("Beginning with cropped images")
train_loader = loaders["train"]

print("Preparing model")
model = model_cfg.base(
    *model_cfg.args,
    num_classes=num_classes,
    **model_cfg.kwargs,
    use_aleatoric=args.loss == "aleatoric"
)
model.cuda()
model.apply(train_utils.weights_init)

if args.optimizer == "RMSProp":
    optimizer = torch.optim.RMSprop(
        model.parameters(), lr=args.lr_init, weight_decay=args.wd
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay)
else:
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr_init, weight_decay=args.wd, momentum=0.9
    )

start_epoch = 1

if args.loss == "cross_entropy":
    criterion = losses.seg_cross_entropy
else:
    criterion = losses.seg_ale_cross_entropy

if args.use_weights:
    class_weights = torch.FloatTensor(
        [
            0.58872014284134,
            0.51052379608154,
            2.6966278553009,
            0.45021694898605,
            1.1785038709641,
            0.77028578519821,
            2.4782588481903,
            2.5273461341858,
            1.0122526884079,
            3.2375309467316,
            4.1312313079834,
        ]
    ).cuda()

    criterion = partial(criterion, weight=class_weights)

if args.resume is not None:
    print("Resume training from %s" % args.resume)
    checkpoint = torch.load(args.resume)
    start_epoch = checkpoint["epoch"]
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    del checkpoint


if args.swa:
    print("SWAG training")
    swag_model = SWAG(
        model_cfg.base,
        no_cov_mat=False,
        max_num_models=20,
        *model_cfg.args,
        num_classes=num_classes,
        use_aleatoric=args.loss == "aleatoric",
        **model_cfg.kwargs
    )
    swag_model.to(args.device)
else:
    print("SGD training")

if args.swa and args.swa_resume is not None:
    checkpoint = torch.load(args.swa_resume)
    swag_model = SWAG(
        model_cfg.base,
        no_cov_mat=False,
        max_num_models=20,
        *model_cfg.args,
        num_classes=num_classes,
        use_aleatoric=args.loss == "aleatoric",
        **model_cfg.kwargs
    )
    swag_model.to(args.device)
    swag_model.load_state_dict(checkpoint["state_dict"])

for epoch in range(start_epoch, args.epochs + 1):
    since = time.time()

    ### Train ###
    if epoch == args.ft_start:
        print("Now replacing data loader with fine-tuned data loader.")
        train_loader = loaders["fine_tune"]

    trn_loss, trn_err = train_utils.train(model, train_loader, optimizer, criterion)
    print(
        "Epoch {:d}\nTrain - Loss: {:.4f}, Acc: {:.4f}".format(
            epoch, trn_loss, 1 - trn_err
        )
    )
    time_elapsed = time.time() - since
    print("Train Time {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))

    if epoch % args.eval_freq is 0:
        ### Test ###
        val_loss, val_err, val_iou = train_utils.test(model, loaders["val"], criterion)
        print(
            "Val - Loss: {:.4f} | Acc: {:.4f} | IOU: {:.4f}".format(
                val_loss, 1 - val_err, val_iou
            )
        )

    time_elapsed = time.time() - since
    print("Total Time {:.0f}m {:.0f}s\n".format(time_elapsed // 60, time_elapsed % 60))

    if (
        args.swa
        and (epoch + 1) > args.swa_start
        and (epoch + 1 - args.swa_start) % args.swa_c_epochs == 0
    ):
        print("Saving SWA model at epoch: ", epoch)
        swag_model.collect_model(model)

        if epoch % args.eval_freq is 0:
            swag_model.sample(0.0)
            bn_update(train_loader, swag_model)
            val_loss, val_err, val_iou = train_utils.test(
                swag_model, loaders["val"], criterion
            )
            print(
                "SWA Val - Loss: {:.4f} | Acc: {:.4f} | IOU: {:.4f}".format(
                    val_loss, 1 - val_err, val_iou
                )
            )

    ### Checkpoint ###
    if epoch % args.save_freq is 0:
        print("Saving model at Epoch: ", epoch)
        save_checkpoint(
            dir=args.dir,
            epoch=epoch,
            state_dict=model.state_dict(),
            optimizer=optimizer.state_dict(),
        )
        if args.swa and (epoch + 1) > args.swa_start:
            save_checkpoint(
                dir=args.dir,
                epoch=epoch,
                name="swag",
                state_dict=swag_model.state_dict(),
            )

    if args.optimizer == "RMSProp":
        ### Adjust Lr ###
        if epoch < args.ft_start:
            scheduler.step(epoch=epoch)
        else:
            scheduler.step(epoch=-1)  # reset to args.lr_init

    elif args.optimizer == "SGD":
        lr = schedule(
            epoch, args.lr_init, args.epochs, args.swa, args.swa_start, args.swa_lr
        )
        adjust_learning_rate(optimizer, lr)

### Test set ###
if args.swa:
    swag_model.sample(0.0)
    bn_update(train_loader, swag_model)
    test_loss, test_err, test_iou = train_utils.test(
        swag_model, loaders["test"], criterion
    )
    print(
        "SWA Test - Loss: {:.4f} | Acc: {:.4f} | IOU: {:.4f}".format(
            test_loss, 1 - test_err, test_iou
        )
    )

test_loss, test_err, test_iou = train_utils.test(model, loaders["test"], criterion)
print(
    "SGD Test - Loss: {:.4f} | Acc: {:.4f} | IOU: {:.4f}".format(
        test_loss, 1 - test_err, test_iou
    )
)
