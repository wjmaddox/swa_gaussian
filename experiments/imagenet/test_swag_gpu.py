import argparse
import os
import random
import sys
import time
import tabulate

import torch
import torch.nn.functional as F
import torchvision.models

import data
from swag import utils, losses
from swag.posteriors import SWAG

parser = argparse.ArgumentParser(description="SGD/SWA training")

parser.add_argument(
    "--model",
    type=str,
    default=None,
    required=True,
    metavar="MODEL",
    help="model name (default: None)",
)
parser.add_argument(
    "--pretrained",
    action="store_true",
    help="pretrained model usage flag (default: off)",
)


args = parser.parse_args()

args.device = None
if torch.cuda.is_available():
    args.device = torch.device("cuda")
else:
    args.device = torch.device("cpu")

# torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

print("Using model %s" % args.model)
model_class = getattr(torchvision.models, args.model)

num_classes = 1000

print("Preparing model")
model = model_class(pretrained=args.pretrained, num_classes=num_classes)
model.to(args.device)

print("SWAG training")
swag_model = SWAG(
    model_class, no_cov_mat=False, max_num_models=20, num_classes=num_classes
)
swag_model.to("cpu")

for k in range(100):
    swag_model.collect_model(model)
    print(k + 1)
