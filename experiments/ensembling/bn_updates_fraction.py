import argparse
import torch
import numpy as np
import tabulate
import time

from swag import data, models, utils, losses
from swag.posteriors import SWAG

parser = argparse.ArgumentParser(
    description="Dependence of the faction iof samples for BN updates"
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

parser.add_argument("--cov_mat", action="store_true", help="use cov mat (default: off)")
parser.add_argument(
    "--swag_rank", type=int, default=20, metavar="R", help="SWAG rank (default: 20)"
)

parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument(
    "--save_path",
    type=str,
    default=None,
    required=True,
    help="path to npz results file",
)

parser.add_argument(
    "--N", type=int, default=11, metavar="N", help="grid size (default: 25)"
)
parser.add_argument(
    "--S",
    type=int,
    default=30,
    metavar="S",
    help="number of samples for SWAG (default: 30)",
)

parser.add_argument(
    "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
)

args = parser.parse_args()

eps = 1e-12

torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)

args.device = None
if torch.cuda.is_available():
    args.device = torch.device("cuda")
else:
    args.device = torch.device("cpu")

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
)

print("Preparing model")
swag_model = SWAG(
    model_cfg.base,
    no_cov_mat=not args.cov_mat,
    max_num_models=args.swag_rank,
    *model_cfg.args,
    num_classes=num_classes,
    **model_cfg.kwargs
)
swag_model.to(args.device)

ckpt = torch.load(args.checkpoint)

criterion = losses.cross_entropy


fractions = np.logspace(-np.log10(0.005 * len(loaders["train"].dataset)), 0.0, args.N)
swa_accuracies = np.zeros(args.N)
swa_nlls = np.zeros(args.N)
swag_accuracies = np.zeros(args.N)
swag_nlls = np.zeros(args.N)

columns = ["fraction", "swa_acc", "swa_loss", "swag_acc", "swag_loss", "time"]

for i, fraction in enumerate(fractions):
    start_time = time.time()
    swag_model.load_state_dict(ckpt["state_dict"])

    swag_model.sample(0.0)
    utils.bn_update(loaders["train"], swag_model, subset=fraction)
    swa_res = utils.eval(loaders["test"], swag_model, criterion)
    swa_accuracies[i] = swa_res["accuracy"]
    swa_nlls[i] = swa_res["loss"]

    predictions = np.zeros((len(loaders["test"].dataset), num_classes))

    for j in range(args.S):
        swag_model.load_state_dict(ckpt["state_dict"])
        swag_model.sample(scale=0.5, cov=args.cov_mat)
        utils.bn_update(loaders["train"], swag_model, subset=fraction)
        sample_res = utils.predict(loaders["test"], swag_model)
        predictions += sample_res["predictions"]
        targets = sample_res["targets"]
    predictions /= args.S

    swag_accuracies[i] = np.mean(np.argmax(predictions, axis=1) == targets)
    swag_nlls[i] = -np.mean(
        np.log(predictions[np.arange(predictions.shape[0]), targets] + eps)
    )

    run_time = time.time() - start_time
    values = [
        fraction * 100.0,
        swa_accuracies[i],
        swa_nlls[i],
        swag_accuracies[i],
        swag_nlls[i],
        run_time,
    ]
    table = tabulate.tabulate([values], columns, tablefmt="simple", floatfmt="8.4f")
    if i == 0:
        table = table.split("\n")
        table = "\n".join([table[1]] + table)
    else:
        table = table.split("\n")[2]
    print(table)

np.savez(
    args.save_path,
    fractions=fractions,
    swa_accuracies=swa_accuracies,
    swa_nlls=swa_nlls,
    swag_accuracies=swag_accuracies,
    swag_nlls=swag_nlls,
)
