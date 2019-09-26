import argparse
import torch
import numpy as np
import sklearn.decomposition
import tabulate
import time

from swag import data, models, utils, losses
from swag.posteriors import SWAG

parser = argparse.ArgumentParser(description="PCA plane")

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
    "--swag_rank", type=int, default=20, metavar="R", help="SWAG rank (default: 20)"
)

parser.add_argument("--checkpoint", action="append")
parser.add_argument(
    "--save_path",
    type=str,
    default=None,
    required=True,
    help="path to npz results file",
)

parser.add_argument(
    "--dist",
    type=float,
    default=30.0,
    metavar="D",
    help="dist to travel along a direction (default: 30.0)",
)
parser.add_argument(
    "--N",
    type=int,
    default=21,
    metavar="N",
    help="number of points on a grid (default: 31)",
)
parser.add_argument(
    "--PC1",
    type=int,
    default=0,
    metavar="PC1",
    help="index of the first principal axis (default: 0)",
)
parser.add_argument(
    "--PC2",
    type=int,
    default=1,
    metavar="PC2",
    help="index of the second principal axis (default: 1)",
)


parser.add_argument(
    "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
)

args = parser.parse_args()

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
model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
model.to(args.device)

swag_model = SWAG(
    model_cfg.base,
    no_cov_mat=False,
    max_num_models=args.swag_rank,
    *model_cfg.args,
    num_classes=num_classes,
    **model_cfg.kwargs
)
swag_model.to(args.device)

criterion = losses.cross_entropy

W = []
num_checkpoints = len(args.checkpoint)
for path in args.checkpoint:
    print("Loading %s" % path)
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["state_dict"])

    swag_model.collect_model(model)
    W.append(
        np.concatenate([p.detach().cpu().numpy().ravel() for p in model.parameters()])
    )
W = np.array(W)

mean, variances, cov_mat_list = swag_model.export_numpy_params(export_cov_mat=True)
cov_mat = np.hstack([mat.reshape(args.swag_rank, -1) for mat in cov_mat_list])

tsvd = sklearn.decomposition.TruncatedSVD(n_components=args.swag_rank, n_iter=7)
tsvd.fit(cov_mat)

train_acc = np.zeros((args.N, args.N))
train_loss = np.zeros((args.N, args.N))
test_acc = np.zeros((args.N, args.N))
test_loss = np.zeros((args.N, args.N))

u = tsvd.components_[args.PC1, :].copy()
u /= np.linalg.norm(u)
v = tsvd.components_[args.PC2, :].copy()
v /= np.linalg.norm(v)

ex_diag = np.sum(u * variances * u)
ey_diag = np.sum(v * variances * v)
exy_diag = np.sum(u * variances * v)
E_diag = np.array([[ex_diag, exy_diag], [exy_diag, ey_diag]])

ex = ex_diag + np.sum(np.square(np.dot(cov_mat, u))) / (cov_mat.shape[0] - 1)
ey = ey_diag + np.sum(np.square(np.dot(cov_mat, v))) / (cov_mat.shape[0] - 1)
exy = exy_diag + np.sum(np.dot(cov_mat, u) * np.dot(cov_mat, v)) / (
    cov_mat.shape[0] - 1
)
E_cov = np.array([[ex, exy], [exy, ey]])

dist_x = max(args.dist, 3.4 * np.sqrt(ex))
dist_y = max(args.dist, 3.4 * np.sqrt(ey))
dist = max(dist_x, dist_y)

component_variances = np.dot(
    np.dot(tsvd.components_, cov_mat.T), np.dot(cov_mat, tsvd.components_.T)
) / (cov_mat.shape[0] - 1)

M = np.vstack((u[None, :], v[None, :]))
W_dev = W - np.mean(W, axis=0, keepdims=True)
trajectory_projection = np.dot(M, W_dev.T)

trajectory_full_variance = np.sum(np.sum(np.square(W_dev), axis=1)) / (W.shape[0] - 1)
trajectory_exp_variance = np.dot(
    np.dot(tsvd.components_, W_dev.T), np.dot(W_dev, tsvd.components_.T)
) / (W.shape[0] - 1)

xs = np.linspace(-dist, dist, args.N)
ys = np.linspace(-dist, dist, args.N)

columns = ["x", "y", "tr_loss", "tr_acc", "te_loss", "te_acc", "time"]

for i, x in enumerate(xs):
    for j, y in enumerate(ys):
        t_start = time.time()
        w = mean + x * u + y * v

        offset = 0
        for param in model.parameters():
            size = np.prod(param.size())
            param.data.copy_(
                param.new_tensor(w[offset : offset + size].reshape(param.size()))
            )
            offset += size

        utils.bn_update(loaders["train"], model)
        train_res = utils.eval(loaders["train"], model, criterion)
        test_res = utils.eval(loaders["test"], model, criterion)

        train_acc[i, j] = train_res["accuracy"]
        train_loss[i, j] = train_res["loss"]
        test_acc[i, j] = test_res["accuracy"]
        test_loss[i, j] = test_res["loss"]

        t = time.time() - t_start
        values = [
            x,
            y,
            train_loss[i, j],
            train_acc[i, j],
            test_loss[i, j],
            test_acc[i, j],
            t,
        ]
        table = tabulate.tabulate([values], columns, tablefmt="simple", floatfmt="8.4f")
        if j == 0:
            table = table.split("\n")
            table = "\n".join([table[1]] + table)
        else:
            table = table.split("\n")[2]
        print(table)

np.savez(
    args.save_path,
    N=num_checkpoints,
    dim=W.shape[1],
    explained_variance=tsvd.explained_variance_,
    explained_variance_ratio=tsvd.explained_variance_ratio_,
    pc1_id=args.PC1,
    pc2_id=args.PC2,
    trajectory_projection=trajectory_projection,
    xs=xs,
    ys=ys,
    train_acc=train_acc,
    train_err=100.0 - train_acc,
    train_loss=train_loss,
    test_acc=test_acc,
    test_err=100.0 - test_acc,
    test_loss=test_loss,
    E_diag=E_diag,
    E_cov=E_cov,
    component_variances=component_variances,
    trajectory_full_variance=trajectory_full_variance,
    trajectory_exp_variance=trajectory_exp_variance,
)
