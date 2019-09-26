import argparse
import os
import os

os.sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
from temp_scaling import optimal_temp_scale, rescale_temp
import numpy as np
import pickle

parser = argparse.ArgumentParser(description="SGD/SWA training")
parser.add_argument(
    "--path", type=str, default=None, required=True, help="dataset name"
)
parser.add_argument(
    "--append", type=str, default="", required=False, help="string to add to filenames"
)
parser.add_argument("--print_names", action="store_true", help="print method names")
parser.add_argument("--num_bins", type=int, default=20, help="bin number for ECE")
args = parser.parse_args()


def parse(npz_arr):
    return npz_arr["predictions"], npz_arr["targets"]


def calibration_curve(npz_arr):
    outputs, labels = parse(npz_arr)
    if outputs is None:
        out = None
    else:
        confidences = np.max(outputs, 1)
        step = (confidences.shape[0] + args.num_bins - 1) // args.num_bins
        bins = np.sort(confidences)[::step]
        if confidences.shape[0] % step != 1:
            bins = np.concatenate((bins, [np.max(confidences)]))
        # bins = np.linspace(0.1, 1.0, 30)
        predictions = np.argmax(outputs, 1)
        bin_lowers = bins[:-1]
        bin_uppers = bins[1:]

        accuracies = predictions == labels

        xs = []
        ys = []
        zs = []

        # ece = Variable(torch.zeros(1)).type_as(confidences)
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = (confidences > bin_lower) * (confidences < bin_upper)
            prop_in_bin = in_bin.mean()
            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                xs.append(avg_confidence_in_bin)
                ys.append(accuracy_in_bin)
                zs.append(prop_in_bin)
        xs = np.array(xs)
        ys = np.array(ys)
        zs = np.array(zs)

        out = {"confidence": xs, "accuracy": ys, "p": zs, "ece": ece}
    return out


def get_path(filename, append="", no_npz=False):
    path = os.path.join(args.path, filename + append)
    if not no_npz:
        path += ".npz"
    print(path)
    return path


def nonedict():
    return {"predictions": None, "targets": None}


try:
    swag_cov = np.load(get_path("unc_swagcov", append=args.append))
except:
    swag_cov = nonedict()
    print("failed")
try:
    swag_diag = np.load(get_path("unc_swagdiag", append=args.append))
except:
    swag_diag = nonedict()
    print("failed")
try:
    swa = np.load(get_path("unc_swa", append=args.append))
except:
    swa = nonedict()
    print("failed")
try:
    sgd = np.load(get_path("unc_sgd", append=args.append))
except:
    sgd = nonedict()
    print("failed")
try:
    swa_val = np.load(get_path("unc_swa_val"))
except:
    swa_val = nonedict()
    print("failed")
try:
    sgd_val = np.load(get_path("unc_sgd_val"))
except:
    sgd_val = nonedict()
    print("failed")
try:
    swa_drop = np.load(get_path("unc_swa_drop", append=args.append))
except:
    swa_drop = nonedict()
    print("failed")
try:
    sgd_drop = np.load(get_path("unc_sgd_drop", append=args.append))
except:
    sgd_drop = nonedict()
    print("failed")
try:
    laplace_sgd = np.load(get_path("unc_laplace", append=args.append))
except:
    laplace_sgd = nonedict()
    print("failed")
try:
    sgd_ens = np.load(get_path("sgd_ens_preds", append=args.append))
except:
    sgd_ens = nonedict()
    print("failed")

# Temperature scaling
if swa["predictions"] is not None and swa_val["predictions"] is not None:
    print("Rescaling SWA temp")
    T_swa, rescaled_swa = optimal_temp_scale(
        swa_val["predictions"], swa_val["targets"], max_iter=50, lr=1e-3
    )
    print(T_swa)
    if np.isnan(T_swa):
        T_swa = 1
    preds = rescale_temp(swa["predictions"], T_swa)
    swa_temp = {"predictions": preds, "targets": swa["targets"]}
else:
    swa_temp = nonedict()

if sgd["predictions"] is not None and sgd_val["predictions"] is not None:
    print("Rescaling SGD temp")
    T_sgd, rescaled_sgd = optimal_temp_scale(
        sgd_val["predictions"], sgd_val["targets"], max_iter=50, lr=1e-3
    )
    if np.isnan(T_sgd):
        T_sgd = 1
    preds = rescale_temp(sgd["predictions"], T_sgd)
    sgd_temp = {"predictions": preds, "targets": sgd["targets"]}
else:
    sgd_temp = nonedict()

methods = [
    (sgd, "SGD"),
    (swa, "SWA"),
    (swag_diag, "SWAG-Diag"),
    (swag_cov, "SWAG-Cov"),
    (sgd_drop, "SGD-Drop"),
    (swa_drop, "SWA-Drop"),
    (sgd_temp, "SGD-temp"),
    (swa_temp, "SWA-temp"),
    (laplace_sgd, "Laplace-SGD"),
    (nonedict(), "Laplace-SWA"),
    (sgd_ens, "SGD-Ens"),
]

results = dict()

for method, name in methods:
    results[name] = calibration_curve(method)
with open(get_path("calibration_curves.pkl", no_npz=True), "wb") as f:
    pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)
