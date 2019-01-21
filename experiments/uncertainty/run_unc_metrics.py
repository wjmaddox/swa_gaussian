import argparse
import os
import os
os.sys.path.append("/home/izmailovpavel/Documents/Projects/private_swa_uncertainties/experiments/uncertainty")
from temp_scaling import optimal_temp_scale, rescale_temp
import numpy as np
import pickle

parser = argparse.ArgumentParser(description='SGD/SWA training')
parser.add_argument('--path', type=str, default=None, required=True, help='dataset name')
parser.add_argument('--append', type=str, default="", required=False, help='string to add to filenames')
parser.add_argument('--print_names', action='store_true', help = 'print method names')

args = parser.parse_args()

def parse(npz_arr):
    return npz_arr["predictions"], npz_arr["targets"]

def accuracy(npz_arr, name="", print_name=False):
    outputs, labels = parse(npz_arr)
    if outputs is None:
        acc = None
    else:
        acc = np.mean(np.argmax(outputs, axis=1) == labels)
    if print_name:
        print(name, end=": ")
    if acc is not None:
        print("%.4f"%acc)
    else:
        print("-")
    return acc


def ece(npz_arr, name="", bins=np.arange(0.1, 1.05, 0.05), print_name=False):
    outputs, labels = parse(npz_arr)
    if outputs is None:
        ece = None
    else:
        confidences = np.max(outputs, 1)
        predictions = np.argmax(outputs,1)
        bin_lowers = bins[:-1]
        bin_uppers = bins[1:]
        
        accuracies = predictions == labels

        #ece = Variable(torch.zeros(1)).type_as(confidences)
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = (confidences > bin_lower) * (confidences < bin_upper)
            prop_in_bin = in_bin.mean()
            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin-accuracy_in_bin) * prop_in_bin
    if print_name:
        print(name, end=": ")
    if ece is not None:
        print("%.3f"%ece)
    else:
        print("-")
    return ece


def nll(npz_arr, name="", print_name=False):
    outputs, labels = parse(npz_arr)
    if outputs is None:
        nll = None
    else:
        labels = labels.astype(int)
        idx = (np.arange(labels.size), labels)
        ps = outputs[idx]
        nll = -np.sum(np.log(ps))
    if print_name:
        print(name, end=": ")
    if nll is not None:
        print("%.3f"%nll)
    else:
        print("-")
    return nll


def get_path(filename, append="", no_npz=False):
    path = os.path.join(args.path, filename + append)
    if not no_npz:
        path += ".npz"
    print(path)
    return path


def nonedict():
    return {"predictions": None, "targets": None}

try: swag_cov = np.load(get_path("unc_swagcov", append=args.append))
except: swag_cov = nonedict(); print("failed") 
try: swag_diag = np.load(get_path("unc_swagdiag", append=args.append))
except: swag_diag = nonedict(); print("failed") 
try: swa = np.load(get_path("unc_swa", append=args.append))
except: swa = nonedict(); print("failed")
try: sgd = np.load(get_path("unc_sgd", append=args.append))
except: sgd = nonedict(); print("failed")
try: swa_val = np.load(get_path("unc_swa_val"))
except: swa_val = nonedict(); print("failed")
try: sgd_val = np.load(get_path("unc_sgd_val"))
except: sgd_val = nonedict(); print("failed")
try: swa_drop = np.load(get_path("unc_swa_drop", append=args.append))
except: swa_drop = nonedict(); print("failed")
try: sgd_drop = np.load(get_path("unc_sgd_drop", append=args.append))
except: sgd_drop = nonedict(); print("failed")
try: laplace_sgd = np.load(get_path("unc_laplace", append=args.append))
except: laplace_sgd = nonedict(); print("failed")
try: sgd_ens = np.load(get_path("sgd_ens_preds", append=args.append))
except: sgd_ens = nonedict(); print("failed")

# Temperature scaling
if swa["predictions"] is not None and swa_val["predictions"] is not None:
    print("Rescaling SWA temp")
    T_swa, rescaled_swa = optimal_temp_scale(swa_val["predictions"], swa_val["targets"], max_iter=50, lr=1e-3)
    print(T_swa)
    if np.isnan(T_swa):
        T_swa=1
    preds = rescale_temp(swa["predictions"], T_swa)
    swa_temp = {"predictions": preds, "targets": swa["targets"]}
else:
    swa_temp = nonedict()

if sgd["predictions"] is not None and sgd_val["predictions"] is not None:
    print("Rescaling SGD temp")
    T_sgd, rescaled_sgd = optimal_temp_scale(sgd_val["predictions"], sgd_val["targets"], max_iter=50, lr=1e-3)
    if np.isnan(T_sgd):
        T_sgd=1
    preds = rescale_temp(sgd["predictions"], T_sgd)
    sgd_temp = {"predictions": preds, "targets": sgd["targets"]}
else:
    sgd_temp = nonedict()

methods = [(sgd, "SGD"), (swa, "SWA"), 
           (swag_diag, "SWAG-Diag"),
           (swag_cov, "SWAG-Cov"), 
           (sgd_drop, "SGD-Drop"), (swa_drop, "SWA-Drop"), 
           (sgd_temp, "SGD-temp"), (swa_temp, "SWA-temp"), 
           (laplace_sgd, "Laplace-SGD"),
           (nonedict(), "Laplace-SWA"),
           (sgd_ens, "SGD-Ens")
           ]

metrics = [(accuracy, "Accuracy"), (ece, "ECE"), (nll, "NLL")]
results = {name: {} for _, name in methods}

for metric, metric_name in metrics:
    print(metric_name)
    for method, name in methods:
        res = metric(method, name, print_name=args.print_names)
        results[name][metric_name] = res
    print()
with open(get_path("results.pkl", no_npz=True), 'wb') as f:
    pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)
