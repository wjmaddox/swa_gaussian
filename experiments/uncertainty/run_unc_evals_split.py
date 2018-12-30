import argparse
import os

parser = argparse.ArgumentParser(description='SGD/SWA training')
parser.add_argument('--cmd', type=str, default=None, required=True, help='ckpt')
parser.add_argument('--path', type=str, default=None, required=True, help='path')
parser.add_argument('--epoch', type=int, default=300, required=False, help='epoch')

args = parser.parse_args()
prefix = "srun -p default_gpu --gres=gpu:1 --mem=20G --pty python3 " + args.cmd
for cov in ["diag", "full", "SWA"]:
    for split_classes in [0, 1]:
        cmd = prefix + " --split_classes=%d "%(split_classes) + "--file=" + os.path.join(args.path, "swag-%d"%(args.epoch)+".pt")
        if cov == "full":
            filename = "unc_cov_split" 
        elif cov=="diag":
            filename = "unc_split"
        else:
            filename = "unc_swa_split"
        if cov == "diag":
            cmd += " --use_diag"
        elif cov == "SWA":
            cmd += " --N=1 --scale=0. --use_diag"
        save_path = os.path.join(args.path, filename+str(split_classes)+".npz")
        cmd += " --save_path=" + save_path
        print("Running:", cmd)
        os.system(cmd)

