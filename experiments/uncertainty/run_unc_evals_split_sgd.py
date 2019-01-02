import argparse
import os

parser = argparse.ArgumentParser(description='SGD/SWA training')
parser.add_argument('--cmd', type=str, default=None, required=True, help='ckpt')
parser.add_argument('--path', type=str, default=None, required=True, help='path')
parser.add_argument('--epoch', type=int, default=300, required=False, help='epoch')

args = parser.parse_args()
prefix = "srun -p default_gpu --gres=gpu:1 --mem=20G --pty python3 uncertainty.py " + args.cmd
for split_classes in [0, 1]:
    cmd = prefix + " --method=SGD --split_classes=%d "%(split_classes) + "--file=" + os.path.join(args.path, "checkpoint-%d"%(args.epoch)+".pt")
    filename = "unc_sgd_split"
    cmd += " --N=1"
    save_path = os.path.join(args.path, filename+str(split_classes)+".npz")
    cmd += " --save_path=" + save_path
    print("Running:", cmd)
    os.system(cmd)

