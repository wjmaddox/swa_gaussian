import argparse
import os

parser = argparse.ArgumentParser(description='SGD/SWA training')
parser.add_argument('--cmd', type=str, default=None, required=True, help='ckpt')
parser.add_argument('--path', type=str, default=None, required=True, help='path')
parser.add_argument('--epoch', type=int, default=300, required=False, help='epoch')
parser.add_argument('--drop', action='store_true', help = 'use SWAGDrop method')

args = parser.parse_args()

method = "Dropout" if args.drop else "SGD"
prefix = "srun -p default_gpu --gres=gpu:1 --mem=20G --pty python3 experiments/uncertainty/uncertainty.py " + args.cmd
for split_classes in [0, 1]:
    cmd = prefix + " --method=SGD --split_classes=%d "%(split_classes) + "--file=" + os.path.join(args.path, "checkpoint-%d"%(args.epoch)+".pt")
    if args.drop:
        filename = "unc_drop"
    else:
        cmd += " --N=1"
        filename = "unc_sgd"
    filename += "_split"
    save_path = os.path.join(args.path, filename+str(split_classes)+".npz")
    cmd += " --save_path=" + save_path
    print("Running:", cmd)
    #os.system(cmd)

