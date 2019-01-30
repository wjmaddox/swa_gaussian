## Image Classification README

The scripts in `experiments/train/run_swag.py` allow to train SWA, SWAG and SGD models on CIFAR-10 and CIFAR-100. We list the scripts for reproducing the results from the paper below.

PreResNet164:
```bash
# SWAG, CIFAR100
python3 experiments/train/run_swag.py --data_path=<PATH> --epochs=300 --dataset=CIFAR100 --save_freq=300 \
      --model=PreResNet164 --lr_init=0.1 --wd=3e-4 --swa --swa_start=161 --swa_lr=0.05 --cov_mat --use_test \
      --dir=<DIR>

# SWAG, CIFAR10
python experiments/train/run_swag.py --data_path=<PATH> --epochs=300 --dataset=CIFAR10 --save_freq=300 \  
      --model=PreResNet164 --lr_init=0.1 --wd=3e-4 --swa --swa_start=161 --swa_lr=0.01 --cov_mat --use_test \
      --dir=<DIR>
# SGD
python experiments/train/run_swag.py --data_path=<PATH> --epochs=300 --dataset=CIFAR100 --save_freq=300 \
      --model=PreResNet164 --lr_init=0.1 --wd=3e-4 --use_test --dir=<DIR>
```

WideResNet28x10:
```bash
# SWAG
python experiments/train/run_swag.py --data_path=<PATH> --epochs=300 --dataset=CIFAR100 --save_freq=300 \
      --model=WideResNet28x10 --lr_init=0.1 --wd=5e-4 --swa --swa_start=161 --swa_lr=0.05 --cov_mat --use_test \
      --dir=<DIR>

# SGD
python experiments/train/run_swag.py --data_path=<PATH> --epochs=300 --dataset=CIFAR100 --save_freq=300 \
      --model=WideResNet28x10 --lr_init=0.1 --wd=5e-4 --use_test --dir=<DIR>
```

VGG16:
```bash
# SWAG
python experiments/train/run_swag.py --data_path=<PATH> --epochs=300 --dataset=CIFAR100 --save_freq=300 \
      --model=VGG16 --lr_init=0.05 --wd=5e-4 --swa --swa_start=161 --swa_lr=0.01 --cov_mat --use_test \
      --dir=<DIR>

# SGD
python experiments/train/run_swag.py --data_path=<PATH> --epochs=300 --dataset=CIFAR100 --save_freq=300 \
      --model=VGG16 --lr_init=0.05 --wd=5e-4 --use_test --dir=<DIR>
```
### Explanation of some other options

`--cov_mat` store covariance matrices with SWAG; default is SWAG-Diagonal

`--swa` run SWAG

`--split_classes` to train on only 5 of the 10 classes of CIFAR10 (either 0 or 1); all experiments on this use the same CIFAR10 hyper-parameters

### Table of Results

**from paper**
