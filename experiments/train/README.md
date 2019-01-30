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
# SWAG and SWAG-Dropout, CIFAR100
python experiments/train/run_swag.py --dataset=[CIFAR10/CIFAR100] --data_path=[data_path] --use_test --model=WideResNet28x10 \
      --epochs=300 --lr_init=0.1 --wd=5e-4 --swa --swa_start=161 --swa_lr=0.05 --cov_mat --dir=[dir]

# SGD, Laplace, and Dropout, CIFAR100
python experiments/train/run_swag.py --data_path=[data_path] --dataset=[CIFAR100/CIFAR10] --use_test --model=WideResNet28x10 \
      --epochs=300 --lr_init=0.1 --wd=5e-4 --dir=[dir]
```

VGG16:
```bash
#CIFAR10 SGD
python run_swag.py --dataset=[CIFAR10/100] --data_path=[data_path] --model=VGG16 --epochs=300 --lr_init=0.05 --wd=3e-4 --dir=[dir] --use_test
```
### Explanation of some other options

`--cov_mat` store covariance matrices with SWAG; default is SWAG-Diagonal

`--swa` run SWAG

`--split_classes` to train on only 5 of the 10 classes of CIFAR10 (either 0 or 1); all experiments on this use the same CIFAR10 hyper-parameters

### Table of Results

**from paper**
