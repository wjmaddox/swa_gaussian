## Image Classification README

### Scripts to reproduce the results

PreResNet164:
```bash
# SWAG, CIFAR100
python experiments/train/run_swag.py --dataset=CIFAR100 --data_path=[data_path] --use_test --model=PreResNet164 \
      --epochs=300 --lr_init=0.1 --wd=3e-4 --swa --swa_start=161 --swa_lr=0.05 --cov_mat --dir=[dir]
# SWAG, CIFAR10, only difference is SWA start
python experiments/train/run_swag.py --dataset=CIFAR10 --data_path=[data_path] --use_test --model=PreResNet164 --epochs=300 \
      --lr_init=0.1 --wd=3e-4 --swa --swa_start=126 --swa_lr=0.05 --cov_mat --dir=[dir]

# SGD, CIFAR100
python experiments/train/run_swag.py --dataset=[CIFAR10/CIFAR100] --data_path=[data_path] --model=PreResNet164 --epochs=300 \   
      --lr_init=0.1 --wd=3e-4 --use_test --epochs 200
```
**@pavel: double check the sgd epochs here**

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
