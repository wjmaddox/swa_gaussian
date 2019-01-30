## Image Classification README

The scripts in `experiments/train/run_swag.py` allow to train SWA, SWAG and SGD models on CIFAR-10 and CIFAR-100. 

To train SWAG use
```
python experiments/train/run_swag.py
      --dir=<DIR> \
      --dataset=<DATASET> \
      --data_path=<PATH> \
      --model=<MODEL> \
      --epochs=<EPOCHS> \
      --lr_init=<LR_INIT> \
      --wd=<WD> \
      --swa \
      --swa_start=<SWA_START> \
      --swa_lr=<SWA_LR> \
      [--cov_mat \]
      [--use_test \]
      [--split_classes=<SPLIT> \]
```
Parameters:

* ```DIR``` &mdash; path to training directory where checkpoints will be stored
* ```DATASET``` &mdash; dataset name [CIFAR10/CIFAR100] (default: CIFAR10)
* ```PATH``` &mdash; path to the data directory
* ```MODEL``` &mdash; DNN model name:
    - VGG16/VGG16BN/VGG19/VGG19BN
    - PreResNet110/PreResNet164
    - WideResNet28x10
* ```EPOCHS``` &mdash; number of training epochs (default: 200)
* ```LR_INIT``` &mdash; initial learning rate (default: 0.1)
* ```WD``` &mdash; weight decay (default: 1e-4)
* ```SWA_START``` &mdash; the number of epoch after which SWA will start to average models (default: 161)
* ```SWA_LR``` &mdash;  SWA learning rate (default: 0.05)
* ```--cov_mat``` &mdash; store covariance matrices with SWAG; default is SWAG-Diagonal. 
* ```--use_test``` &mdash; use test data to evaluate the method; by default validation data is used for evaluation. 
* ```--split_classes``` &mdash; use this flag to train on only 5 of the 10 classes of CIFAR10 (set `SPLIT` to either 0 or 1);

To train SGD models, you can use the same script  without specifying the `--swa`, `--swa_start`, `--swa_lr` and `--cov_mat` flags.

### Reproducing results from the paper

We list the scripts for reproducing the results from the paper below.

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
