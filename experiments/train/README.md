## Image Classification README

The script `experiments/train/run_swag.py` allows to train SWA, SWAG and SGD models on CIFAR-10 and CIFAR-100. 
The script and the following README are based on [the repo implementing SWA](https://github.com/timgaripov/swa).

To train SWAG use
```
python experiments/train/run_swag.py \
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
    - VGG16/VGG16Drop
    - PreResNet164/PreResNet164Drop
    - WideResNet28x10/WideResNet28x10Drop
* ```EPOCHS``` &mdash; number of training epochs (default: 200)
* ```LR_INIT``` &mdash; initial learning rate (default: 0.1)
* ```WD``` &mdash; weight decay (default: 1e-4)
* ```SWA_START``` &mdash; the number of epoch after which SWA will start to average models (default: 161)
* ```SWA_LR``` &mdash;  SWA learning rate (default: 0.05)
* ```--cov_mat``` &mdash; store covariance matrices with SWAG; default is SWAG-Diagonal. 
* ```--use_test``` &mdash; use test data to evaluate the method; by default validation data is used for evaluation. 
* ```--split_classes``` &mdash; use this flag to train on only 5 of the 10 classes of CIFAR10 (set `SPLIT` to either 0 or 1);

To train SGD models, you can use the same script  without specifying the `--swa`, `--swa_start`, `--swa_lr` and `--cov_mat` flags. Models `VGG16Drop`, `PreResNet164Drop` and `WideResNet28x10Drop` are the same as `VGG16`, `PreResNet164` and `WideResNet28x10` respectively, but with dropout added before each layer. 

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

### Results

Once the models are trained, you can evaluate them with `experiments/uncertainty/uncertainty.py` (see description [here](https://github.com/wjmaddox/private_swa_uncertainties/blob/master/experiments/uncertainty/README.md)).
In the tables below we present the negative log likelihoods (NLL) for SWAG versions and baselines on CIFAR datasets.
Please see the paper for more detailed results.

#### CIFAR100 

| DNN                       |  SGD        | SWA         |SWAG       | SWAG-Diagonal | SWA-Dropout | SWA-Temp |
| ------------------------- |:-----------:|:-----------:|:---------:|:-------------:|:-----------:|:--------:|
| VGG16                     | 1.73 ± 0.01 | 1.28 ± 0.01 | 0.95 ± 0.0 | 1.02 ± 0.0 | 1.19 ± 0.05 | 1.04 ± 0.01 | 
| PreResNet164              | 0.95 ± 0.02 | 0.74 ± 0.03 | 0.71 ± 0.02 | 0.68 ± 0.02 | -         | 0.68 ± 0.02 |
| WideResNet28x10           | 0.80 ± 0.01  | 0.67 ± 0.0 | 0.60 ± 0.0 | 0.62 ± 0.0 | 0.06 ± 0.0 | 0.02 ± 0.00 |

#### CIFAR10

| DNN                       |  SGD        | SWA         |SWAG       | SWAG-Diagonal | SWA-Dropout | SWA-Temp |
| ------------------------- |:-----------:|:-----------:|:---------:|:-------------:|:-----------:|:--------:|
| VGG16                     | 0.33 ± 0.01 | 0.26 ± 0.01 | 0.20 ± 0.0 | 0.22 ± 0.01 | 0.23 ± 0.0 | 0.25 ± 0.02 | 
| PreResNet164              | 0.18 ± 0.0  | 0.15 ± 0.00 | 0.12 ± 0.0 | 0.13 ± 0.0  | 0.13 ± 0.0 | 0.13 ± 0.0 |
| WideResNet28x10           | 0.13 ± 0.0  | 0.11 ± 0.00 | 0.11 ± 0.0 | 0.11 ± 0.0  | 0.11 ± 0.0 | 0.11 ± 0.0 |

