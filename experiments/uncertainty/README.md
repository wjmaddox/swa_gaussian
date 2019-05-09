## Uncertainty experiments README

The script `experiments/uncertainty/uncertainty.py` evaluates the predictions of the models trained with `experiments/train/run_swag.py` on test data:
```
python experiments/uncertainty/uncertainty.py \
      --file=<FILE> \
      --save_path=<SAVEPATH> \
      --dataset=<DATASET> \
      --data_path=<PATH> \
      --model=<MODEL> \
      --method=<METHOD> \
      --scale=<SCALE> \
      --N=<SAMPLES> \
      [--cov_mat \]
      [--use_test \]
      [--use_diag \]
      [--split_classes=<SPLIT> \]
```
Parameters:
* ```<FILE>``` &mdash; path to the checkpoint
* ```<SAVEPATH>``` &mdash; path to save the predictions of the model
* ```<METHOD>``` &mdash; method to evaluate 
      - `SWAG`
      - `KFACLaplace`
      - `SGD`
      - `Dropout`
      - `SWAGDrop`
* ```<SCALE>``` &mdash; scale parameter for re-scaling the posterior approximation; in the experiments we set it equal to `0.5` for `SWAG` and to `1.` for `SWAG-diagonal` and `KFAC-Laplace` (default: `1`)
* ```<SAMPLES>``` &mdash; number of samples from the approximate posterior to use in Bayesian model averaging (default: `30`)

See the README of `experiments/train/run_swag.py` [here](https://github.com/wjmaddox/swa_gaussian/blob/master/experiments/train/README.md) for the description of other parameters.

Below we provide example commands for different methods using WideResNet28x10 on CIFAR100.

```bash
# SGD:
python3 experiments/uncertainty/uncertainty.py  --data_path=<PATH> --dataset=CIFAR100 --model=WideResNet28x10 --use_test \
      --method=SGD --N=1 --file=<FILE> --save_path=<SAVEPATH>

# SWA:
python3 experiments/uncertainty/uncertainty.py  --data_path=<PATH> --dataset=CIFAR100 --model=WideResNet28x10 --use_test \
      --cov_mat --method=SWAG --use_diag --N=1 --scale=0. --file=<FILE> --save_path=<SAVEPATH>

# SWAG
python3 experiments/uncertainty/uncertainty.py  --data_path=<PATH> --dataset=CIFAR100 --model=WideResNet28x10 --use_test \
      --cov_mat --method=SWAG --scale=0.5 --file=<FILE> --save_path=<SAVEPATH>

# SWAG-Diagonal
python3 experiments/uncertainty/uncertainty.py  --data_path=<PATH> --dataset=CIFAR100 --model=WideResNet28x10 --use_test \
      --cov_mat --method=SWAG --use_diag --file=<FILE> --save_path=<SAVEPATH>

# Dropout:
python3 experiments/uncertainty/uncertainty.py  --data_path=<PATH> --dataset=CIFAR100 --model=WideResNet28x10Drop \
      --use_test --method=Dropout --file=<FILE> --save_path=<SAVEPATH>

#SWA-Dropout:
python3 experiments/uncertainty/uncertainty.py  --data_path=<PATH> --dataset=CIFAR100 --model=WideResNet28x10Drop \
      --cov_mat --use_test --method=SWAGDrop --scale=0. --use_diag --file=<FILE> --save_path=<SAVEPATH>
```
