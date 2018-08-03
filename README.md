# Fast Uncertainty Estimates and Bayesian Model Averaging of DNNs

This repo contains a [PyTorch](https://pytorch.org) implementation of our paper [Fast Uncertainty Estimates and Bayesian Model Averaging of DNNs]()
by Wesley Maddox, Timur Garipov, Pavel Izmailov, Dmitry Vetrov, Andrew Gordon Wilson,
which appeared at the [2018 Uncertainty in Deep Learning workshop at UAI](https://sites.google.com/view/udl2018/home?authuser=0).


*I don't know if we should ask people to cite SWA paper :)*


If you use this in your work, please consider citing both it and the related SWA paper, [Averaging Weights Leads to Wider Optima and Better Generalization](https://arxiv.org/abs/1803.05407)
by Pavel Izmailov, Dmitrii Podoprikhin, Timur Garipov, Dmitry Vetrov and Andrew Gordon Wilson.

This repo is based on Timur Garipov's [SWA repo](https://github.com/timgaripov/swa) (from 6/7/2018).

# Dependencies
* [PyTorch](http://pytorch.org/)
* [torchvision](https://github.com/pytorch/vision/)
* [tabulate](https://pypi.python.org/pypi/tabulate/)

# Usage

## Training SWAG

Here we provide code for running SWAG ensembling and conventional SGD training, with examples on the CIFAR-10 and CIFAR-100 datasets.

To run SWAG use the following command:

```bash
python3 run_swag.py --dir=<DIR> \
                 --dataset=<DATASET> \
                 --data_path=<PATH> \
                 --model=<MODEL> \
                 --epochs=<EPOCHS> \
                 --lr_init=<LR_INIT> \
                 --wd=<WD> \
                 --swa \
                 --swa_start=<SWA_START> \
                 --swa_lr=<SWA_LR>
                 --cov_mat \
                 --split_classes
```

Parameters:

* ```DIR``` &mdash; path to training directory where checkpoints will be stored
* ```DATASET``` &mdash; dataset name [CIFAR10/CIFAR100] (default: CIFAR10)
* ```PATH``` &mdash; path to the data directory
* ```MODEL``` &mdash; DNN model name:
    - VGG16/VGG16BN/VGG19/VGG19BN
    - PreResNet110
    - WideResNet28x10
* ```EPOCHS``` &mdash; number of training epochs (default: 200)
* ```LR_INIT``` &mdash; initial learning rate (default: 0.1)
* ```WD``` &mdash; weight decay (default: 1e-4)
* ```SWA_START``` &mdash; the number of epoch after which SWA will start to average models (default: 161)
* ```SWA_LR``` &mdash; SWA learning rate (default: 0.05)
* ```cov_mat``` &mdash; to store the sample covariance matrix along with SWA estimates, rather than simply the second moments
* ```split_classes``` &mdash; to split CIFAR10 classes and only train on 5 random classes (useful for our uncertainty experiments)


To run conventional SGD training use the following command:
```bash
python3 run_swag.py --dir=<DIR> \
                 --dataset=<DATASET> \
                 --data_path=<PATH> \
                 --model=<MODEL> \
                 --epochs=<EPOCHS> \
                 --lr_init=<LR_INIT> \
                 --wd=<WD> 
```

### Examples

To reproduce the DNN ensembling results from the paper run the following comands (we use same parameters for both CIFAR-10 and CIFAR-100):
```bash
#VGG16
python3 run_swag.py --dir=<DIR> --dataset=CIFAR100 --data_path=<PATH> --model=VGG16 --epochs=300 --lr_init=0.05 --wd=5e-4 # SGD
python3 run_swag.py --dir=<DIR> --dataset=CIFAR100 --data_path=<PATH> --model=VGG16 --epochs=300 --lr_init=0.05 --wd=5e-4 --swa --swa_start=161 --swa_lr=0.01 # SWA 1.5 Budgets
```

## Figure Replications


```bash
# Figure 1
python swag_pathological_example.py
# Note that this saves the plot and shows it.

# Figure 2
#MLP
python run_swag.py --model MLP --dataset toy_regression --data_path None --dir swa_exps/regression_1 --batch_size 20 --epochs 300 --swa --cov_mat --loss MSE --lr_init 0.001 --use_test --no_schedule
python plot_regression_output.py --model MLP --dataset toy_regression --data_path None --dir swa_exps/regression_1 --batch_size 20 --epoch 300 --swa --use_test --cov_mat
#the second command will store output in a plots/ folder
```

See the packaged ipython notebook in `notebooks/Uncertainty.ipynb` to reproduce the figures from the appendix of the paper.

# References
 
 Provided model implementations were adapted from
 * VGG: [github.com/pytorch/vision/](https://github.com/pytorch/vision/)
 * PreResNet: [github.com/bearpaw/pytorch-classification](https://github.com/bearpaw/pytorch-classification)
 * WideResNet: [github.com/meliketoy/wide-resnet.pytorch](https://github.com/meliketoy/wide-resnet.pytorch)

