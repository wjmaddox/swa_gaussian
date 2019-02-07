# A Simple Baseline for Bayesian Deep Learning

This repository contains a PyTorch implementation of Stochastic Weight Averaging-Gaussian (SWAG) from the paper

*A Simple Baseline for Bayesian Uncertainty in Deep Learning*

by Wesley Maddox, Timur Garipov, Pavel Izmailov, Dmitry Vetrov, and Andrew Gordon Wilson

## Introduction

SWA-Gaussian (SWAG) is a convenient method for uncertainty representation and calibration in Bayesian deep learning.
The key idea of SWAG is that the SGD iterates act like samples from a Gaussian distribution; SWAG fits this Gaussian distribution by capturing the [SWA](https://arxiv.org/abs/1803.05407) mean and a covariance matrix, representing the first two moments of SGD iterates. We use this Gaussian distribution as a posterior over neural network weights, and then perform a Bayesian model average, for uncertainty representation and calibration.

<p align="center">
  <img src="https://user-images.githubusercontent.com/14368801/52224039-09ab0b80-2875-11e9-9c12-c72b88abf4a9.png" width=350>
  <img src="https://user-images.githubusercontent.com/14368801/52224049-0dd72900-2875-11e9-9de8-540ceaae60b3.png" width=350>
</p>


In this repo, we implement SWAG for image classification with several different architectures on both CIFAR datasets and ImageNet. We also implement SWAG for semantic segmentation on CamVid using our implementation of a FCDenseNet67.
We additionally include several other experiments on exploring the covariance of the gradients of the SGD iterates, the eigenvalues of the Hessian, and width/PCA decompositions of the SWAG approximate posterior.

CIFAR10 -> STL10             |  CIFAR100
:-------------------------:|:-------------------------:
![](plots/stl_wrn.jpg)  |  ![](plots/c100_resnet110.jpg)

<<<<<<< HEAD
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
=======
Please cite our work if you find it useful:
```
@article{maddoxfast,
  title={A Simple Baseline for Bayesian Uncertainty in Deep Learning},
  author={Maddox, Wesley and Garipov, Timur and Izmailov, Pavel and Vetrov, Dmitry and Wilson, Andrew Gordon},
  journal={arXiv pre-print},
  year={2019}
}
```

## Installation:

```bash
python setup.py develop
>>>>>>> alt/master
```

See requirements.txt file for requirements that came from our setup. We use Pytorch 1.0.0 in our experiments.

<<<<<<< HEAD
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
=======
Unless otherwise described, all experiments were run on a single GPU.
>>>>>>> alt/master

## File Structure

<<<<<<< HEAD
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
#VGG16 train models
#SGD training
python3 run_swag.py --dir=<DIR> --dataset=CIFAR100 --data_path=<PATH> --model=VGG16 --epochs=300 --lr_init=0.05 --wd=5e-4 # SGD

#SWAG training
python3 run_swag.py --dir=<DIR> --dataset=CIFAR100 --data_path=<PATH> --model=VGG16 --epochs=300 --lr_init=0.05 --wd=5e-4 --swa --swa_start=161 --swa_lr=0.01 # SWA 1.5 Budgets

#SWAG-LR training
python3 run_swag.py --dir=<DIR> --dataset=CIFAR100 --data_path=<PATH> --model=VGG16 --epochs=300 --lr_init=0.05 --wd=5e-4 --swa --swa_start=161 --swa_lr=0.01 --cov_mat # SWA 1.5 Budgets


#model ensembling
#note that the number of samples is hard-coded in
#with SWAG
python swag_ensembles.py --dir=<DIR> --dataset=CIFAR100 --data_path=<PATH> --model=VGG16 --epochs=300 --lr_init=0.05 --wd=5e-4 --swa --swa_start=161
#with SWAG-LR
python swag_ensembles.py --dir=<DIR> --dataset=CIFAR100 --data_path=<PATH> --model=VGG16 --epochs=300 --lr_init=0.05 --wd=5e-4 --swa --swa_start=161 --cov_mat
#with empirical SGD distribution
python sgd_ecdf_ensembles.py --dir=<DIR> --dataset=CIFAR100 --data_path=<PATH> --model=VGG16 --epochs=300 --lr_init=0.05 --wd=5e-4 
```

## Figure Replications
=======
```
.
+-- swag/
|   +-- posteriors/
    |   +-- swag.py (class definition for SWA, SWAG and SWAG-Diag)
    |   +-- laplace.py (class definition for KFAC Laplace)
|   +-- models/ (Folder with all model definitions)
|   +-- utils.py (utility functions)
+-- experiments/
|   +-- train/ (folder containing standard training scripts for non-ImageNet data)
|   +-- imagenet/ (folder containing ImageNet training scripts)
|   +-- grad_cov/ (gradient covariance and optimal learning rate experiments)      

|   +-- hessian_eigs/ (folder for eigenvalues of hessian)

|   +-- segmentation/ (folder containing training scripts for segmentation experiments)
|   +-- uncertainty/ (folder containing scripts and methods for all uncertainty experiments)
|   +-- width/ (folder containing scripts for PCA and SVD of SGD trajectories)
+-- tests/ (folder containing tests for SWAG sampling and SWAG log-likelihood calculation.)
```

### Example Commands

**See experiments/* for particular READMEs**

[Image Classification](experiments/train/README.md)

[Segmentation](experiments/segmentation/README.md)

[Uncertainty](experiments/uncertainty/README.md)

Some other commands are listed here:

*Hessian eigenvalues*

```cd experiments/hessian_eigs; python run_hess_eigs.py --dataset CIFAR100 --data_path [data_path] --model PreResNet110 --use_test --file [ckpt] --save_path [output.npz] ```

*Gradient covariances*

```cd experiments/grad_cov; python run_grad_cov.py --dataset CIFAR100 --data_path [data_path] --model VGG16 --use_test --epochs=300 --lr_init=0.05 --wd=5e-4 --swa --swa_start 161 --swa_lr=0.01 --grad_cov_start 251 --dir [dir] ```

Note that this will output the gradient covariances onto the console, so you ought to write these into a log file and retrieve them afterwards.
>>>>>>> alt/master

## References for Code Base

<<<<<<< HEAD
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
=======
Stochastic weight averaging: [Pytorch repo](https://github.com/timgaripov/swa/); most of the base methods and model definitions are built off of this repo.

Model implementations:
  - VGG: https://github.com/pytorch/vision/
  - PreResNet: https://github.com/bearpaw/pytorch-classification
  - WideResNet: https://github.com/meliketoy/wide-resnet.pytorch
  - FCDensenet67: https://github.com/bfortuner/pytorch_tiramisu

Hessian eigenvalue computation: [PyTorch repo](https://github.com/tomgoldstein/loss-landscape), but we ultimately ended up using [GPyTorch](https://gpytorch.ai) as it allows calculation of more eigenvalues.
>>>>>>> alt/master

Segmentation evaluation metrics: [Lasagne repo](https://github.com/SimJeg/FC-DenseNet/blob/master/metrics.py)
