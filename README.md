# A Simple and Effective Baseline for Bayesian DNNs

This repository contains a PyTorch implementation of Stochastic Weight Averaging-Gaussian (SWAG) from the paper

Fast Uncertainty Estimates and Bayesian Model Averaging of DNNs

by Wesley Maddox, Timur Garipov, Pavel Izmailov, Dmitry Vetrov, and Andrew Gordon Wilson

## Introduction

SWA-Gaussian (SWAG) is a simple method for Bayesian deep learning that can be used as a drop-in replacement for SWA and the standard SGD training procedure (as long as some sort of regularization is used).
The key idea of SWAG is that the SGD iterates act like samples from a Gaussian distribution; SWAG fits this Gaussian distribution by capturing the SWA mean and a covariance matrix.

In this repo, we implement SWAG for image classification with several different architectures on both CIFAR datasets and ImageNet. We also implement SWAG for semantic segmentation on CamVid using our implementation of a FCDenseNet67.
Additionally included are several other experiments on exploring the covariance of the gradients of the SGD iterates, the eigenvalues of the Hessian, and width/PCA decompositions of the SWAG approximate posterior.

**Include width plot here**

Please cite our work if you find it useful:
```
@article{maddoxfast,
  title={Fast Uncertainty Estimates and Bayesian Model Averaging of DNNs},
  author={Maddox, Wesley and Garipov, Timur and Izmailov, Pavel and Vetrov, Dmitry and Wilson, Andrew Gordon},
  journal={UAI Workshop on Uncertainty in Deep Learning},
  year={2018}
}
```

## Installation:

```bash
python setup.py develop
```

See requirements.txt file for requirements that came from our setup. We use Pytorch 1.0.0 in our experiments.

Unless otherwise described, all experiments were run on a single GPU.

## File Structure

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

Hessian maximum and minimum eigenvalue command:

```cd experiments/hessian_eigs; python run_hess_eigs.py --dataset CIFAR100 --data_path [data_path] --model PreResNet110 --use_test --file [ckpt] --save_path [output.npz] ```

Gradient covariance experiment:

```cd experiments/grad_cov; python run_grad_cov.py --dataset CIFAR100 --data_path [data_path] --model VGG16 --use_test --epochs=300 --lr_init=0.05 --wd=5e-4 --swa --swa_start 161 --swa_lr=0.01 --grad_cov_start 251 --dir [dir] ```


## References for Code Base

Stochastic weight averaging: [Pytorch repo](https://github.com/timgaripov/swa/); most of the base methods and model definitions are built off of this repo.

Model implementations:
  - VGG: https://github.com/pytorch/vision/
  - PreResNet: https://github.com/bearpaw/pytorch-classification
  - WideResNet: https://github.com/meliketoy/wide-resnet.pytorch
  - FCDensenet67: https://github.com/bfortuner/pytorch_tiramisu

Hessian eigenvalue computation: [PyTorch repo](https://github.com/tomgoldstein/loss-landscape), but we ultimately ended up using [GPyTorch](https://gpytorch.ai) as it allows calculation of more eigenvalues.

Segmentation evaluation metrics: [Lasagne repo](https://github.com/SimJeg/FC-DenseNet/blob/master/metrics.py)
