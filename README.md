# A Simple and Effective Baseline for Bayesian DNNs

Installation:
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

Hessian maximum and minimum eigenvalue command:

```cd experiments/hessian_eigs; python run_hess_eigs.py --dataset CIFAR100 --data_path [data_path] --model PreResNet110 --use_test --file [ckpt] --save_path [output.npz] ```

Gradient covariance experiment:

```cd experiments/grad_cov; python run_grad_cov.py --dataset CIFAR100 --data_path [data_path] --model VGG16 --use_test --epochs=300 --lr_init=0.05 --wd=5e-4 --swa --swa_start 161 --swa_lr=0.01 --grad_cov_start 251 --dir [dir] ```

Segmentation:
  - train SWAG model

  ```python train.py --data_path [data_path] --model FCDenseNet67 --loss cross_entropy --optimizer SGD --lr_init 1e-2 --batch_size 4 --ft_start 750 --ft_batch_size 1 --epochs 1000 --swa --swa_start=850 --swa_lr=1e-3 --dir [dir] ```

  - train SGD model
  
  ```python train.py --data_path [data_path] --model FCDenseNet67 --loss cross_entropy --optimizer SGD --lr_init 1e-2 --batch_size 4 --ft_start 750 --ft_batch_size 1 --epochs 1000 --dir [dir] ```
  
  - run MC Dropout at test time
  
  ```python eval_ensemble.py --data_path [data_path] --batch_size 4 --method Dropout --loss cross_entropy --N 50 --file [sgd_checkpoint] --save_path [output.npz] ```
  
  - run SWAG at test time
  
  ```python eval_ensemble.py --data_path [data_path] --batch_size 4 --method SWAG --scale=0.5 --loss cross_entropy --N 50 --file [swag_checkpoint] --save_path [output.npz] ```

## References for Code Base

Stochastic weight averaging: [Pytorch repo](https://github.com/timgaripov/swa/); most of the base methods and model definitions are built off of this repo.

Model implementations:
  - VGG: https://github.com/pytorch/vision/
  - PreResNet: https://github.com/bearpaw/pytorch-classification
  - WideResNet: https://github.com/meliketoy/wide-resnet.pytorch
  - FCDensenet67: https://github.com/bfortuner/pytorch_tiramisu

Hessian eigenvalue computation: [PyTorch repo](https://github.com/tomgoldstein/loss-landscape)

Segmentation evaluation metrics: [Lasagne repo](https://github.com/SimJeg/FC-DenseNet/blob/master/metrics.py)

## Figures 

<!b ToDo -->