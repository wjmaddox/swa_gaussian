# One Hundred Layers Tiramisu
PyTorch implementation of [The One Hundred Layers Tiramisu: Fully Convolutional DenseNets for Semantic Segmentation](https://arxiv.org/pdf/1611.09326).

Tiramisu combines [DensetNet](https://arxiv.org/abs/1608.06993) and [U-Net](https://arxiv.org/abs/1505.04597) for high performance semantic segmentation. In this repository, we attempt to replicate the authors' results on the CamVid dataset.

## Commands

  - train SWAG model

  ```python train.py --data_path [data_path] --model FCDenseNet67 --loss cross_entropy --optimizer SGD --lr_init 1e-2 --batch_size 4 --ft_start 750 --ft_batch_size 1 --epochs 1000 --swa --swa_start=850 --swa_lr=1e-3 --dir [dir] ```

  - train SGD model
  
  ```python train.py --data_path [data_path] --model FCDenseNet67 --loss cross_entropy --optimizer SGD --lr_init 1e-2 --batch_size 4 --ft_start 750 --ft_batch_size 1 --epochs 1000 --dir [dir] ```
  
  - run MC Dropout at test time
  
  ```python eval_ensemble.py --data_path [data_path] --batch_size 4 --method Dropout --loss cross_entropy --N 50 --file [sgd_checkpoint] --save_path [output.npz] ```
  
  - run SWAG at test time
  
  ```python eval_ensemble.py --data_path [data_path] --batch_size 4 --method SWAG --scale=0.5 --loss cross_entropy --N 50 --file [swag_checkpoint] --save_path [output.npz] ```
  
## Dataset

Download

* [CamVid Website](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/)
* [Download](https://github.com/alexgkendall/SegNet-Tutorial/tree/master/CamVid)

Specs

* Training: 367 frames
* Validation: 101 frames
* TestSet: 233 frames
* Dimensions: 360x480
* Classes: 11 (+1 background)

## Architecture

Tiramisu adopts the UNet design with downsampling, bottleneck, and upsampling paths and skip connections. It replaces convolution and max pooling layers with Dense blocks from the DenseNet architecture. Dense blocks contain residual connections like in ResNet except they concatenate, rather than sum, prior feature maps.

![](docs/architecture_paper.png)


**Layers**

![](docs/denseblock.png)

**FCDenseNet103**

![](docs/fcdensenet103_arch.png)

## Our Results

**FCDenseNet67**

**FIX ME**

| Dataset     | Loss  | Accuracy |
| ----------- |:-----:| --------:|
| Validation  | .209  | 92.5     |
| Testset     | .435  | 86.8     |


## Training

**Hyperparameters**

* WeightInitialization = HeUniform
* Optimizer = RMSProp
* LR = .001 with exponential decay of 0.995 after each epoch
* Data Augmentation = Random Crops, Vertical Flips
* ValidationSet with early stopping based on IoU or MeanAccuracy with patience of 100 (50 during finetuning)
* WeightDecay = .0001
* Finetune with full-size images, LR = .0001
* Dropout = 0.2
* BatchNorm "we use current batch stats at training, validation, and test time"

## References and Links

* [bfortuner's repo](https://github.com/bfortuner/pytorch_tiramisu)
* [FastAI Project Thread](http://forums.fast.ai/t/one-hundred-layers-tiramisu/2266)
* [Author's Implementation](https://github.com/SimJeg/FC-DenseNet)
* https://github.com/bamos/densenet.pytorch
* https://github.com/liuzhuang13/DenseNet
