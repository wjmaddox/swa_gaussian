# One Hundred Layers Tiramisu
PyTorch implementation of [The One Hundred Layers Tiramisu: Fully Convolutional DenseNets for Semantic Segmentation](https://arxiv.org/pdf/1611.09326).

Tiramisu combines [DensetNet](https://arxiv.org/abs/1608.06993) and [U-Net](https://arxiv.org/abs/1505.04597) for high performance semantic segmentation. In this repository, we attempt to replicate the authors' results on the CamVid dataset.

## Setup

Requires Anaconda for Python3 installed.

```
conda create --name tiramisu python=3.6
source activate tiramisu
conda install pytorch torchvision -c pytorch
```

The ```train.ipynb``` notebook shows a basic train/test workflow.

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

## Authors' Results

![Authors Results on CamVid](docs/authors_results_table.png)

![Authors Results on CamVid](docs/authors_resuls.png)

## Our Results

**FCDenseNet67**

We trained for 670 epochs (224x224 crops) with 100 epochs fine-tuning (full-size images). The authors mention "global accuracy" of 90.8 for FC-DenseNet67 on Camvid, compared to our 86.8. If we exclude the 'background' class, accuracy increases to ~89%. We think the authors did this, but haven't confirmed. 

| Dataset     | Loss  | Accuracy |
| ----------- |:-----:| --------:|
| Validation  | .209  | 92.5     |
| Testset     | .435  | 86.8     |

![Our Results on CamVid](docs/fcdensenet67_trainin_error.png)

**FCDenseNet103**

We trained for 874 epochs with 50 epochs fine-tuning.

| Dataset     | Loss  | Accuracy |
| ----------- |:-----:| --------:|
| Validation  | .178  | 92.8     |
| Testset     | .441  | 86.6     |

![Our Results on CamVid](docs/tiramisu-103-loss-error.png)

**Predictions**

![Our Results on CamVid](docs/example_output.png)

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

* [Project Thread](http://forums.fast.ai/t/one-hundred-layers-tiramisu/2266)
* [Author's Implementation](https://github.com/SimJeg/FC-DenseNet)
* https://github.com/bamos/densenet.pytorch
* https://github.com/liuzhuang13/DenseNet
