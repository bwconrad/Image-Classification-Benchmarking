# Image Classification Benchmarking
Codebase for testing new techniques in image classification. Simple to configure dataset, network
architecture, data augmentations, regularization and more for easy experimentation with newly
proposed ideas. Implemented in Pytorch Lightning.

## Included Methods
__Datasets__: CIFAR10, CIFAR100 (Fashion MNIST, KMNIST)

__Architectures__: PreAct ResNet, XResNet

__Augmentations__: AutoAugment, Cutout, RandAugment, Augmix, Gridmask

__Regularization__: Label Smoothing, Mixup, Cutmix (Fmix)

__Learning Rate Schedules__: Step, Cosine (Flat cosine, triangular)

__Optimizers__: SGD, Adam, Ranger (RangerLARS)

## Results
Results from all experiments can be found in ```results.csv```. 


