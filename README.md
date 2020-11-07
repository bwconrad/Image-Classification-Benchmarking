# Image Classification Benchmarking
Codebase for testing new techniques in image classification. Implemented in Pytorch Lightning.

## Included Methods
__Datasets__: [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html), [CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html) 

__Architectures__: [PreAct ResNet](https://github.com/kuangliu/pytorch-cifar) ([Source](https://github.com/vikasverma1077/manifold_mixup)), [XResNet](https://arxiv.org/abs/1812.01187v2) ([Source](https://towardsdatascience.com/xresnet-from-scratch-in-pytorch-e64e309af722)) 

__Augmentations__: [AutoAugment](https://arxiv.org/abs/1805.09501) ([Source](https://github.com/DeepVoltaire/AutoAugment)), [Cutout](https://arxiv.org/abs/1708.04552) ([Source](https://github.com/uoguelph-mlrg/Cutout)), [RandAugment](https://arxiv.org/abs/1909.13719) ([Source](https://github.com/ildoonet/pytorch-randaugment)), [Augmix](https://arxiv.org/abs/1912.02781v2), [Gridmask](https://arxiv.org/abs/2001.04086) ([Source](https://github.com/Jia-Research-Lab/GridMask))

__Regularization__: [Label Smoothing](https://arxiv.org/abs/1512.00567), [Mixup](https://arxiv.org/abs/1710.09412) ([Source](https://github.com/vikasverma1077/manifold_mixup)), [Cutmix](https://arxiv.org/abs/1905.04899) ([Source](https://github.com/clovaai/CutMix-PyTorch)) 

__Learning Rate Schedules__: Step, [Cosine Annealing](https://arxiv.org/abs/1608.03983), [Flat cosine annealing](https://medium.com/@lessw/how-we-beat-the-fastai-leaderboard-score-by-19-77-a-cbb2338fab5c) ([Source](https://github.com/pabloppp/pytorch-tools)) 

__Optimizers__: SGD, [Adam](https://arxiv.org/abs/1412.6980), [Ranger](https://medium.com/@lessw/new-deep-learning-optimizer-ranger-synergistic-combination-of-radam-lookahead-for-the-best-of-2dc83f79a48d) ([Source](https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer)) 

## Results
Results from all experiments can be found in ```results.csv```. 


