import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

from .cutout import Cutout
from .gridmask import GridMask
from .autoaugment import CIFAR10Policy
from .randaugment import RandAugment
from .augmix import AugMix

class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, hparams):
        super(CIFAR10DataModule, self).__init__()
        self.hparams = hparams
        self.data_dir = self.hparams.data_path
        self.train_transforms = get_transforms(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010],
            **vars(hparams)
        )
        self.test_transforms = transforms.Compose([
            transforms.Resize(hparams.size),
            transforms.CenterCrop(hparams.size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])]
        )
        self.dims = (3, 32, 32)
        self.n_classes = 10

    def prepare_data(self):
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage='fit'):
        if stage == 'fit':
            self.cifar_train = CIFAR10(self.data_dir, train=True, transform=self.train_transforms)
            self.cifar_val = CIFAR10(self.data_dir, train=False, transform=self.test_transforms)
        elif stage == 'test':
            self.cifar_test = CIFAR10(self.data_dir, train=False, transform=self.test_transforms)

    def train_dataloader(self):
        return DataLoader(self.cifar_train, batch_size = self.hparams.batch_size, 
                          shuffle=True, num_workers=self.hparams.workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.cifar_val, batch_size = self.hparams.batch_size, 
                          shuffle=False, num_workers=self.hparams.workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.cifar_test, batch_size = self.hparams.batch_size, 
                          shuffle=False, num_workers=self.hparams.workers, pin_memory=True)

    
def get_transforms(mean, std, size, padding, **kwargs):
    name = kwargs['transforms']

    if name == 'standard':
        print("Using Standard data transformations.")
        return transforms.Compose([
            transforms.RandomCrop(size, padding=padding),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])
        
    elif name == 'cutout':
        print("Using Cutout data transformations.")
        return transforms.Compose([
            transforms.RandomCrop(size, padding=padding),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            Cutout(n_holes=kwargs['cutout_n_holes'], length=kwargs['cutout_size']),
            transforms.Normalize(mean, std)])

    elif name == 'autoaugment':
        print("Using AutoAugment data transformations.")
        return transforms.Compose([
            transforms.RandomCrop(size, padding=padding),
            transforms.RandomHorizontalFlip(),
            CIFAR10Policy(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])
    
    elif name == 'autoaugment_cutout':
        print("Using AutoAugment and Cutout data transformations.")
        return transforms.Compose([
            transforms.RandomCrop(size, padding=padding),
            transforms.RandomHorizontalFlip(),
            CIFAR10Policy(),
            transforms.ToTensor(),
            Cutout(n_holes=1, length=16),
            transforms.Normalize(mean, std)])

    elif name == 'randaugment':
        print("Using RandAugment data transformations.")
        return transforms.Compose([
            transforms.RandomCrop(size, padding=padding),
            transforms.RandomHorizontalFlip(),
            RandAugment(n=kwargs['randaug_N'], m=kwargs['randaug_M'], include_cutout=kwargs['include_cutout']),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])

    elif name == 'randaugment_cutout':
        print("Using RandAugment and Cutout data transformations.")
        return transforms.Compose([
            transforms.RandomCrop(size, padding=padding),
            transforms.RandomHorizontalFlip(),
            RandAugment(n=kwargs['randaug_N'], m=kwargs['randaug_M']),
            transforms.ToTensor(),
            Cutout(n_holes=1, length=16),
            transforms.Normalize(mean, std)])

    elif name == 'gridmask':
        print("Using GridMask data transformations.")
        return transforms.Compose([
            transforms.RandomCrop(size, padding=padding),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            GridMask(d1=kwargs['gridmask_minD'], d2=kwargs['gridmask_maxD'], rotate=kwargs['gridmask_rotate'], ratio=kwargs['gridmask_r']),
            transforms.Normalize(mean, std)])    

    elif name == 'autoaugment_gridmask':
        print("Using Autoaugment and GridMask data transformations.")
        return transforms.Compose([
            transforms.RandomCrop(size, padding=padding),
            transforms.RandomHorizontalFlip(),
            CIFAR10Policy(),
            transforms.ToTensor(),
            GridMask(d1=kwargs['gridmask_minD'], d2=kwargs['gridmask_maxD'], rotate=kwargs['gridmask_rotate'], ratio=kwargs['gridmask_r']),
            transforms.Normalize(mean, std)])    
    
    elif name == 'augmix':
        print("Using AugMix data transformations.")
        return transforms.Compose([
            transforms.RandomCrop(size, padding=padding),
            transforms.RandomHorizontalFlip(),
            AugMix(width=kwargs['augmix_width'], depth=kwargs['augmix_depth'], severity=kwargs['augmix_severity'], 
                   alpha=kwargs['augmix_alpha'], include_cutout=kwargs['include_cutout']),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])

    else:
        print("Using no data transformations.")
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])


def get_datamodule(hparams):
    if hparams.dataset in datamodule_dict:
        print(f'Loading {hparams.dataset.upper()} dataset...')
        return datamodule_dict[hparams.dataset](hparams)
    else:
        raise NotImplementedError('{} is not an available dataset'.format(hparams.dataset))


datamodule_dict = {
    'cifar10': CIFAR10DataModule
}