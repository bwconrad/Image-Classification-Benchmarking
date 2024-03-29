from functools import partial

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder

from .augments.augmix import AugMix
from .augments.autoaugment import CIFAR10Policy
from .augments.cutout import Cutout
from .augments.gridmask import GridMask
from .augments.randaugment import RandAugment
from .datasets.medmnist import MedMNIST, PathMNIST


def get_datamodule(hparams):
    if hparams.dataset in datamodule_dict:
        print(f"Loading {hparams.dataset.upper()} dataset...")
        return datamodule_dict[hparams.dataset](hparams)
    else:
        raise NotImplementedError(
            "{} is not an available dataset".format(hparams.dataset)
        )


class ImageNet100DataModule(pl.LightningDataModule):
    def __init__(self, hparams):
        super(ImageNet100DataModule, self).__init__()
        self.params = hparams
        self.data_dir = self.params.data_path
        hparams.size = 224
        self.dims = (3, hparams.size, hparams.size)
        self.n_classes = 100

        self.train_transforms = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.test_transforms = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def setup(self, stage="fit"):
        if stage == "fit":
            self.train_dataset = ImageFolder(
                f"{ self.data_dir }/train/", transform=self.train_transforms
            )
            self.val_dataset = ImageFolder(
                f"{ self.data_dir }/val/", transform=self.test_transforms
            )
        elif stage == "test":
            self.test_dataset = ImageFolder(
                f"{self.data_dir}/val/", transform=self.test_transforms
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.params.batch_size,
            shuffle=True,
            num_workers=self.params.workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.params.batch_size,
            shuffle=False,
            num_workers=self.params.workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.params.batch_size,
            shuffle=False,
            num_workers=self.params.workers,
            pin_memory=True,
        )


class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, hparams):
        super(CIFAR10DataModule, self).__init__()
        self.params = hparams
        self.data_dir = self.params.data_path
        hparams.size = (
            hparams.size if hparams.size != 0 else 32
        )  # If size=0, default to 32
        self.dims = (3, hparams.size, hparams.size)
        self.n_classes = 10

        self.train_transforms = get_transforms(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616], **vars(hparams)
        )
        self.test_transforms = transforms.Compose(
            [
                transforms.Resize(hparams.size),
                transforms.CenterCrop(hparams.size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
                ),
            ]
        )

    def prepare_data(self):
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage="fit"):
        if stage == "fit":
            self.cifar_train = CIFAR10(
                self.data_dir, train=True, transform=self.train_transforms
            )
            self.cifar_val = CIFAR10(
                self.data_dir, train=False, transform=self.test_transforms
            )
        elif stage == "test":
            self.cifar_test = CIFAR10(
                self.data_dir, train=False, transform=self.test_transforms
            )

    def train_dataloader(self):
        return DataLoader(
            self.cifar_train,
            batch_size=self.params.batch_size,
            shuffle=True,
            num_workers=self.params.workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.cifar_val,
            batch_size=self.params.batch_size,
            shuffle=False,
            num_workers=self.params.workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.cifar_test,
            batch_size=self.params.batch_size,
            shuffle=False,
            num_workers=self.params.workers,
            pin_memory=True,
        )


class CIFAR100DataModule(pl.LightningDataModule):
    def __init__(self, hparams):
        super(CIFAR100DataModule, self).__init__()
        self.params = hparams
        self.data_dir = self.params.data_path
        hparams.size = (
            hparams.size if hparams.size != 0 else 32
        )  # If size=0, default to 32
        self.dims = (3, hparams.size, hparams.size)
        self.n_classes = 100

        self.train_transforms = get_transforms(
            mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761], **vars(hparams)
        )
        self.test_transforms = transforms.Compose(
            [
                transforms.Resize(hparams.size),
                transforms.CenterCrop(hparams.size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]
                ),
            ]
        )

    def prepare_data(self):
        CIFAR100(self.data_dir, train=True, download=True)
        CIFAR100(self.data_dir, train=False, download=True)

    def setup(self, stage="fit"):
        if stage == "fit":
            self.cifar_train = CIFAR100(
                self.data_dir, train=True, transform=self.train_transforms
            )
            self.cifar_val = CIFAR100(
                self.data_dir, train=False, transform=self.test_transforms
            )
        elif stage == "test":
            self.cifar_test = CIFAR100(
                self.data_dir, train=False, transform=self.test_transforms
            )

    def train_dataloader(self):
        return DataLoader(
            self.cifar_train,
            batch_size=self.params.batch_size,
            shuffle=True,
            num_workers=self.params.workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.cifar_val,
            batch_size=self.params.batch_size,
            shuffle=False,
            num_workers=self.params.workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.cifar_test,
            batch_size=self.params.batch_size,
            shuffle=False,
            num_workers=self.params.workers,
            pin_memory=True,
        )


class PathMNISTDataModule(pl.LightningDataModule):
    def __init__(self, hparams, n_classes):
        super(PathMNISTDataModule, self).__init__()
        self.params = hparams
        self.data_dir = self.params.data_path
        hparams.size = (
            hparams.size if hparams.size != 0 else 32
        )  # If size=0, default to 32
        self.dims = (3, hparams.size, hparams.size)
        self.n_classes = n_classes

        self.train_transforms = get_transforms(mean=[0.5], std=[0.5], **vars(hparams))
        self.test_transforms = transforms.Compose(
            [
                transforms.Resize(hparams.size),
                transforms.CenterCrop(hparams.size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )

    def prepare_data(self):
        PathMNIST(root=self.data_dir, split="train", download=True)
        PathMNIST(root=self.data_dir, split="val", download=True)
        PathMNIST(root=self.data_dir, split="test", download=True)

    def setup(self, stage="fit"):
        if stage == "fit":
            self.dataset_train = PathMNIST(
                root=self.data_dir, split="train", transform=self.train_transforms
            )
            self.dataset_val = PathMNIST(
                root=self.data_dir, split="val", transform=self.test_transforms
            )
        elif stage == "test":
            self.dataset_test = PathMNIST(
                root=self.data_dir, split="test", transform=self.test_transforms
            )

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            batch_size=self.params.batch_size,
            shuffle=True,
            num_workers=self.params.workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_val,
            batch_size=self.params.batch_size,
            shuffle=False,
            num_workers=self.params.workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset_test,
            batch_size=self.params.batch_size,
            shuffle=False,
            num_workers=self.params.workers,
            pin_memory=True,
        )


def get_transforms(mean, std, size, padding, **kwargs):
    name = kwargs["transforms"]

    if name == "standard":
        print("Using Standard data transformations.")
        return transforms.Compose(
            [
                transforms.RandomCrop(size, padding=padding),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

    elif name == "cutout":
        print("Using Cutout data transformations.")
        return transforms.Compose(
            [
                transforms.RandomCrop(size, padding=padding),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                Cutout(n_holes=kwargs["cutout_n_holes"], length=kwargs["cutout_size"]),
                transforms.Normalize(mean, std),
            ]
        )

    elif name == "autoaugment":
        print("Using AutoAugment data transformations.")
        return transforms.Compose(
            [
                transforms.RandomCrop(size, padding=padding),
                transforms.RandomHorizontalFlip(),
                CIFAR10Policy(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

    elif name == "autoaugment_cutout":
        print("Using AutoAugment and Cutout data transformations.")
        return transforms.Compose(
            [
                transforms.RandomCrop(size, padding=padding),
                transforms.RandomHorizontalFlip(),
                CIFAR10Policy(),
                transforms.ToTensor(),
                Cutout(n_holes=1, length=16),
                transforms.Normalize(mean, std),
            ]
        )

    elif name == "randaugment":
        print("Using RandAugment data transformations.")
        return transforms.Compose(
            [
                transforms.RandomCrop(size, padding=padding),
                transforms.RandomHorizontalFlip(),
                RandAugment(
                    n=kwargs["randaug_N"],
                    m=kwargs["randaug_M"],
                    include_cutout=kwargs["include_cutout"],
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

    elif name == "randaugment_cutout":
        print("Using RandAugment and Cutout data transformations.")
        return transforms.Compose(
            [
                transforms.RandomCrop(size, padding=padding),
                transforms.RandomHorizontalFlip(),
                RandAugment(n=kwargs["randaug_N"], m=kwargs["randaug_M"]),
                transforms.ToTensor(),
                Cutout(n_holes=1, length=16),
                transforms.Normalize(mean, std),
            ]
        )

    elif name == "gridmask":
        print("Using GridMask data transformations.")
        return transforms.Compose(
            [
                transforms.RandomCrop(size, padding=padding),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                GridMask(
                    d1=kwargs["gridmask_minD"],
                    d2=kwargs["gridmask_maxD"],
                    rotate=kwargs["gridmask_rotate"],
                    ratio=kwargs["gridmask_r"],
                ),
                transforms.Normalize(mean, std),
            ]
        )

    elif name == "autoaugment_gridmask":
        print("Using Autoaugment and GridMask data transformations.")
        return transforms.Compose(
            [
                transforms.RandomCrop(size, padding=padding),
                transforms.RandomHorizontalFlip(),
                CIFAR10Policy(),
                transforms.ToTensor(),
                GridMask(
                    d1=kwargs["gridmask_minD"],
                    d2=kwargs["gridmask_maxD"],
                    rotate=kwargs["gridmask_rotate"],
                    ratio=kwargs["gridmask_r"],
                ),
                transforms.Normalize(mean, std),
            ]
        )

    elif name == "augmix":
        print("Using AugMix data transformations.")
        return transforms.Compose(
            [
                transforms.RandomCrop(size, padding=padding),
                transforms.RandomHorizontalFlip(),
                AugMix(
                    width=kwargs["augmix_width"],
                    depth=kwargs["augmix_depth"],
                    severity=kwargs["augmix_severity"],
                    alpha=kwargs["augmix_alpha"],
                    include_cutout=kwargs["include_cutout"],
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

    else:
        print("Using no data transformations.")
        return transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]
        )


datamodule_dict = {
    "cifar10": CIFAR10DataModule,
    "cifar100": CIFAR100DataModule,
    "pathmnist": partial(PathMNISTDataModule, n_classes=9),
    "imagenet100": ImageNet100DataModule,
}
