'''
From: https://github.com/ildoonet/pytorch-randaugment/blob/master/RandAugment/augmentations.py
'''

import random
from .augments import *


class RandAugment:
    def __init__(self, n, m, include_cutout=False):
        self.n = n
        self.m = m      # [0, 30]
        self.augment_list = augment_list(include_cutout)

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)
        for op, minval, maxval in ops:
            val = (float(self.m) / 30) * float(maxval - minval) + minval
            img = op(img, val)

        return img


def augment_list(include_cutout=False):  
    l = [
        (AutoContrast, 0, 1),
        (Equalize, 0, 1),
        (Invert, 0, 1),
        (Rotate, 0, 30),
        (Posterize, 0, 4),
        (Solarize, 0, 256),
        (SolarizeAdd, 0, 110),
        (Color, 0.1, 1.9),
        (Contrast, 0.1, 1.9),
        (Brightness, 0.1, 1.9),
        (Sharpness, 0.1, 1.9),
        (ShearX, 0., 0.3),
        (ShearY, 0., 0.3),
        (TranslateXabs, 0., 100),
        (TranslateYabs, 0., 100),
    ]

    if include_cutout:
        l.append((CutoutAbs, 0, 40))

    return l