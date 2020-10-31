import numpy as np 
import random
from PIL import Image
from .augments import *

class AugMix:
    def __init__(self, width, depth, severity, alpha, include_cutout=False):
        self.width = width
        self.depth = depth
        self.severity = severity
        self.alpha = alpha
        self.augment_list = augment_list(include_cutout)

    def __call__(self, img):
        weights = np.float32(np.random.dirichlet([self.alpha] * self.width)) # Augmented imgs mixing weights
        augment_images = np.zeros_like(img)
        for i in range(self.width):
            img_copy = img.copy()
            d = np.random.randint(1, self.depth+1) # Number of augs in current chain
            ops = random.choices(self.augment_list, k=d)
            
            # Apply augmentation chain
            for op, minval, maxval in ops:
                val = (float(self.severity) / 30) * float(maxval - minval) + minval
                img_copy = op(img_copy, val)
            augment_images += (weights[i]*img_copy).astype('uint8')
        
        lam = np.float32(np.random.beta(self.alpha, self.alpha))
        mixed = augment_images*lam + img*(1-lam)
        mixed = Image.fromarray(np.uint8(mixed))

        return mixed


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