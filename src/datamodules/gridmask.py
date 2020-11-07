''' 
From: https://github.com/Jia-Research-Lab/GridMask 
'''

import torch
import numpy as np 
import math
from PIL import Image


class Grid(object):
    def __init__(self, d1, d2, rotate = 1, ratio = 0.5, mode=0, prob=1.):
        self.d1 = d1
        self.d2 = d2
        self.rotate = rotate
        self.ratio = ratio
        self.mode=mode
        self.st_prob = self.prob = prob

    def set_prob(self, epoch, max_epoch):
        self.prob = self.st_prob * min(1, epoch / max_epoch)

    def __call__(self, img):
        if np.random.rand() > self.prob:
            return img
        h = img.size(1)
        w = img.size(2)
        hh = math.ceil((math.sqrt(h*h + w*w)))
    
        d = np.random.randint(self.d1, self.d2)
        
        self.l = math.ceil(d*self.ratio)
        
        mask = np.ones((hh, hh), np.float32)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)
        for i in range(-1, hh//d+1):
            s = d*i + st_h
            t = s+self.l
            s = max(min(s, hh), 0)
            t = max(min(t, hh), 0)
            mask[s:t,:] *= 0
        for i in range(-1, hh//d+1):
            s = d*i + st_w
            t = s+self.l
            s = max(min(s, hh), 0)
            t = max(min(t, hh), 0)
            mask[:,s:t] *= 0
        r = np.random.randint(self.rotate)
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask)
        mask = mask[(hh-h)//2:(hh-h)//2+h, (hh-w)//2:(hh-w)//2+w]

        mask = torch.from_numpy(mask).float()
        mask = mask.expand_as(img)
        img = img * (1-mask) 

        return img

class GridMask(object):
    def __init__(self, d1, d2, rotate = 1, ratio = 0.5, mode=0, prob=1.):
        self.rotate = rotate
        self.ratio = ratio
        self.mode = mode
        self.st_prob = prob
        self.grid = Grid(d1, d2, rotate, ratio, mode, prob)

    def set_prob(self, epoch, max_epoch):
        self.grid.set_prob(epoch, max_epoch)

    def __call__(self, x):
        c,h,w = x.size()
        y = []
        y.append(self.grid(x))
        y = torch.cat(y).view(c,h,w)
        return y