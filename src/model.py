import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl 
from pytorch_lightning.metrics.classification import Accuracy
import numpy as np
from collections import OrderedDict

from .networks import load_network
from .utils import get_optimizer, get_scheduler

class Model(pl.LightningModule):
    def __init__(self, hparams):
        super(Model, self).__init__()
        self.hparams = hparams
        self.model = load_network(hparams)
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = F.one_hot(y, num_classes=self.hparams.n_classes).float()
        
        # Apply label smoothing
        if self.hparams.smoothing > 0:
            y = self._smooth(y)

        # Apply mixup
        if self.hparams.training == 'mixup' and np.random.rand(1) <= self.hparams.mix_prob:
            x, y = self._mixup(x, y)

        # Apply cutmix
        if self.hparams.training == 'cutmix' and np.random.rand(1) <= self.hparams.mix_prob:
            x, y = self._cutmix(x, y)

        # Pass through model
        pred = self(x)

        # Calculate loss and accuracy
        loss = self.criterion(pred, y)
        acc = self.train_acc(pred.max(1).indices, y.max(1).indices)
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        self.log('lr', self.optimizer.param_groups[0]['lr'], prog_bar=True)

        return loss


    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = F.one_hot(y, num_classes=self.hparams.n_classes).float()

        # Pass through model
        pred = self(x)

        # Calculate loss and accuracy
        loss = self.criterion(pred, y)
        acc = self.val_acc(pred.max(1).indices, y.max(1).indices)
        self.log('val_loss', loss)
        self.log('val_acc', acc)

        return OrderedDict({
            'val_acc': acc,
            'val_loss': loss,
        })


    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        print(f'Validation Loss: {avg_loss} Validation Accuracy: {avg_acc}')

    def test_step(self, batch, batch_idx):
        x, y = batch
        y = F.one_hot(y, num_classes=self.hparams.n_classes).float()

        # Pass through model
        pred = self(x)

        # Calculate loss and accuracy
        loss = self.criterion(pred, y)
        acc = self.val_acc(pred.max(1).indices, y.max(1).indices)
    
        return OrderedDict({
            'test_acc': acc,
            'test_loss': loss,
        })

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        ###########################################################################

    def configure_optimizers(self):
        optimizer = get_optimizer(self.parameters(), self.hparams)
        self.optimizer = optimizer # Make the optimizer availble to access current lr
        scheduler = get_scheduler(optimizer, self.hparams)
        return [optimizer], [scheduler]

    def _smooth(self, y):
        smoothing = self.hparams.smoothing
        confidence = 1.0 - smoothing # Confidence for target class
        label_shape = y.size()
        other = smoothing / (label_shape[1] - 1) # Confidence for non-target classes

        # Create new smoothed target vector
        smooth_y = torch.empty(size=label_shape, device=self.device)
        smooth_y.fill_(other)
        smooth_y.add_(y*confidence).sub_(y*other)

        return smooth_y

    def _mixup(self, x, y):
        lam = np.random.beta(self.hparams.mix_alpha, self.hparams.mix_alpha)
        indices = np.random.permutation(x.size(0))
        x_mix = x*lam + x[indices]*(1-lam)
        y_mix = y*lam + y[indices]*(1-lam)
        return x_mix, y_mix

    def _cutmix(self, x, y):
        def rand_bbox(size, lam):
            ''' From: https://github.com/clovaai/CutMix-PyTorch/blob/master/train.py '''
            W = size[2]
            H = size[3]
            cut_rat = np.sqrt(1. - lam)
            cut_w = np.int(W * cut_rat)
            cut_h = np.int(H * cut_rat)

            cx = np.random.randint(W)
            cy = np.random.randint(H)

            x1 = np.clip(cx - cut_w // 2, 0, W)
            y1 = np.clip(cy - cut_h // 2, 0, H)
            x2 = np.clip(cx + cut_w // 2, 0, W)
            y2 = np.clip(cy + cut_h // 2, 0, H)

            return x1, y1, x2, y2

        lam = np.random.beta(self.hparams.mix_alpha, self.hparams.mix_alpha)
        indices = np.random.permutation(x.size(0))

        # Perform cutmix
        x1, y1, x2, y2 = rand_bbox(x.size(), lam) # Select a random rectangle
        x[:, :, x1:x2, y1:y2] = x[indices, :, x1:x2, y1:y2] # Replace the cutout section with the other image's pixels

        # Adjust target
        lam = 1 - ((x2 - x1) * (y2 - y1) / (x.size()[-1] * x.size()[-2]))
        y_mix = y*lam + y[indices]*(1-lam)
        return x, y_mix 
