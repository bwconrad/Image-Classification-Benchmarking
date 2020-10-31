from torch.optim import lr_scheduler
import torch.optim as optim

from .optim.ranger import Ranger

def get_optimizer(parameters, hparams):
    name = hparams.optimizer

    if name == 'sgd':
        print('Using SGD optimizer')
        return optim.SGD(
            parameters,
            lr=hparams.lr,
            momentum=hparams.momentum,
            weight_decay=hparams.weight_decay, 
            nesterov=hparams.nesterov
        )
    elif name == 'adam':
        print('Using Adam optimizer')
        return optim.Adam(
            parameters,
            lr=hparams.lr,
            betas=(hparams.beta1, hparams.beta2),
            weight_decay=hparams.weight_decay 
        )
    elif name == 'ranger':
        print('Using Ranger optimizer')
        return Ranger(
            parameters,
            lr=lr,
            alpha=hparams.ranger_alpha,
            k=hparams.ranger_k,
            betas=(hparams.beta1, hparams.beta2),
            weight_decay=hparams.weight_decay,
            use_gc=hparams.ranger_gc, 
        )


def get_scheduler(optimizer, hparams):
    name = hparams.schedule

    if name == 'cosine':
        print('Using Cosine LR schedule')
        return {
            'scheduler': lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=hparams.epochs,
                verbose=True
            ),
        }
    elif name == 'step':
        print(f'Using Step LR scheule with steps={hparams.steps} and step_size={hparams.step_size}')
        return {
            'scheduler': lr_scheduler.MultiStepLR(
                    optimizer,
                    milestones=hparams.steps,
                    gamma=hparams.step_size,
                    verbose=True
            ),
        }
