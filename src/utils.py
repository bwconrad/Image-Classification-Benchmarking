from torch.optim import lr_scheduler
import torch.optim as optim
import pprint
from argparse import Namespace
import os
import yaml
import warnings
warnings.filterwarnings("ignore", message="Please also save or load the state of the optimizer when saving or loading the scheduler.")

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
            lr=hparams.lr,
            alpha=hparams.ranger_alpha,
            k=hparams.ranger_k,
            betas=(hparams.beta1, hparams.beta2),
            weight_decay=hparams.weight_decay,
            use_gc=hparams.ranger_gc, 
        )
    else:
        raise NotImplementedError(f'{name} is not an available optimizer')

def get_scheduler(optimizer, hparams):
    name = hparams.schedule

    if name == 'cosine':
        print('Using Cosine LR schedule')
        return {
            'scheduler': lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=hparams.epochs,
            ),
        }
    elif name == 'step':
        print(f'Using Step LR scheule with steps={hparams.steps} and step_size={hparams.step_size}')
        return {
            'scheduler': lr_scheduler.MultiStepLR(
                    optimizer,
                    milestones=hparams.steps,
                    gamma=hparams.step_size,
            ),
        }
    elif name == 'none':
        print('Using no LR schedule')
        return {
            'scheduler': lr_scheduler.LambdaLR(
                    optimizer,
                    lr_lambda=lambda epoch: 1,
            ),
        }
    else:
        raise NotImplementedError(f'{name} is not an available learning rate schedule')

def hparams_from_config(config_path):
    hparams = default_hparams
    
    if os.path.isfile(config_path):
        # Load config
        with open(config_path) as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)

        # Replace defaults with config file values
        for key, value in config.items():
            hparams[key] = value

    # Print the config file contents
    pp = pprint.PrettyPrinter(indent=4)
    print('Loaded hparams:')
    pp.pprint(hparams)
    return Namespace(**hparams)


default_hparams = {
    'resume': None,                         
    'seed': None, 
    'experiment_name': 'test',  
    'data_path': 'data/',  
    'output_path': 'output/',
    'dataset': 'cifar10',
    'size': 0,
    'padding': 4,
    'transforms': 'standard', 
    'include_cutout': False, 
    'randaug_N': 2,
    'randaug_M': 5, 
    'gridmask_r': 0.4,  
    'gridmask_minD': 8,
    'gridmask_maxD': 32,
    'gridmask_rotate': 360, 
    'augmix_depth': 3,
    'augmix_width': 3,
    'augmix_severity': 3,
    'augmix_alpha': 1,
    'cutout_size': 16,
    'cutout_n_holes': 1,
    'arch': 'preactresnet18',
    'weight_init': 'normal', 
    'weight_init_gain': 0.02, 
    'training': 'vanilla',
    'smoothing': 0,
    'mix_alpha': 1, 
    'mix_prob': 1,
    'optimizer': 'sgd',
    'batch_size': 128,
    'workers': 6,
    'lr': 0.1,
    'momentum': 0.9,
    'nesterov': True,
    'weight_decay': 0.0001,
    'beta1': 0.9,
    'beta2': 0.999,
    'ranger_alpha': 0.5,
    'ranger_k': 6,
    'ranger_gc': True,
    'epochs': 200, 
    'schedule': 'none', 
    'steps': [100, 150],
    'step_size': 0.1,
}