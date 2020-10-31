from torch.nn import init
from .preresnet import *

networks = {
    'preactresnet18': preactresnet18,
    'preactresnet34': preactresnet34,
    'preactresnet50': preactresnet50,
    'preactresnet101': preactresnet101,
    'preactresnet152': preactresnet152
}

def load_network(hparams):
    config = vars(hparams)
    arch_name = config['arch']
    if arch_name in networks.keys():
        print('Initializing {} network...'.format(arch_name))
        net = networks[arch_name](**config)
        init_weights(net, config['weight_init'], config['weight_init_gain'])
        return net   
    else:
        raise NotImplementedError('{} is not an available architecture'.format(arch_name))

def init_weights(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__

        # Conv and linear layers
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method {} is not implemented'.format(init_type))

            # Biases
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

        # Batch norm layers
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)


    print('\tInitializing weights as {}'.format(init_type.upper()))
    net.apply(init_func)