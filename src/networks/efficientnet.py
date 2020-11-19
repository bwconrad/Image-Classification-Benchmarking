from efficientnet_pytorch import EfficientNet

def efficientnet_b0(**kwargs):
    return EfficientNet.from_name('efficientnet-b0', in_channels=kwargs['dims'][0], num_classes=kwargs['n_classes'])

def efficientnet_b1(**kwargs):
    return EfficientNet.from_name('efficientnet-b1', in_channels=kwargs['dims'][0], num_classes=kwargs['n_classes'])

def efficientnet_b2(**kwargs):
    return EfficientNet.from_name('efficientnet-b2', in_channels=kwargs['dims'][0], num_classes=kwargs['n_classes'])

def efficientnet_b3(**kwargs):
    return EfficientNet.from_name('efficientnet-b3', in_channels=kwargs['dims'][0], num_classes=kwargs['n_classes'])

def efficientnet_b4(**kwargs):
    return EfficientNet.from_name('efficientnet-b4', in_channels=kwargs['dims'][0], num_classes=kwargs['n_classes'])

def efficientnet_b5(**kwargs):
    return EfficientNet.from_name('efficientnet-b5', in_channels=kwargs['dims'][0], num_classes=kwargs['n_classes'])

def efficientnet_b6(**kwargs):
    return EfficientNet.from_name('efficientnet-b6', in_channels=kwargs['dims'][0], num_classes=kwargs['n_classes'])

def efficientnet_b7(**kwargs):
    return EfficientNet.from_name('efficientnet-b7', in_channels=kwargs['dims'][0], num_classes=kwargs['n_classes'])

