from torch.nn import init
from torchvision.models import resnet18, resnet50

# from .efficientnet import *
from .preresnet import *
from .xresnet import *

networks = {
    "resnet18": resnet18(),
    "resnet50": resnet50(),
    "preactresnet18": preactresnet18,
    "preactresnet34": preactresnet34,
    "preactresnet50": preactresnet50,
    "preactresnet101": preactresnet101,
    "preactresnet152": preactresnet152,
    "xresnet18": xresnet18,
    "xresnet34": xresnet34,
    "xresnet50": xresnet50,
    "xresnet101": xresnet101,
    "xresnet152": xresnet152,
    # "efficientnet-b0": efficientnet_b0,
    # "efficientnet-b1": efficientnet_b1,
    # "efficientnet-b2": efficientnet_b2,
    # "efficientnet-b3": efficientnet_b3,
    # "efficientnet-b4": efficientnet_b4,
    # "efficientnet-b5": efficientnet_b5,
    # "efficientnet-b6": efficientnet_b6,
    # "efficientnet-b7": efficientnet_b7,
}


def load_network(hparams):
    config = vars(hparams)
    arch_name = config["arch"]
    if arch_name in networks.keys():
        # Initialize network
        print("Initializing {} network...".format(arch_name))
        if arch_name in ["resnet18", "resnet50"]:
            net = networks[arch_name]
            net.fc = nn.Linear(net.fc.in_features, config["n_classes"])
        else:
            net = networks[arch_name](**config)

        # Initialize weights
        if config["weight_init"] != "default":
            init_weights(net, config["weight_init"], config["weight_init_gain"])

        return net
    else:
        raise NotImplementedError(
            "{} is not an available architecture".format(arch_name)
        )


def init_weights(net, init_type="normal", init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__

        # Conv and linear layers
        if hasattr(m, "weight") and (
            classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(
                    "initialization method {} is not implemented".format(init_type)
                )

            # Biases
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

        # Batch norm layers
        elif classname.find("BatchNorm2d") != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print("Initializing weights as {}".format(init_type.upper()))
    net.apply(init_func)
