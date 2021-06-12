"""https://github.com/weiaicunzai/pytorch-cifar100"""

import torch
import models.attention as attention
import models.densenet as densenet
import models.googlenet as googlenet
import models.inceptionv3 as inceptionv3
import models.inceptionv4 as inceptionv4
import models.mobilenet as mobilenet
import models.mobilenetv2 as mobilenetv2
import models.nasnet as nasnet
import models.preactresnet as preactresnet
import models.resnet as resnet
import models.resnext as resnext
import models.rir as rir
import models.senet as senet
import models.shufflenet as shufflenet
import models.shufflenetv2 as shufflenetv2
import models.squeezenet as squeezenet
import models.vgg as vgg
import models.xception as xception
from models.jocor_model import MLPNet, CNN
from models.preact_resnet import PreActResNet18


def get_model(model_name, dataset_name):
    if dataset_name == "mnist":
        grayscale = True
        num_classes = 10
    elif dataset_name == "cifar10":
        grayscale = False
        num_classes = 10
    elif dataset_name == "cifar100":
        grayscale = False
        num_classes = 100
    elif dataset_name == "tiny-imagenet":
        grayscale = False
        num_classes = 200
    elif dataset_name == "clothing1m":
        grayscale = False
        num_classes = 14
    else:
        raise NameError("Invalid dataset")

    if model_name == "jocor_model":
        if dataset_name == "mnist":
            net = MLPNet()
        elif dataset_name == "cifar10":
            net = CNN(n_outputs=10)
        elif dataset_name == "cifar100":
            net = CNN(n_outputs=100)
        elif dataset_name == "tiny-imagenet":
            net = PreActResNet18(num_classes)
        elif dataset_name == "clothing1m":
            model = getattr(resnet, "resnet18")
            net = model(grayscale, num_classes)
            net = torch.nn.DataParallel(net, device_ids=[0, 1, 2, 3])
        return net

    elif model_name.startswith("attention"):
        model = getattr(attention, model_name)
    elif model_name.startswith("densenet"):
        model = getattr(densenet, model_name)
    elif model_name.startswith("googlenet"):
        model = getattr(googlenet, model_name)
    elif model_name.startswith("inceptionv3"):
        model = getattr(inceptionv3, model_name)
    elif model_name.startswith("inception"):
        model = getattr(inceptionv4, model_name)
    elif model_name.startswith("mobilenetv2"):
        model = getattr(mobilenetv2, model_name)
    elif model_name.startswith("mobilenet"):
        model = getattr(mobilenet, model_name)
    elif model_name.startswith("nasnet"):
        model = getattr(nasnet, model_name)
    elif model_name.startswith("preactresnet"):
        model = getattr(preactresnet, model_name)
    elif model_name.startswith("resnet"):
        model = getattr(resnet, model_name)
    elif model_name.startswith("resnext"):
        model = getattr(resnext, model_name)
    elif model_name.startswith("rir"):
        model = getattr(rir, model_name)
    elif model_name.startswith("seresnet"):
        model = getattr(senet, model_name)
    elif model_name.startswith("shufflenetv2"):
        model = getattr(shufflenetv2, model_name)
    elif model_name.startswith("shufflenet"):
        model = getattr(shufflenet, model_name)
    elif model_name.startswith("squeezenet"):
        model = getattr(squeezenet, model_name)
    elif model_name.startswith("vgg"):
        model = getattr(vgg, model_name)
    elif model_name.startswith("xception"):
        model = getattr(xception, model_name)
    else:
        raise NameError("Invalid model")

    net = model(grayscale, num_classes)
    if dataset_name == "clothing1m":
        net = torch.nn.DataParallel(net, device_ids=[0, 1, 2, 3])

    return model(grayscale, num_classes)
