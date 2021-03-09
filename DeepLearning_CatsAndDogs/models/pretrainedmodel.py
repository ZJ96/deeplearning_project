import torch
from torchvision import models
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict


model_urls={
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

def get_resnet_50(num_classes, pretrained=True):
    model = models.resnet50(num_classes = num_classes)
    if pretrained:
        state_dict = model_zoo.load_url(model_urls["resnet50"])
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if "fc" not in k:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict,strict = False)
        print("Build {} successfully!".format("resnet50"))
    return model

def get_resnext_50_32x4d(num_classes, pretrained=True):
    model = models.resnext50_32x4d(num_classes = num_classes)
    if pretrained:
        state_dict = model_zoo.load_url(model_urls["resnext50_32x4d"])
        new_state_dict = OrderedDict()
        for k,v in state_dict.items():
            if "fc" not in k:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict,strict = False)
        print("Build {} successfully!".format("resnext50_32x4d"))
    return model

def get_wide_resnet50_2(num_classes, pretrained=True):
    model = models.wide_resnet50_2(num_classes=num_classes)
    if pretrained:
        state_dict = model_zoo.load_url(model_urls["wide_resnet50_2"])
        new_state_dict = OrderedDict()
        for k,v in state_dict.items():
            if "fc" not in k:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict,strict = False)
        print("Build {} successfully!".format("wide_resnet50_2"))
    return model

