import torchvision
import torch.nn as nn
from . import resnet

def build_model(model_name, num_classes, pretrained):
    print(" [*] pretrained: %s"%pretrained)
    if model_name.startswith('resnet'): # no pretrained
        model = resnet.__dict__[model_name](num_classes)
        #model = model(pretrained=pretrained)
        #if '101' in model_name or '152' in model_name:
        #    model.linear = nn.Linear(512*4, num_classes)
        #else:
        #    model.linear = nn.Linear(512, num_classes)
    elif model_name == "alexnet":
        model = torchvision.models.alexnet(pretrained=pretrained)
    elif model_name.startswith('squeezenet'):
        model = torchvision.models.squeezenet.__dict__[model_name]
        model = model(pretrained=pretrained)
        model.classifier._modules["1"] = nn.Conv2d(512, num_classes, kernel_size=(1, 1))
        model.num_classes = num_classes
    elif model_name.startswith('vgg'):
        model = torchvision.models.vgg.__dict__[model_name]
        model = model(pretrained=pretrained)
        model.classifier[6] = nn.Linear(4096, num_classes)
    elif model_name.startswith('densenet'):
        model = torchvision.models.densenet.__dict__[model_name]
        model = model(num_classes=num_classes, pretrained=pretrained)
    else:
        raise NotImplementedError(" [!] Not implemented model name is given: %s, Please correct the model name or define your model on models directory"%model_name)

    return model
