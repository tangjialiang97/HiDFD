from .resnet_scale import ResNet18s, ResNet34s, ResNet50s, ResNet101s, ResNet152s
from .resnet_cifar import ResNet18, ResNet34
from .resnet_imagenet import resnet34
from .resnet_tiny import ResNet34t as ResNet34_tiny
model_dict = {
    'resnet34_scale': ResNet34s,
    'resnet18_scale': ResNet18s,
    'resnet34_cifar': ResNet34,
    'resnet18_cifar': ResNet18,
    'resnet34_imagenet': resnet34,
    'resnet34_tiny': ResNet34_tiny,
}