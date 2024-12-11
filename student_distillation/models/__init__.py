from .resnet import resnet38, resnet110, resnet116, resnet14x2, resnet38x2, resnet110x2
from .resnet import resnet8x4, resnet14x4, resnet32x4, resnet38x4, resnet20x4, resnet20, resnet32, resnet56
from .resnet_cifar import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from .resnet_scale import ResNet18s, ResNet34s, ResNet50s, ResNet101s, ResNet152s
from .resnet_tiny import ResNet34t, ResNet18t, ResNet50t, ResNet101t
from .vgg import vgg8_bn, vgg13_bn, vgg16_bn, vgg11_bn
from .vgg_imagenet import vgg13_bn as vgg13_bn_imagenet
from .vgg_imagenet import vgg16_bn as vgg16_bn_imagenet
from .vgg_tiny import vgg8_bnt, vgg13_bnt, vgg16_bnt, vgg11_bnt, vgg19_bnt
from .wrn import wrn_16_2, wrn_40_2
from .wrn_tiny import wrn_40_2t, wrn_16_2t, wrn_40_1t, wrn_16_1t, wrn_28_1t, wrn_28_2t
from .mobilenetv2 import mobile_half, mobile_half_double
from .ShuffleNetv1 import ShuffleV1
from .ShuffleNetv1_tiny import ShuffleV1_tiny
from .ShuffleNetv2 import ShuffleV2, ShuffleV2_1_5
from .ShuffleNetv2_tiny import ShuffleV2_tiny, ShuffleV2_1_5_tiny
from .resnet_imagenet import resnet18, resnet34, resnet50, wide_resnet50_2, resnext50_32x4d
from .resnet_imagenet import wide_resnet10_2, wide_resnet18_2, wide_resnet34_2
from .mobilenetv2_imagenet import mobilenet_v2
from .shuffleNetv2_imagenet import shufflenet_v2_x1_0
from .mobilenetv2_tiny import mobile_half_tiny, mobile_half_double_tiny
model_dict = {
    'wrn16_2_tiny': wrn_16_2t,
    'wrn40_2_tiny': wrn_40_2t,
    'wrn28_2_tiny': wrn_28_2t,
    'wrn28_1_tiny': wrn_28_1t,
    'wrn40_1_tiny': wrn_40_1t,
    'wrn16_1_tiny': wrn_16_1t,
    'resnet101_tiny': ResNet101t,
    'resnet50_tiny': ResNet50t,
    'resnet34_tiny': ResNet34t,
    'resnet18_tiny': ResNet18t,
    'resnet34_scale': ResNet34s,
    'resnet18_scale': ResNet18s,
    'resnet34_cifar': ResNet34,
    'resnet18_cifar': ResNet18,
    'resnet50_cifar': ResNet50,
    'resnet20': resnet20,
    'resnet32': resnet32,
    'resnet56': resnet56,
    'resnet110': resnet110,
    'resnet116': resnet116,
    'resnet14x2': resnet14x2,
    'resnet38x2': resnet38x2,
    'resnet110x2': resnet110x2,
    'resnet8x4': resnet8x4,
    'resnet20x4': resnet20x4,
    'resnet14x4': resnet14x4,
    'resnet32x4': resnet32x4,
    'resnet38x4': resnet38x4,
    'vgg8': vgg8_bn,
    'vgg13': vgg13_bn,
    'vgg11': vgg11_bn,
    'vgg16': vgg16_bn,
    'vgg8_tiny': vgg8_bnt,
    'vgg11_tiny': vgg11_bnt,
    'vgg13_tiny': vgg13_bnt,
    'vgg16_tiny': vgg16_bnt,
    'vgg19_tiny': vgg19_bnt,
    'wrn402': wrn_40_2,
    'wrn162': wrn_16_2,
    'MobileNetV2': mobile_half,
    'MobileNetV2_tiny': mobile_half_tiny,
    'MobileNetV2_1_0_tiny': mobile_half_double_tiny,
    'MobileNetV2_1_0': mobile_half_double,
    'ShuffleV1': ShuffleV1,
    'ShuffleV2': ShuffleV2,
    'ShuffleV2_tiny': ShuffleV2_tiny,
    'ShuffleV1_tiny': ShuffleV1_tiny,
    'ShuffleV2_1_5': ShuffleV2_1_5,
    'ShuffleV2_1_5_tiny': ShuffleV2_1_5_tiny,
    'vgg13_imagenet': vgg13_bn_imagenet,
    'vgg16_imagenet': vgg16_bn_imagenet,

    'ResNet18': resnet18,
    'ResNet34': resnet34,
    'ResNet50': resnet50,
    'resnext50_32x4d': resnext50_32x4d,
    'ResNet10x2': wide_resnet10_2,
    'ResNet18x2': wide_resnet18_2,
    'ResNet34x2': wide_resnet34_2,
    'wrn_50_2': wide_resnet50_2,

    'MobileNetV2_Imagenet': mobilenet_v2,
    'ShuffleV2_Imagenet': shufflenet_v2_x1_0,
}
