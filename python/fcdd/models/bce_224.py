import os.path as pt

import torch.nn as nn
import torch.nn.functional as F
import torchvision
from fcdd.models.bases import FCDDNet, BaseNet
from torch.hub import load_state_dict_from_url
from fcdd.models.fcdd_cnn_224 import FCDD_CNN224, FCDD_CNN224_VGG, FCDD_CNN224_VGG_NOPT


class VGG_BCE(BaseNet):
    fcdd_cls = FCDD_CNN224_VGG_NOPT  # Not sure to add this or not.

    # VGG_11BN based net with randomly initialized weights (pytorch default).
    def __init__(self, in_shape, **kwargs):
        super().__init__(in_shape, **kwargs)
        assert self.bias, "VGG net is only supported with bias atm!"
        model = torchvision.models.vgg11_bn(False)
        # model.classifier = model.classifier[:-3]
        self.vgg = model
        self.lin = nn.Linear(1000, 1, bias=self.bias)

    def forward(self, x, ad=True):
        x = self.vgg(x)
        x = self.lin(x)
        return x


class VGG_BCE_1000(BaseNet):
    fcdd_cls = FCDD_CNN224_VGG_NOPT  # Not sure to add this or not.

    # VGG_11BN based net with randomly initialized weights (pytorch default).
    def __init__(self, in_shape, **kwargs):
        super().__init__(in_shape, **kwargs)
        assert self.bias, "VGG net is only supported with bias atm!"
        model = torchvision.models.vgg11_bn(False)
        model.classifier = model.classifier[:-3]
        self.vgg = model
        self.lin = nn.Linear(4096, 1, bias=self.bias)

    def forward(self, x, ad=True):
        x = self.vgg(x)
        x = self.lin(x)
        return x


class VGG_BCE_CROP(BaseNet):
    fcdd_cls = FCDD_CNN224_VGG_NOPT  # Not sure to add this or not.

    # VGG_11BN based net with randomly initialized weights (pytorch default).
    def __init__(self, in_shape, **kwargs):
        super().__init__(in_shape, **kwargs)
        assert self.bias, "VGG net is only supported with bias atm!"
        model = torchvision.models.vgg11_bn(False)
        self.core = nn.Sequential(*list(model.children())[0])
        self.features = nn.Sequential(*list(self.core.children())[:-8])
        self.features_n = nn.Sequential(
            self.features,
            # nn.Conv2d(512, 512, 3, 1, 1),
            # nn.BatchNorm2d(512),
            # nn.ReLU(True),
            # nn.Conv2d(512, 512, 3, 1, 1),
            # nn.BatchNorm2d(512),
            # nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.AdaptiveAvgPool2d(7),
            nn.Flatten(1),
        )
        self.lin = nn.Linear(in_features=25088, out_features=1, bias=self.bias)

    def forward(self, x, ad=True):
        x = self.features_n(x)
        x = self.lin(x)
        return x
