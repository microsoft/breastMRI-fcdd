# Copyright (c) 2021 liznerski (original FCDD work)
# Copyright (c) 2025 Microsoft Corporation (FCDD for breast cancer detection)
# Licensed under the MIT License.

import torch.nn as nn
from fcdd.models.bases import FCDDNet
from fcdd.models.fcdd_cnn_224 import (
    FCDD_CNN224_VGG_F,
    FCDD_CNN224_VGG,
    FCDD_CNN224_VGG_NOPT,
)
from fcdd.models.shallow_cnn_224 import CNN224_VGG_F, CNN224_VGG, CNN224_VGG_NOPT


class FCDD_REF_CNN224_VGG_NOPT(FCDDNet):
    # VGG_11BN based net with randomly initialized weights (pytorch default).
    def __init__(self, in_shape, **kwargs):
        super().__init__(in_shape, **kwargs)
        assert self.bias, "VGG net is only supported with bias atm!"
        self.model = FCDD_CNN224_VGG_NOPT(in_shape, out_channels=128, **kwargs)
        self.ref_model = CNN224_VGG_NOPT(in_shape, **kwargs)
        self.ref_model.vgg.classifier.add_module("relu", nn.ReLU())
        self.ref_model.vgg.classifier.add_module("dropout", nn.Dropout(p=0.5))
        self.ref_model.vgg.classifier.add_module("fc", nn.Linear(4096, 128))
        self.set_reception(
            **{
                k.replace("img_shape", "in_shape"): v
                for k, v in self.model.reception.items()
            }
        )

    def forward(self, x, ref, ad=True):
        assert ad, "AE for REF not implemented"
        out_x, out_ref = self.model(x), self.model(
            ref
        )  # FCDD outputs for both x (image of interest) and ref (reference image)
        ref_x, ref_ref = self.ref_model(x), self.ref_model(
            ref
        )  # Reference net outputs for both x and ref
        return ((out_x, out_ref), (ref_x, ref_ref))


class FCDD_REF_CNN224_VGG_NOPT_12(FCDDNet):
    # VGG_11BN based net with randomly initialized weights (pytorch default).
    def __init__(self, in_shape, **kwargs):
        super().__init__(in_shape, **kwargs)
        assert self.bias, "VGG net is only supported with bias atm!"
        self.model = FCDD_CNN224_VGG_NOPT(in_shape, out_channels=12, **kwargs)
        self.ref_model = CNN224_VGG_NOPT(in_shape, **kwargs)
        self.ref_model.vgg.classifier.add_module("relu", nn.ReLU())
        self.ref_model.vgg.classifier.add_module("dropout", nn.Dropout(p=0.5))
        self.ref_model.vgg.classifier.add_module("fc", nn.Linear(4096, 12))
        self.set_reception(
            **{
                k.replace("img_shape", "in_shape"): v
                for k, v in self.model.reception.items()
            }
        )

    def forward(self, x, ref, ad=True):
        assert ad, "AE for REF not implemented"
        out_x, out_ref = self.model(x), self.model(
            ref
        )  # FCDD outputs for both x (image of interest) and ref (reference image)
        ref_x, ref_ref = self.ref_model(x), self.ref_model(
            ref
        )  # Reference net outputs for both x and ref
        return ((out_x, out_ref), (ref_x, ref_ref))


class FCDD_REF_CNN224_VGG_NOPT_1(FCDDNet):
    # VGG_11BN based net with randomly initialized weights (pytorch default).
    def __init__(self, in_shape, **kwargs):
        super().__init__(in_shape, **kwargs)
        assert self.bias, "VGG net is only supported with bias atm!"
        self.model = FCDD_CNN224_VGG_NOPT(in_shape, out_channels=1, **kwargs)
        self.ref_model = CNN224_VGG_NOPT(in_shape, **kwargs)
        self.ref_model.vgg.classifier.add_module("relu", nn.ReLU())
        self.ref_model.vgg.classifier.add_module("dropout", nn.Dropout(p=0.5))
        self.ref_model.vgg.classifier.add_module("fc", nn.Linear(4096, 1))
        self.set_reception(
            **{
                k.replace("img_shape", "in_shape"): v
                for k, v in self.model.reception.items()
            }
        )

    def forward(self, x, ref, ad=True):
        assert ad, "AE for REF not implemented"
        out_x, out_ref = self.model(x), self.model(
            ref
        )  # FCDD outputs for both x (image of interest) and ref (reference image)
        ref_x, ref_ref = self.ref_model(x), self.ref_model(
            ref
        )  # Reference net outputs for both x and ref
        return ((out_x, out_ref), (ref_x, ref_ref))


class FCDD_REF_CNN224_VGG_NOPT_4(FCDDNet):
    # VGG_11BN based net with randomly initialized weights (pytorch default).
    def __init__(self, in_shape, **kwargs):
        super().__init__(in_shape, **kwargs)
        assert self.bias, "VGG net is only supported with bias atm!"
        self.model = FCDD_CNN224_VGG_NOPT(in_shape, out_channels=4, **kwargs)
        self.ref_model = CNN224_VGG_NOPT(in_shape, **kwargs)
        self.ref_model.vgg.classifier.add_module("relu", nn.ReLU())
        self.ref_model.vgg.classifier.add_module("dropout", nn.Dropout(p=0.5))
        self.ref_model.vgg.classifier.add_module("fc", nn.Linear(4096, 4))
        self.set_reception(
            **{
                k.replace("img_shape", "in_shape"): v
                for k, v in self.model.reception.items()
            }
        )

    def forward(self, x, ref, ad=True):
        assert ad, "AE for REF not implemented"
        out_x, out_ref = self.model(x), self.model(
            ref
        )  # FCDD outputs for both x (image of interest) and ref (reference image)
        ref_x, ref_ref = self.ref_model(x), self.ref_model(
            ref
        )  # Reference net outputs for both x and ref
        return ((out_x, out_ref), (ref_x, ref_ref))


class FCDD_REF_CNN224_VGG(FCDDNet):
    # VGG_11BN based net with most of the VGG layers having weights pretrained on the ImageNet classification task.
    def __init__(self, in_shape, **kwargs):
        super().__init__(in_shape, **kwargs)
        assert self.bias, "VGG net is only supported with bias atm!"
        self.model = FCDD_CNN224_VGG(in_shape, out_channels=128, **kwargs)
        self.ref_model = CNN224_VGG(in_shape, **kwargs)
        # self.ref_model.vgg.classifier.append(nn.ReLU())
        # self.ref_model.vgg.classifier.append(nn.Dropout(p=0.5))
        # self.ref_model.vgg.classifier.append(nn.Linear(4096, 128))
        self.ref_model.vgg.classifier.add_module("relu", nn.ReLU())
        self.ref_model.vgg.classifier.add_module("dropout", nn.Dropout(p=0.5))
        self.ref_model.vgg.classifier.add_module("fc", nn.Linear(4096, 128))
        self.set_reception(
            **{
                k.replace("img_shape", "in_shape"): v
                for k, v in self.model.reception.items()
            }
        )

    def forward(self, x, ref, ad=True):
        assert ad, "AE for REF not implemented"
        out_x, out_ref = self.model(x), self.model(
            ref
        )  # FCDD outputs for both x (image of interest) and ref (reference image)
        ref_x, ref_ref = self.ref_model(x), self.ref_model(
            ref
        )  # Reference net outputs for both x and ref
        return ((out_x, out_ref), (ref_x, ref_ref))


class FCDD_REF_CNN224_VGG_F(FCDD_REF_CNN224_VGG):
    # VGG_11BN based net with most of the VGG layers having weights pretrained on the ImageNet classification task.
    # Additionally, these weights get frozen, i.e., the weights will not get updated during training.
    def __init__(self, in_shape, **kwargs):
        super().__init__(in_shape, **kwargs)
        assert self.bias, "VGG net is only supported with bias atm!"
        self.model = FCDD_CNN224_VGG_F(in_shape, out_channels=128, **kwargs)
        self.ref_model = CNN224_VGG_F(in_shape, **kwargs)
        self.ref_model.vgg.classifier.add_module("relu", nn.ReLU())
        self.ref_model.vgg.classifier.add_module("dropout", nn.Dropout(p=0.5))
        self.ref_model.vgg.classifier.add_module("fc", nn.Linear(4096, 128))
        self.set_reception(
            **{
                k.replace("img_shape", "in_shape"): v
                for k, v in self.model.reception.items()
            }
        )

    def forward(self, x, ref, ad=True):
        assert ad, "AE for REF not implemented"
        out_x, out_ref = self.model(x), self.model(
            ref
        )  # FCDD outputs for both x (image of interest) and ref (reference image)
        ref_x, ref_ref = self.ref_model(x), self.ref_model(
            ref
        )  # Reference net outputs for both x and ref
        return ((out_x, out_ref), (ref_x, ref_ref))