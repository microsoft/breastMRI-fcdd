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


# class FCDD_REF_CNN224(FCDDNet):
#     def __init__(self, in_shape, **kwargs):
#         super().__init__(in_shape, **kwargs)
#         self.conv1 = self._create_conv2d(in_shape[0], 8, 5, bias=self.bias, padding=2)
#         self.bn2d1 = nn.BatchNorm2d(8, eps=1e-04, affine=self.bias)
#         self.pool1 = self._create_maxpool2d(3, 2, 1)  # 8 x 112 x 112
#
#         self.conv2 = self._create_conv2d(8, 32, 5, bias=self.bias, padding=2)
#         self.bn2d2 = nn.BatchNorm2d(32, eps=1e-04, affine=self.bias)
#         self.pool2 = self._create_maxpool2d(3, 2, 1)  # 32 x 56 x 56
#
#         self.conv3 = self._create_conv2d(32, 64, 3, bias=self.bias, padding=1)
#         self.bn2d3 = nn.BatchNorm2d(64, eps=1e-04, affine=self.bias)
#         self.conv4 = self._create_conv2d(64, 128, 3, bias=self.bias, padding=1)
#         self.bn2d4 = nn.BatchNorm2d(128, eps=1e-04, affine=self.bias)
#         self.pool3 = self._create_maxpool2d(3, 2, 1)  # 128 x 28 x 28
#
#         self.conv5 = self._create_conv2d(128, 128, 3, bias=self.bias, padding=1)
#         self.encoder_out_shape = (128, 28, 28)
#         self.conv_final = self._create_conv2d(128, 1, 1, bias=self.bias)
#
#     def forward(self, x, ad=True):
#         x = self.conv1(x)
#         x = F.leaky_relu(self.bn2d1(x))
#         x = self.pool1(x)
#
#         x = self.conv2(x)
#         x = F.leaky_relu(self.bn2d2(x))
#         x = self.pool2(x)
#
#         x = self.conv3(x)
#         x = F.leaky_relu(self.bn2d3(x))
#         x = self.conv4(x)
#         x = F.leaky_relu(self.bn2d4(x))
#         x = self.pool3(x)
#
#         x = self.conv5(x)
#
#         if ad:
#             x = self.conv_final(x)  # n x heads x h' x w'
#
#         return x
#
#
# class FCDD_REF_CNN224_W(FCDDNet):
#     def __init__(self, in_shape, **kwargs):
#         super().__init__(in_shape, **kwargs)
#         self.conv1 = self._create_conv2d(in_shape[0], 32, 5, bias=self.bias, padding=2)
#         self.bn2d1 = nn.BatchNorm2d(32, eps=1e-04, affine=self.bias)
#         self.pool1 = self._create_maxpool2d(3, 2, 1)  # 32 x 112 x 112
#
#         self.conv2 = self._create_conv2d(32, 128, 5, bias=self.bias, padding=2)
#         self.bn2d2 = nn.BatchNorm2d(128, eps=1e-04, affine=self.bias)
#         self.pool2 = self._create_maxpool2d(3, 2, 1)  # 128 x 56 x 56
#
#         self.conv3 = self._create_conv2d(128, 256, 3, bias=self.bias, padding=1)
#         self.bn2d3 = nn.BatchNorm2d(256, eps=1e-04, affine=self.bias)
#         self.conv4 = self._create_conv2d(256, 256, 3, bias=self.bias, padding=1)
#         self.bn2d4 = nn.BatchNorm2d(256, eps=1e-04, affine=self.bias)
#         self.pool3 = self._create_maxpool2d(3, 2, 1)  # 256 x 28 x 28
#
#         self.conv5 = self._create_conv2d(256, 128, 3, bias=self.bias, padding=1)
#         self.encoder_out_shape = (128, 28, 28)
#         self.conv_final = self._create_conv2d(128, 1, 1, bias=self.bias)
#
#     def forward(self, x, ad=True):
#         x = self.conv1(x)
#         x = F.leaky_relu(self.bn2d1(x))
#         x = self.pool1(x)
#
#         x = self.conv2(x)
#         x = F.leaky_relu(self.bn2d2(x))
#         x = self.pool2(x)
#
#         x = self.conv3(x)
#         x = F.leaky_relu(self.bn2d3(x))
#         x = self.conv4(x)
#         x = F.leaky_relu(self.bn2d4(x))
#         x = self.pool3(x)
#
#         x = self.conv5(x)
#
#         if ad:
#             x = self.conv_final(x)  # n x heads x h' x w'
#
#         return x
