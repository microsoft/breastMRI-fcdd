import os
import os.path as pt
import random
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from fcdd.datasets.bases import GTSubset, TorchvisionDataset, ThreeReturnsDataset
from fcdd.datasets.online_supervisor import OnlineSupervisor
from fcdd.datasets.preprocessing import get_target_label_idx, MultiCompose, ImgTransformWrap
from fcdd.util.logging import Logger
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets.folder import default_loader
from torchvision.transforms.functional import to_pil_image, to_tensor


def extract_custom_classes(datapath: str) -> List[str]:
    dir = os.path.join(datapath, "custom", "test")
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    return classes


class ADImageRefDataset(TorchvisionDataset):
    base_folder = "custom"
    gtm = False

    def __init__(
        self,
        root: str,
        normal_class: int,
        preproc: str,
        nominal_label: int,
        supervise_mode: str,
        noise_mode: str,
        oe_limit: int,
        online_supervision: bool,
        logger: Logger = None,
    ):
        """
        This is a general-purpose implementation for custom datasets with csf files specifying the samples.
        It expects the data being contained in class folders and distinguishes between

        :param root: root directory where data is found.
        :param normal_class: the class considered nominal.
        :param preproc: the kind of preprocessing pipeline.
        :param nominal_label: the label that marks nominal samples in training. The scores in the heatmaps always
            rate label 1, thus usually the nominal label is 0, s.t. the scores are anomaly scores.
        :param supervise_mode: the type of generated artificial anomalies.
            See :meth:`fcdd.datasets.bases.TorchvisionDataset._generate_artificial_anomalies_train_set`.
        :param noise_mode: the type of noise used, see :mod:`fcdd.datasets.noise_mode`.
        :param oe_limit: limits the number of different anomalies in case of Outlier Exposure (defined in noise_mode).
        :param online_supervision: whether to sample anomalies online in each epoch,
            or offline before training (same for all epochs in this case).
        :param logger: logger.
        """
        assert (
            online_supervision
        ), "Artificial anomaly generation for custom datasets needs to be online"
        self.trainpath = pt.join(root, self.base_folder, "train_ref.csv")
        self.testpath = pt.join(root, self.base_folder, "test_ref.csv")
        super().__init__(root, logger=logger)

        self.n_classes = 2  # 0: normal, 1: outlier
        self.raw_shape = (3, 248, 248)
        self.shape = (
            3,
            224,
            224,
        )  # shape of your data samples in channels x height x width after image preprocessing
        self.normal_classes = (0, )
        self.outlier_classes = (1, )
        assert nominal_label in [0, 1]
        self.nominal_label = nominal_label
        self.anomalous_label = 1 if self.nominal_label == 0 else 0

        # precomputed mean and std of your training data
        self.mean, self.std = self.extract_mean_std(self.trainpath, normal_class)

        if preproc in ["", None, "default", "none"]:
            assert self.raw_shape == self.shape, 'in case of no augmentation, raw shape needs to fit net input shape'
            transform = test_transform = MultiCompose([
                transforms.Resize((self.shape[-2], self.shape[-1])),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ])
        elif preproc in ["aug1"]:
            transform = MultiCompose([
                transforms.Resize(self.raw_shape[-1]),
                ImgTransformWrap(transforms.ColorJitter(
                    brightness=0.01, contrast=0.01, saturation=0.01, hue=0.01
                )),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(self.shape[-1]),
                transforms.ToTensor(),
                ImgTransformWrap(transforms.Lambda(lambda x: x + 0.001 * torch.randn_like(x))),
                transforms.Normalize(self.mean, self.std),
            ])
            test_transform = MultiCompose([
                transforms.Resize((self.raw_shape[-1])),
                transforms.CenterCrop(self.shape[-1]),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ])
        elif preproc in ["aug2"]:
            transform = MultiCompose([
                transforms.Resize(self.raw_shape[-1]),
                ImgTransformWrap(transforms.ColorJitter(
                    brightness=0.01, contrast=0.01, saturation=0.01, hue=0.01
                )),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(self.shape[-1]),
                transforms.ToTensor(),
                ImgTransformWrap(transforms.Lambda(lambda x: x + 0.001 * torch.randn_like(x))),
                transforms.Normalize(self.mean, self.std),
            ])
            test_transform = MultiCompose([
                transforms.Resize((self.raw_shape[-1])),
                transforms.CenterCrop(self.shape[-1]),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ])
        #  here you could define other pipelines with augmentations
        else:
            raise ValueError("Preprocessing pipeline {} is not known.".format(preproc))

        self.target_transform = transforms.Lambda(
            lambda x: self.anomalous_label
            if x in self.outlier_classes
            else self.nominal_label
        )
        if supervise_mode not in ["unsupervised", "other"]:
            self.all_transform = OnlineSupervisor(
                self, supervise_mode, noise_mode, oe_limit
            )
        else:
            self.all_transform = None

        self._train_set = DatasetREF(
            self.trainpath,
            supervise_mode,
            self.raw_shape,
            self.nominal_label,
            self.anomalous_label,
            normal_classes=self.normal_classes,
            transform=transform,
            target_transform=self.target_transform,
            all_transform=self.all_transform,
        )
        if supervise_mode == "other":  # (semi)-supervised setting
            self.balance_dataset()
        else:
            self._train_set = Subset(
                self._train_set,
                np.argwhere(
                    (np.asarray(self._train_set.anomaly_labels) == self.nominal_label)
                )
                .flatten()
                .tolist(),
            )

        self._test_set = DatasetREF(
            self.testpath,
            supervise_mode,
            self.raw_shape,
            self.nominal_label,
            self.anomalous_label,
            normal_classes=self.normal_classes,
            transform=test_transform,
            target_transform=self.target_transform,
        )

    def balance_dataset(self, gtm=False):
        nominal_mask = np.asarray(self._train_set.anomaly_labels) == self.nominal_label
        anomaly_mask = np.asarray(self._train_set.anomaly_labels) != self.nominal_label

        if anomaly_mask.sum() == 0:
            return

        CLZ = Subset if not gtm else GTSubset
        self._train_set = (
            CLZ(  # randomly pick n_nominal anomalies for a balanced training set
                self._train_set,
                np.concatenate(
                    [
                        np.argwhere(nominal_mask).flatten().tolist(),
                        np.random.choice(
                            np.argwhere(anomaly_mask).flatten().tolist(),
                            nominal_mask.sum(),
                            replace=True,
                        ),
                    ]
                ),
            )
        )

    def extract_mean_std(
        self, path: str, cls: int
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        transform = MultiCompose([
            transforms.Resize((self.shape[-2], self.shape[-1])),
            transforms.ToTensor(),
        ])
        ds = DatasetREF(  # We may need to double check this
            path,
            "unsupervised",
            self.raw_shape,
            self.nominal_label,
            self.anomalous_label,
            normal_classes=[cls],
            transform=transform,
            target_transform=transforms.Lambda(
                lambda x: self.anomalous_label
                if x in self.outlier_classes
                else self.nominal_label
            ),
        )
        ds = Subset(
            ds,
            np.argwhere(
                np.isin(ds.targets, np.asarray([cls]))
                * np.isin(ds.anomaly_labels, np.asarray([self.nominal_label]))
            )
            .flatten()
            .tolist(),
        )
        loader = DataLoader(
            dataset=ds, batch_size=2, shuffle=False, num_workers=4, pin_memory=False
        )
        all_x = []
        for x, _, x_ref in loader:
            all_x.append(x)
        all_x = torch.cat(all_x)
        return all_x.permute(1, 0, 2, 3).flatten(1).mean(1), all_x.permute(
            1, 0, 2, 3
        ).flatten(1).std(1)


class DatasetREF(ThreeReturnsDataset):
    """Defines dataset from .csv files containing paths of actual and reference images."""

    def __init__(
        self,
        ref_path: str,  # .csv file with absolute paths
        supervise_mode: str,
        raw_shape: Tuple[int, int, int],
        nominal_label: int,  # not used
        anomalous_label: int,  # not used
        transform: MultiCompose = None,
        target_transform=None,
        normal_classes=None,
        all_transform=None,
    ):
        self.ref_df = pd.read_csv(ref_path)
        self.transform = transform
        self.target_transform = target_transform
        self.ref_path = self.ref_df.Reference

        self.anomaly_labels = (
            self.ref_df.Label
        )  # hard coding 0 is normal, 1 is anomalous
        self.normal_classes = normal_classes
        self.all_transform = all_transform  # contains the OnlineSupervisor
        self.supervise_mode = supervise_mode
        self.raw_shape = torch.Size(raw_shape)
        self.samples = list(zip(self.ref_df.Actual, self.ref_df.Label))
        self.targets = self.anomaly_labels.values  # targets and anomaly labels are the same in this case

    def __len__(self):
        return len(self.ref_df)

    def __getitem__(self, index: int) -> Tuple[Tensor, int, Tensor]:
        target = self.anomaly_labels[index]
        ref_img = default_loader(self.ref_path[index])

        if self.target_transform is not None:
            pass  # already applied since we use self.anomaly_labels instead of self.targets

        if self.all_transform is not None:
            replace = random.random() < 0.5
            if replace:
                if self.supervise_mode not in [
                    "malformed_normal",
                    "malformed_normal_gt",
                ]:
                    img, _, target = self.all_transform(
                        torch.empty(self.raw_shape), None, target, replace=replace
                    )
                else:
                    path, _ = self.samples[index]
                    img = to_tensor(default_loader(path)).mul(255).byte()
                    img, _, target = self.all_transform(
                        img, None, target, replace=replace
                    )
                img = to_pil_image(img)
            else:
                path, _ = self.samples[index]
                img = default_loader(path)

        else:
            path, _ = self.samples[index]
            img = default_loader(path)

        if self.transform is not None:
            img, ref_img = self.transform((img, ref_img))

        return img, target, ref_img  # returns actual img, label and reference image
