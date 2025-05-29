# Copyright (c) 2021 liznerski (original FCDD work)
# Copyright (c) 2025 Microsoft Corporation (FCDD for breast cancer detection)
# Licensed under the MIT License.

from fcdd.training.bases import BaseADTrainer
from torch import Tensor, sigmoid
from torch.nn.functional import binary_cross_entropy_with_logits


class BCETrainer(BaseADTrainer):
    def loss(
        self,
        outs: Tensor,
        ins: Tensor,
        labels: Tensor,
        gtmaps: Tensor = None,
        reduce="mean",
    ):
        """computes the BCE loss"""
        assert reduce in ["mean", "none"]
        if self.objective in ["bce"]:
            loss = self.__bce_loss(outs, ins, labels, gtmaps, reduce)
        else:
            raise NotImplementedError(
                "Objective {} is not defined yet.".format(self.objective)
            )
        return loss

    def __bce_loss(
        self, outs: Tensor, ins: Tensor, labels: Tensor, gtmaps: Tensor, reduce: str
    ):
        if self.net.training:
            loss = binary_cross_entropy_with_logits(
                outs.squeeze(), labels.float().to(outs.device)
            )
        else:
            loss = sigmoid(outs).squeeze()
        return loss.mean() if reduce == "mean" else loss

    def snapshot(self, epochs: int):
        self.logger.snapshot(self.net, self.opt, self.sched, epochs)
