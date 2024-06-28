from typing import Tuple#
import torch
from fcdd.models.bases import FCDDNet
from fcdd.training.bases import BaseADTrainer
from torch import Tensor


class FCDDRefsTrainer(BaseADTrainer):
    @staticmethod
    def huber(z: Tensor) -> Tensor:
        return z.pow(2).add(1).sqrt().sub(1)

    def loss(self, outs: Tuple[Tensor, Tensor], ins: Tensor, labels: Tensor,
             gtmaps: Tensor = None, refs: Tuple[Tensor, Tensor] = None, reduce='mean'):
        """ computes the FCDD loss """
        assert reduce in ['mean', 'none']
        if gtmaps is not None:
            raise NotImplementedError('FCDDRefs loss that utilizes ground-truth maps is not implemented yet.')
        out_x, out_ref = outs  # FCDD outputs for x (image of interest) and ref (reference image)
        ref_x, ref_ref = refs  # Reference net outputs for x and ref

        loss_x = self.huber(ref_ref[:, :, None, None] - out_x)
        if self.objective == 'fcddrefs_symmetric':
            loss_ref = self.huber(ref_x[:, :, None, None] - out_ref)
        else:
            loss_ref = torch.zeros_like(loss_x)

        if self.net.training and len(set(labels.tolist())) > 1:
            loss_x, loss_ref = self.__supervised_loss(loss_x, loss_ref, labels)
        else:
            loss_x = loss_x.mean(1).unsqueeze(1)
            loss_ref = loss_ref.mean(1).unsqueeze(1)

        loss = loss_x + loss_ref

        return loss.mean() if reduce == 'mean' else loss

    def __supervised_loss(self, loss_x: Tensor, loss_ref: Tensor, labels: Tensor) -> Tuple[Tensor, Tensor]:
        def separate_supervised_loss(loss: Tensor) -> Tensor:
            loss = loss.reshape(labels.size(0), -1).mean(-1)
            norm = loss[labels == 0]
            anom = (-(((1 - (-loss[labels == 1]).exp()) + 1e-31).log()))
            loss[(1-labels).nonzero().squeeze()] = norm
            loss[labels.nonzero().squeeze()] = anom
            return loss

        loss_x = separate_supervised_loss(loss_x)
        if self.objective == 'fcddrefs_symmetric':
            loss_ref = separate_supervised_loss(loss_ref)
        else:
            loss_ref = loss_ref.flatten(1).mean(-1)
        return loss_x, loss_ref

    def snapshot(self, epochs: int):
        self.logger.snapshot(self.net, self.opt, self.sched, epochs)

