import numpy as np
import torch

from .classification_loss import Classification_Loss
from global_cfgs import Global_Cfgs
import torch.nn.functional as F


class Bipolar_Margin_Loss(Classification_Loss):
    """
        assuming m as margin:
                  ⎧ max{0,-y  +m}           if t = +1
        𝓛(y,t) =  ⎨ max{0,|y| -m}           if t =  0
                  ⎩ max{0, y  +m}           if t = -1
        This loss function forces hard-to-tell samples to fall into the margin
        while positive and negative samples are lied outside of it. The labels should be
        bipolar in {+1,0,-1}

        m:.3 ≡ snr: 0.64
        m:.5 ≡ snr: 0.73
        m: 1 ≡ snr: 0.88
        m: 2 ≡ snr: 0.98
    """

    def __init__(self, *args, **kwargs):
        if 'pseudo_mask_margin' in kwargs:
            self.pseudo_mask_margin = kwargs['pseudo_mask_margin']
            del kwargs['pseudo_mask_margin']
        else:
            self.pseudo_mask_margin = Global_Cfgs().get('pseudo_mask_margin', 0.5)
        super().__init__(*args, **kwargs)
        self.margin = -np.log(1. / self.signal_to_noise_ratio - 1) / 2

    def calculate_loss(self, output, target, pos_loss_weight=None, neg_loss_weight=None, mask=None):
        """
        output.shape: [batch_size, no_outputs]
        target.shape: [batch_size] or [batch_size, no_outputs]
        """
        # TODO: Handle multidimensional loss calculation, i.e. [column, ['pos', 'neg']]
        if pos_loss_weight is None and neg_loss_weight is None:
            if len(target.shape) == 1:
                neg_loss_weight = self.loss_weight[0]
                pos_loss_weight = self.loss_weight[1]
            elif len(target.shape) == 2:
                neg_loss_weight = self.loss_weight[:, 0]
                pos_loss_weight = self.loss_weight[:, 1]
            else:
                raise ValueError(f'Invalid loss in {self.get_name()}: {str(self.loss_weight)}')

        assert pos_loss_weight is not None or neg_loss_weight is not None, 'Invalid weight multipliers'

        if mask is None:
            mask = target != self.ignore_index

        positive_mask = target == 1
        boundary_mask = target == 0
        negative_mask = target == -1

        loss = 0
        if len(target.shape) == 1:
            loss += self.__calculate_loss(
                negative_mask=negative_mask,
                positive_mask=positive_mask,
                boundary_mask=boundary_mask,
                output=output,
                pos_loss_weight=pos_loss_weight,
                neg_loss_weight=neg_loss_weight,
                mask=mask,
            )
        else:
            for idx in range(target.shape[1]):
                pw = pos_loss_weight[idx] if isinstance(pos_loss_weight, torch.Tensor) else pos_loss_weight
                nw = neg_loss_weight[idx] if isinstance(neg_loss_weight, torch.Tensor) else neg_loss_weight
                loss += self.__calculate_loss(
                    negative_mask=negative_mask[:, idx],
                    positive_mask=positive_mask[:, idx],
                    boundary_mask=boundary_mask[:, idx],
                    output=output[:, idx],
                    pos_loss_weight=pw,
                    neg_loss_weight=nw,
                    mask=mask[:, idx] if isinstance(mask, torch.Tensor) else mask,
                )

        return loss

    def __calculate_loss(
            self,
            negative_mask,
            positive_mask,
            boundary_mask,
            output,
            pos_loss_weight: float,
            neg_loss_weight: float,
            mask,
    ):
        if mask is not None:
            positive_mask = positive_mask[mask]
            boundary_mask = boundary_mask[mask]
            negative_mask = negative_mask[mask]
            output = output[mask]

        no_negative = negative_mask.float().sum()
        no_boundary = boundary_mask.float().sum()
        no_positive = positive_mask.float().sum()

        loss = 0
        if no_negative > 0:
            loss += neg_loss_weight * (output[negative_mask] + self.margin).clamp(min=0).sum()

        if no_boundary > 0:
            loss += (output[boundary_mask].abs() - self.margin).clamp(min=0).sum()

        if no_positive > 0:
            loss += pos_loss_weight * (-output[positive_mask] + self.margin).clamp(min=0).sum()

        return loss / (no_positive + no_boundary + no_negative + 1e-5)

    def calculate_pi_loss(self, pi_output, target):
        mask = target == self.ignore_index
        if mask.sum() <= 0:
            return 0

        pi_loss = 0
        for i in range(pi_output.shape[0]):
            for j in range(pi_output.shape[0]):
                log_p = F.logsigmoid(pi_output[i, mask, :].clone().view(-1))
                # 1-sigmoid(x) = sigmoid(-x) ==>
                # log(1-sigmoid(x)) = log(sigmoid(-x))
                log_1_minus_p = F.logsigmoid(-pi_output[i, mask, :].clone().view(-1))
                q = F.sigmoid(pi_output[j, mask, :].clone().view(-1))
                log_P = torch.stack([log_p, log_1_minus_p]).t()
                Q = torch.stack([q, 1 - q]).t()
                pi_loss += F.kl_div(input=log_P, target=Q.detach(), size_average=False) / pi_output.shape[1]

        return pi_loss

    def calculate_pseudo_loss(self, pseudo_output, output, target):
        mask = target == self.ignore_index

        if mask.sum() <= 0:
            return 0

        # We recreate a target here that we can use with the regular loss. An alternative could be to use
        # some loss that tries to mimic directly the target output, e.g. using KL-divergence (didn't work when tested
        # early on)
        pseudo_target = output.clone()
        if len(target.shape) == 1 and len(pseudo_target.shape) > 1:
            pseudo_target = pseudo_target.view(-1)

        # We have a boundary but this should be ignored as we only want the
        # strong prediction and hence the mask at the end
        boundary = self.pseudo_mask_margin
        positive_mask = pseudo_target >= boundary
        negative_mask = pseudo_target <= -boundary
        less = pseudo_target < boundary
        more = pseudo_target > -boundary
        boundary_mask = less == more
        pseudo_target[positive_mask] = 1
        pseudo_target[boundary_mask] = 0
        pseudo_target[negative_mask] = -1

        mask[boundary_mask] = True

        return self.calculate_loss(output=pseudo_output,
                                   target=pseudo_target,
                                   pos_loss_weight=0.5,
                                   neg_loss_weight=0.5,
                                   mask=mask)

    def pool_and_reshape_target(self, target):
        # There is some overriding done earlier on that we must cancel :-S
        return target
