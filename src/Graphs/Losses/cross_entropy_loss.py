import torch.nn.functional as F
from .classification_loss import Classification_Loss


class Cross_Entropy_Loss(Classification_Loss):

    def calculate_loss(self, output, target):
        return F.cross_entropy(
            output,
            target,
            weight=self.loss_weight,
            ignore_index=self.ignore_index,
        )

    def calculate_pi_loss(self, pi_output, target):
        mask = target == self.ignore_index
        if mask.sum() <= 0:
            return 0

        pi_loss = 0
        for i in range(pi_output.shape[0]):
            for j in range(pi_output.shape[0]):
                pi_loss += F.kl_div(F.log_softmax(pi_output[i, mask, :].clone(), dim=1),
                                    F.softmax(pi_output[j, mask, :].clone(), dim=1).detach())
        return pi_loss

    def calculate_pseudo_loss(self, pseudo_output, output, target):
        mask = target == self.ignore_index

        if (mask.sum() <= 0):
            return 0

        _, pseudo_labels = output[mask, :].clone().max(dim=1)
        return F.cross_entropy(pseudo_output[mask, :], pseudo_labels)
