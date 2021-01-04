import torch.nn.functional as F

from .classification_loss import Classification_Loss


class Hierarchical_BCE_Loss(Classification_Loss):

    def calculate_loss(self, output, target):
        mask = target.sum(dim=1) > 0
        if sum(mask).item() <= 0:
            return 0
        return F.binary_cross_entropy(F.sigmoid(output[mask, :]), target[mask, :])

    def calculate_pi_loss(self, pi_output, target):
        mask = target.sum(dim=1) == 0
        if mask.sum() <= 0:
            return 0

        pi_loss = 0
        for i in range(pi_output.shape[0]):
            for j in range(pi_output.shape[0]):
                pi_loss += F.mse_loss(F.sigmoid(pi_output[i, mask, :].clone()),
                                      F.sigmoid(pi_output[j, mask, :].clone().detach()))
        return pi_loss

    def pool_and_reshape_target(self, target):
        return target

    def calculate_pseudo_loss(self, pseudo_output, output, target):
        mask = target.sum(dim=1) == 0

        if mask.sum() <= 0:
            return 0

        return F.mse_loss(F.sigmoid(pseudo_output[mask, :]), F.sigmoid(output[mask, :].clone().detach()))
