import torch.nn.functional as F

from .base_convergent_loss import Base_Convergent_Loss


class Classification_Loss(Base_Convergent_Loss):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.output_shape = self.get_cfgs('output_shape')

    def pool_and_reshape_output(self, output):
        output = output.view([-1, *self.output_shape])
        pi_output = None
        # At this point, the output is in the shape of:
        # [batch_size x num_views x num_jitter x num_classes]

        # compute pi_loss
        if output.shape[2] > 1:
            pi_output = output.permute([2, 0, 1, 3])
            pi_output = pi_output.contiguous().view([pi_output.shape[0], -1, pi_output.shape[3]])
            if (pi_output.shape[2] > 1):
                pi_output = F.softmax(pi_output, dim=2)

        if self.jitter_pool.lower() == 'mean'.lower():
            output = output.mean(dim=2)
        elif self.jitter_pool.lower() == 'max'.lower():
            output, _ = output.max(dim=2)
        # Alight, now it's [batch_size x num_views x num_classes]
        if self.to_each_view_its_own_label:
            output = output.view([-1, self.output_shape[-1]])
            return output, pi_output

        if self.view_pool.lower() == 'mean'.lower():
            output = output.mean(dim=1)
        elif self.view_pool.lower() == 'max'.lower():
            output, _ = output.max(dim=1)

        return output, pi_output

    def pool_and_reshape_target(self, target):
        return target.view(-1)
