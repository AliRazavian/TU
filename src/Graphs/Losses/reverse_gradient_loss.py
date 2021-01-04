import torch
import torch.nn.functional as F

from .base_divergent_loss import Base_Divergent_Loss
from Graphs.Models.Blocks.reverse_gradient import Reverse_Gradient


class Reverse_Gradient_Discriminator_Loss(Base_Divergent_Loss):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_loss = 0
        self.reverse_gradient = Reverse_Gradient()

    # This model really doesn't need a discriminator update
    def update_discriminator(self, batch):
        pass

    def calculate_loss(self, real_data, fake_data):
        real_response = self.neural_net(real_data).mean()
        fake_response = self.neural_net(self.reverse_gradient(fake_data)).mean()
        loss = F.binary_cross_entropy_with_logits(torch.stack([real_response, fake_response]),
                                                  torch.FloatTensor([1, 0], device=real_response.device))
        self.last_loss = loss
        return loss

    def step(self):
        # 1/2  .5* ln(.5) = 0.17328679514
        # Only update the discriminator when the discriminator
        #  is fooled by more than 50% of samples
        if self.last_loss > 0.17328679514:
            self.neural_net.step()
