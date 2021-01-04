import torch
import torch.autograd as autograd

from global_cfgs import Global_Cfgs
from .base_divergent_loss import Base_Divergent_Loss


class Wasserstein_GAN_GP_Loss(Base_Divergent_Loss):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.coef = self.get_cfgs('loss_coef', default=1e-2)
        self.update_model_every_step = self.get_cfgs('update_model_every_step', default=5)
        self.dummy_counter = 0

    def update_discriminator(self, real_data, fake_data):
        self.neural_net.zero_grad()
        self.neural_net.train()
        for p in self.neural_net.parameters():
            p.requires_grad = True

        real_response = -self.neural_net(real_data).mean()
        fake_response = self.neural_net(fake_data.detach()).mean()
        gp = self.gradient_penalty(self.neural_net, real_data, fake_data.detach())
        loss = real_response + fake_response + 10 * gp
        return loss

    def update_generator(self, real_data, fake_data):
        for p in self.neural_net.parameters():
            p.requires_grad = False
        loss = -self.neural_net(fake_data).mean()
        return loss

    def calculate_loss(self, real_data, fake_data):
        self.dummy_counter += 1
        if self.training and \
                (self.dummy_counter % self.update_model_every_step != 0):
            return 'discriminator', self.update_discriminator(real_data, fake_data)

        return 'generator', self.update_generator(real_data, fake_data)

    def gradient_penalty(self, neural_net, real_data, fake_data):
        batch_size = min(real_data.shape[0], fake_data.shape[0])
        real_data = real_data[:batch_size, :]
        fake_data = fake_data[:batch_size, :]

        alpha = torch.rand(batch_size, 1)
        alpha = alpha.expand_as(real_data.view(real_data.shape[0], -1)).view_as(real_data)
        if self.use_cuda:
            alpha = alpha.cuda()

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        if Global_Cfgs().get('DEVICE_BACKEND') == 'cuda':
            interpolates = interpolates.cuda()

        interpolates = autograd.Variable(interpolates, requires_grad=True)
        disc_interpolates = neural_net(interpolates)

        gradients = autograd.grad(outputs=disc_interpolates,
                                  inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).cuda()
                                  if self.use_cuda else torch.ones(disc_interpolates.size()),
                                  create_graph=True,
                                  retain_graph=True,
                                  only_inputs=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1)**2).mean() * 10
        return gradient_penalty
