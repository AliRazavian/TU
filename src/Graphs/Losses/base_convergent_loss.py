import time
import torch

from abc import abstractmethod
from .base_loss import Base_Loss


class Base_Convergent_Loss(Base_Loss):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.output_name = self.get_cfgs('output_name')
        self.target_name = self.get_cfgs('target_name')

        self.pi_model = self.get_cfgs('pi_model')
        self.classification = self.get_cfgs('classification')
        self.reconstruction = self.get_cfgs('reconstruction')
        self.regression = self.get_cfgs('regression')

        self.has_pseudo_labels = self.get_cfgs('has_pseudo_labels', False)
        self.pseudo_loss_factor = self.get_cfgs('pseudo_loss_factor', 0.1)
        self.pseudo_output_name = ''
        if (self.output_name.startswith('encoder_')):
            self.pseudo_output_name = self.output_name.replace('encoder_', 'encoder_pseudo_')

    def forward(self, batch):
        start_time = time.time()

        output = self.pool_and_reshape_output(batch[self.output_name], batch['num_views'])
        target = self.pool_and_reshape_target(batch[self.target_name])
        # output.shape = [batch_size, levels] for regular bipolar it is levels=1
        # target.shape = [batch_size]

        pi_output = None
        if isinstance(output, tuple):
            output, pi_output = output

        loss = 0
        pi_loss = 0
        pseudo_loss = 0

        modId = self.modality.get_name()  # Just shorter and easier to read

        if modId not in batch['results']:
            batch['results'][modId] = {
                'output': output,
                'target': target,
            }

        if self.classification or self.reconstruction:
            loss = self.calculate_loss(output, target)

            batch['loss'][self.get_name()] = loss

        if self.regression:
            loss = self.calculate_regression_loss(batch)
            batch['loss'][self.get_name()] = loss

        if loss is not None and loss > 0:
            batch['results'][modId].update({'loss': loss.item()})

        if self.pseudo_output_name in batch:
            pseudo_output = self.pool_and_reshape_output(batch[self.pseudo_output_name], batch['num_views'])
            if isinstance(pseudo_output, tuple):
                pseudo_output, _ = pseudo_output
            batch['results'][modId].update({
                'pseudo_output': pseudo_output,
            })

            regular_pseudo_loss = self.calculate_loss(pseudo_output, target)
            loss += regular_pseudo_loss
            if self.has_pseudo_labels:
                pseudo_loss += self.calculate_pseudo_loss(pseudo_output, output.detach(), target)

            total_pseudo_loss = pseudo_loss + regular_pseudo_loss
            if isinstance(total_pseudo_loss, torch.Tensor):
                batch['results'][modId].update({
                    'pseudo_loss': total_pseudo_loss.item(),
                })

        if self.pi_model and pi_output is not None:
            pi_loss = self.calculate_pi_loss(pi_output, target)
            if pi_loss > 0:
                batch['results'][modId].update({'pi_loss': pi_loss.item()})

        if loss > 0:
            self.analyze_results(batch, loss.item())
        else:
            self.analyze_results(batch, 0.0)

        batch['time']['process'][self.get_name()] = {'start': start_time, 'end': time.time()}

        total_loss = loss + 0.1 * pi_loss + self.pseudo_loss_factor * pseudo_loss
        if total_loss > 10:
            total_loss = 0

        return total_loss

    def analyze_results(self, batch, loss):
        self.modality.analyze_results(batch, loss)

    @abstractmethod
    def calculate_loss(self, output, target):
        return 0

    def calculate_pi_loss(self, pi_output, target):
        return 0

    def calculate_pseudo_loss(self, pseudo_output, output, target):
        return 0

    def calculate_regression_loss(self, batch):
        return 0
