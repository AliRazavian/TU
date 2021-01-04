import time

from abc import ABCMeta

from Graphs.Losses.discriminator_factory import Discriminator_Factory
from .base_loss import Base_Loss


class Base_Divergent_Loss(Base_Loss, metaclass=ABCMeta):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.neural_net = self.get_neural_net()
        self.real_name = self.modality.get_batch_name().lower()
        self.fake_name = self.modality.get_decoder_name().lower()

    def get_neural_net(self):
        neural_net = Discriminator_Factory().get_neural_net(head=self.modality,
                                                            optimizer_type=self.get_cfgs('optimizer_type'))
        neural_net.update_learning_rate(self.initial_learning_rate)
        return neural_net

    def forward(self, batch):
        start_time = time.time()
        real_data = self.pool_and_reshape_output(batch[self.real_name])
        fake_data = self.pool_and_reshape_target(batch[self.fake_name])

        loss = self.calculate_loss(real_data, fake_data)
        name = 'loss'
        if (isinstance(loss, tuple)):
            name, loss = loss
        batch['loss'][self.get_name()] = loss
        if not self.modality.get_name() in batch['results']:
            batch['results'][self.modality.get_name()] = {}
        batch['results'][self.modality.get_name()].update({name: loss.item()})

        self.modality.analyze_results(batch, loss.item())
        batch['time']['process'][self.get_name()] = {'start': start_time, 'end': time.time()}

        return loss
