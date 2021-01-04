import numpy as np

from abc import ABCMeta

from .base_explicit import Base_Explicit


class Base_Distribution(Base_Explicit, metaclass=ABCMeta):

    def get_item(self, index, num_views=None, transforms=None):
        return {self.get_batch_name(): self.generate_random_sample()}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mean = self.get_cfgs('mean', default=0)
        self.std = self.get_cfgs('std', default=1)

        self.distribution = self.get_cfgs('distribution', default='gaussian')
        if self.distribution.lower() == 'gaussian'.lower():
            self.generate_random_sample = self.generate_gaussian_sample

    def generate_gaussian_sample(self):
        gaussian = self.std * np.random.randn(*self.get_shape()) + self.mean
        return gaussian.astype('float32')

    # TODO add more distributions

    def is_distribution(self):
        return True
