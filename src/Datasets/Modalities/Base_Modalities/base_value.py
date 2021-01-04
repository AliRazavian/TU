import math
from abc import ABCMeta, abstractmethod

from .base_number import Base_Number
from .base_csv import Base_CSV


class Base_Value(Base_Number, Base_CSV, metaclass=ABCMeta):

    @abstractmethod
    def get_loss_type(self):
        pass

    @abstractmethod
    def collect_statistics(self):
        pass

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.jitter_pool = self.get_cfgs('jitter_pool', default=None)
        self.view_pool = self.get_cfgs('view_pool', default=None)
        self.ignore_index = self.get_cfgs('ignore_index', default=-100)
        self.to_each_view_its_own_label = self.get_cfgs('to_each_view_its_own_label', default=True)
        self.num_channels = len(self.content.columns)

        self.label_stats = {}

        self.prep_content()
        self.collect_statistics()

    def has_regression_loss(self):
        return True

    def prep_content(self):
        pass

    def has_pseudo_label(self):
        return False

    def get_label_dictionary(self):
        pass

    def get_regression_loss_cfgs(self):
        return {
            'loss_type': self.get_loss_type(),
            'modality_name': self.get_name(),
            'output_name': self.get_encoder_name(),
            'target_name': self.get_batch_name(),
            'ignore_index': self.ignore_index,
            'jitter_pool': self.jitter_pool,
            'view_pool': self.view_pool,
            'to_each_view_its_own_label': self.to_each_view_its_own_label,
            'output_shape': self.get_shape()
        }

    def get_default_model_cfgs(self):
        return {
            'model_type': 'One_to_One',
            'neural_net_cfgs': {
                'neural_net_type': 'fully_connected',
                'num_hidden': 1,
            }
        }

    def get_implicit_modality_cfgs(self):
        nc = max(
            self.get_cfgs('min_channel', default=2),
            2**math.ceil(math.log2(self.get_num_classes())),
        )
        return {
            'type': 'implicit',
            'num_channels': nc,
            'explicit_modality': self.get_name(),
            'has_reconstruction_loss': False,
        }
