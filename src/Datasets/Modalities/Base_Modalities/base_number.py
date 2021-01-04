from abc import ABCMeta

from .base_modality import Base_Modality


class Base_Number(Base_Modality, metaclass=ABCMeta):
    """
    We refer to any modality with 1D consistency as Sequence
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.consistency = self.get_cfgs('consistency', default='Number')
        assert(self.consistency.lower() == 'Number'.lower()),\
            'A Number consistency should be Number, instead found %s' % (
                self.consistency)
        self.set_consistency(self.consistency)
        self.num_channels = self.get_cfgs('num_channels', default=-1)

    def get_tensor_shape(self):
        return [self.get_channels()]

    def get_channels(self):
        return self.num_channels

    def set_channels(self, num_channels):
        self.num_channels = num_channels
        self.modality_cfgs['num_channels'] = num_channels

    def update_cfgs(self, cfgs):
        if 'tensor_shape' in cfgs:
            [num_channels] = cfgs['tensor_shape']
            self.set_channels(num_channels)

        if 'num_channels' in cfgs:
            self.set_channels(cfgs['num_channels'])

    def is_regression(self):
        return True
