from abc import ABCMeta
from .base_modality import Base_Modality


class Base_Sequence(Base_Modality, metaclass=ABCMeta):
    """
    We refer to any modality with 1D consistency as Sequence
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.consistency = self.get_cfgs('consistency', default='1D')
        assert(self.consistency.lower() == '1D'.lower()),\
            'A sequence consistency should be 1D, instead found %s' % (
                self.consistency)
        self.set_consistency('1D')
        self.num_channels = self.get_cfgs('num_channels', default=-1)
        self.width = self.get_cfgs('width', default=-1)

    def get_tensor_shape(self):
        return [
            self.get_channels(),
            self.get_width(),
        ]

    def get_channels(self):
        return self.num_channels

    def get_width(self):
        return self.width

    def set_channels(self, num_channels):
        self.num_channels = num_channels
        self.modality_cfgs['num_channels'] = num_channels

    def set_width(self, width):
        self.width = width
        self.modality_cfgs['width'] = width

    def update_cfgs(self, cfgs):
        if 'tensor_shape' in cfgs:
            [num_channels, width] = cfgs['tensor_shape']
            self.set_channels(num_channels)
            self.set_width(width)

        if 'num_channels' in cfgs:
            self.set_channels(cfgs['num_channels'])
        if 'width' in cfgs:
            self.set_width(cfgs['width'])
