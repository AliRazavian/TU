from abc import ABCMeta
from .base_modality import Base_Modality


class Base_Plane(Base_Modality, metaclass=ABCMeta):
    """
    We refer to any modality with 2D consistency as Plane
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.consistency = self.get_cfgs('consistency', default='2D')
        assert(self.consistency.lower() == '2D'.lower()),\
            'A plain consistency should be 2D, instead found %s' % (
                self.consistency)
        self.set_consistency('2D')
        self.num_channels = self.get_cfgs('num_channels', default=-1)
        self.width = self.get_cfgs('width', default=-1)
        self.height = self.get_cfgs('height', default=-1)
        self.depth = self.get_cfgs('depth', default=-1)

    def get_tensor_shape(self):
        return [
            self.get_channels(),
            self.get_width(),
            self.get_height(),
        ]

    def get_channels(self):
        return self.num_channels

    def get_width(self):
        return self.width

    def get_height(self):
        return self.height

    def set_channels(self, num_channels):
        self.num_channels = num_channels
        self.modality_cfgs['num_channels'] = num_channels

    def set_width(self, width):
        self.width = width
        self.modality_cfgs['width'] = width

    def set_height(self, height):
        self.height = height
        self.modality_cfgs['height'] = height

    def update_cfgs(self, cfgs):
        if 'tensor_shape' in cfgs:
            [num_channels, height, width] = cfgs['tensor_shape']
            self.set_channels(num_channels)
            self.set_width(width)
            self.set_height(height)

        if 'num_channels' in cfgs:
            self.set_channels(cfgs['num_channels'])
        if 'width' in cfgs:
            self.set_width(cfgs['width'])
        if 'height' in cfgs:
            self.set_height(cfgs['height'])
