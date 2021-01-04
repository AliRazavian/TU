from abc import ABCMeta
from .base_plane import Base_Plane

import Utils.image_tools as image_tools


class Base_Image(Base_Plane, metaclass=ABCMeta):
    """
    We refer to any modality with 2D consistency as image, although it doesn't
    necessarily have to be an image.
    This class implements basic transformations that can be applied to 2D surfaces
    this include jittering, augmentation, etc
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.get_cfgs('spatial_transform').lower() == 'random'.lower():
            self.get_transformed_image = image_tools.get_random_transformed_image
        if self.get_cfgs('spatial_transform').lower() == 'fix'.lower():
            self.get_transformed_image = image_tools.get_fixed_transformed_image

        self.scale_to = self.get_cfgs('scale_to')
        self.keep_aspect = self.get_cfgs('keep_aspect')
        self.output_size = (self.height, self.width)

        self.colorspace = self.get_cfgs('colorspace')
        if self.colorspace.lower() == 'Gray'.lower():
            self.set_channels(1)
        elif self.colorspace.lower() == 'RGB'.lower():
            self.set_channels(3)
        else:
            raise BaseException('Unknown colorspace %s. Colorspace can only be "Gray" or "RGB"' % (self.colorspace))
