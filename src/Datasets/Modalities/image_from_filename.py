import os
import numpy as np
from functools import lru_cache
import Utils.image_tools as image_tools

from .Base_Modalities.base_input import Base_Input
from .Base_Modalities.base_image import Base_Image
from .Base_Modalities.base_csv import Base_CSV


class Image_from_Filename(Base_Image, Base_Input, Base_CSV):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.im_root = self.get_img_root()
        if self.im_root is None or not os.path.exists(self.im_root):
            raise Exception(f'The dataset {self.dataset_name.lower()} can\'t be found in configs')

        self.hit = 0
        self.miss = 0
        self.miss_ratio = 0
        self.filenames = self.content
        self.informativeness = self.get_runtime_value('informativeness')

    def get_img_root(self):

        img_root = self.get_cfgs('img_root', None)
        if img_root is None:
            img_root = '%s_root' % (self.dataset_name.lower())
        return self.get_cfgs(img_root)

    def has_discriminator_loss(self):
        return True

    def get_implicit_modality_cfgs(self):
        return {
            'type': 'implicit',  # Probably just a constant?
            'explicit_modality': self.get_name(),
            'consistency': '2D'
        }

    def get_item(self, index, num_view):
        filenames = self.get_content(index)
        original_input_size = np.zeros(
            (len(filenames), self.num_jitters, 3),
            dtype='float32',
        )
        input_size = np.zeros(
            (len(filenames), self.num_jitters, 3),
            dtype='float32',
        )

        M = np.zeros(
            (len(filenames), self.num_jitters, 3, 3),
            dtype='float32',
        )

        ims = np.zeros(
            (len(filenames), self.num_jitters, self.num_channels, self.height, self.width),
            dtype='float32',
        )
        for sub_index in range(len(filenames)):
            filename = filenames[sub_index]
            im, original_im_size,  success = self.load_image(filename)

            for j in range(self.num_jitters):
                tr_im, m, im_size = self.get_transformed_image(im, self.output_size, j)
                ims[sub_index, j, :, :, :] = np.transpose(tr_im, (2, 0, 1))
                M[sub_index, j, :, :] = m
                original_input_size[sub_index, j, :] = original_im_size
                input_size[sub_index, j, :] = im_size
            if (success):
                self.hit += 1
            else:
                self.miss += 1

        return {self.get_batch_name(): ims,
                'spatial_transforms': M,
                'original_input_size': original_input_size,
                'input_size': input_size}

    @lru_cache(maxsize=128)
    def load_image(self, filename):
        im_path = os.path.join(self.im_root, filename.lstrip('/'))
        im, success = image_tools.load_image(im_path)
        im, original_input_size = image_tools.fix_dimension_and_normalize(im, self.keep_aspect, self.scale_to, self.colorspace)
        return im, original_input_size, success

    def get_reconstruction_loss_name(self):
        return '%s_l1_reconst' % self.get_name()

    def get_reconstruction_loss_cfgs(self):
        return {
            'loss_type': 'l1_laplacian_pyramid_loss',
            'modality_name': self.get_name(),
            'output_name': self.get_decoder_name(),
            'target_name': self.get_batch_name(),
            'num_channels': self.get_channels(),
            'pyramid_levels': 3,
            'sigma': 1.,
            'kernel_size': 5
        }

    def get_discriminator_loss_name(self):
        return '%s_real_fake_disc' % self.get_name()

    def get_discriminator_loss_cfgs(self):
        return {
            'loss_type': 'wGAN_gp',
            'modality_name': self.get_name(),
            'real_name': self.get_batch_name(),
            'fake_name': self.get_decoder_name(),
        }

    def get_default_model_cfgs(self):
        return {
            'model_type': 'One_to_One',
            'neural_net_cfgs': {
                'neural_net_type': 'cascade',
                'block_type': 'Basic',
                'add_max_pool_after_each_block': True,
                'block_output_cs': [32, 64],
                'block_counts': [1, 1],
                'kernel_sizes': [5, 3]
            }
        }

    def set_runtime_value(self, runtime_value_name, value, indices, sub_indices):
        pass
