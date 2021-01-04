from .base_output import Base_Output
from .base_distribution import Base_Distribution


class Base_Style(Base_Output, Base_Distribution):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def has_discriminator_loss(self):
        return False

    def get_implicit_modality_cfgs(self):
        return {
            'type': 'implicit',
            'num_channels': self.get_channels(),
            'explicit_modality': self.get_name(),
        }

    def get_discriminator_loss_cfgs(self):
        return {
            'loss_type': 'wGAN_gp',
            'modality_name': self.get_name(),
            'real_name': self.get_batch_name(),
            'fake_name': self.get_encoder_name(),
        }

    def get_discriminator_loss_name(self):
        return '%s_real_fake_disc' % self.get_name()

    def get_default_model_cfgs(self):
        return {
            'model_type': 'One_to_One',
            "neural_net_cfgs": {
                "neural_net_type": "cascade",
                "block_type": "Basic",
                "add_max_pool_after_each_block": False,
                "block_output_cs": [self.get_channels()],
                "block_counts": [1],
                "kernel_sizes": [1],
            }
        }
