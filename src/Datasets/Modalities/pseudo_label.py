from .Base_Modalities.base_number import Base_Number
from .Base_Modalities.base_output import Base_Output
from .Base_Modalities.base_distribution import Base_Distribution


class Pseudo_Label(Base_Number, Base_Output, Base_Distribution):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_channels = self.get_cfgs('num_channels', default=128)

    def has_classification_loss(self):
        return False

    def get_default_model_cfgs(self):
        return {
            'model_type': 'One_to_One',
            'neural_net_cfgs': {
                'neural_net_type': 'fully_connected',
                'num_hidden': 0,
            }
        }

    def get_implicit_modality_cfgs(self):
        return {
            'type': 'implicit',
            'num_channels': max(8, self.num_channels),
            'explicit_modality': self.get_name(),
            'has_reconstruction_loss': False,
        }
