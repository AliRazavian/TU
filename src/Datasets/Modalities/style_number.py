from .Base_Modalities.base_style import Base_Style
from .Base_Modalities.base_number import Base_Number


class Style_Number(Base_Style, Base_Number):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_default_model_cfgs(self):
        return {
            'model_type': 'One_to_One',
            'neural_net_cfgs': {
                'neural_net_type': 'fully_connected',
                'num_hidden': 1,
            }
        }
