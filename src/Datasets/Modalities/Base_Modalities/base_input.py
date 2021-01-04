from abc import ABCMeta
from .base_explicit import Base_Explicit


class Base_Input(Base_Explicit, metaclass=ABCMeta):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def has_reconstruction_loss(self):
        return True

    def is_input_modality(self):
        return True

    def get_reconstruction_loss_name(self):
        return '%s_reconst' % (self.get_name())

    def get_reconstruction_loss_cfgs(self):
        return None
