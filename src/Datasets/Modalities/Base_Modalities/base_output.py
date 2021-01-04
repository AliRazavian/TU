from abc import ABCMeta
from .base_explicit import Base_Explicit


class Base_Output(Base_Explicit, metaclass=ABCMeta):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def is_output_modality(self):
        return True

    def get_batch_name(self):
        return 'target_%s' % (self.get_name())

    # Output distributions do not decode
    def get_decoder_name(self):
        return self.get_encoder_name()

    def get_classification_loss_name(self):
        return '%s_cls_loss' % (self.get_name())

    def get_classification_loss_cfgs(self):
        return None

    def get_regression_loss_name(self):
        return '%s_reg_loss' % (self.get_name())

    def get_regression_loss_cfgs(self):
        return None

    def has_pseudo_label(self):
        return False
