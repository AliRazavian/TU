from .base_modality import Base_Modality


class Base_Implicit(Base_Modality):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def has_reconstruction_loss(self):
        return self.get_cfgs('has_reconstruction_loss', False)

    def is_implicit_modality(self):
        return True

    def get_reconstruction_loss_name(self):
        return '%s_l2_reconst' % self.get_name()

    def has_pseudo_label_loss(self):
        return self.get_cfgs('has_pseudo_label_loss', True)

    def get_reconstruction_loss_cfgs(self):
        return {
            'loss_type': 'l2_loss',
            'modality_name': self.get_name(),
            'relu': True,
            'tensor_shape': self.get_tensor_shape()
        }
