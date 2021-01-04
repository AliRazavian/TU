class ModalityMock:

    def __init__(self, name="fake_modality", tensor_shape=None):
        self.name = name
        self.tensor_shape = tensor_shape

    def get_tensor_shape(self):
        return self.tensor_shape

    def get_name(self):
        return self.name
