from .ModalityMock import ModalityMock


class ExperimentSetMock:

    def __init__(self, name="fake_name"):
        self.name = name
        self.modality = ModalityMock()

    def get_modality(self, modality_name: str):
        return self.modality

    def get_name(self):
        return self.name
