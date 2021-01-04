class FileManagerMock:

    def __init__(self, fake_values={}):
        init_values = {}
        init_values.update(fake_values)
        self.cfgs = init_values

    def get(self, name):
        if name in self.cfgs:
            return self.cfgs[name]

        return None

    def get_dictionary_name(
            dataset_name: str,
            modality_name: str,
    ):
        return f'read_dictionary:{dataset_name}->{modality_name}'

    def read_dictionary(
            self,
            dataset_name: str,
            modality_name: str,
    ):
        csv_name = FileManagerMock.get_dictionary_name(dataset_name=dataset_name, modality_name=modality_name)
        return self.get(name=csv_name)
