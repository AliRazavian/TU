from file_manager import File_Manager
from GeneralHelpers import Singleton


class Dataset_Factory(metaclass=Singleton):

    def __init__(self):
        self.datasets = {}

    def get_dataset(
            self,
            dataset_name,
            batch_size_multiplier: float,
    ):
        fixed_name = dataset_name.lower()
        if fixed_name not in self.datasets:
            predefined_datasets = File_Manager().get_dataset_definitions()
            if (fixed_name in predefined_datasets):
                from .csv_dataset import CSV_Dataset as Dataset
            else:
                raise Exception('The dataset \'%s\' is not among the predefined sets: \'%s\'' %
                                (dataset_name, '\', \''.join(predefined_datasets)))

            self.datasets[fixed_name] = Dataset(
                dataset_name=fixed_name,
                batch_size_multiplier=batch_size_multiplier,
            )
        else:
            self.datasets[fixed_name].set_batch_size_multiplier(batch_size_multiplier)

        return self.datasets[fixed_name]
