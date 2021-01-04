from file_manager import File_Manager
from global_cfgs import Global_Cfgs
from .experiment_set import Experiment_Set


class CSV_Dataset:

    def __init__(
            self,
            dataset_name,
            batch_size_multiplier: float,
    ):
        self.dataset_name = dataset_name.lower()
        self.batch_size_multiplier = batch_size_multiplier
        self.dataset_cfgs = self.get_dataset_cfgs(dataset_name)
        self.setup_experiments()

    def setup_experiments(self):
        self.experiments = {}
        for experiment_name, experiment_cfgs in \
                self.dataset_cfgs['experiments'].items():
            self.experiments[experiment_name] = Experiment_Set(
                dataset_name=self.dataset_name,
                dataset_cfgs=self.dataset_cfgs,
                experiment_name=experiment_name,
                experiment_cfgs=experiment_cfgs,
                batch_size_multiplier=self.batch_size_multiplier,
            )

    def set_batch_size_multiplier(self, number: int):
        for name in self.experiments.keys():
            self.experiments[name].set_batch_size_multiplier(number)

    def get_dataset_cfgs(self, dataset_name):
        dataset_cfgs = File_Manager().read_dataset_config(dataset_name)

        # For the sake of simplicity, when modalities are identical during
        # train and tests, we can just write "modalities": "same_as_X" in
        # the config file.(in this example, X is "train")
        # This piece of code searches for the modalities like this and
        # replace them with the "X" modalities
        for _, experiment_cfgs in dataset_cfgs['experiments'].items():
            if (isinstance(experiment_cfgs['modalities'], str)
                    and experiment_cfgs['modalities'].startswith('same_as_')):
                other_experiment = experiment_cfgs['modalities'][len('same_as_'):]
                experiments = dataset_cfgs['experiments']
                if other_experiment in experiments:
                    experiment_cfgs['modalities'] = experiments[other_experiment]['modalities']
                else:
                    raise KeyError('Could not find the modality \'%s\' among the modalities: \'%s\'' %
                                   (other_experiment, '\', \''.join(experiments.keys())))
        return dataset_cfgs

    def get_cfgs(self, name, default=None):
        if ('dataset_cfgs' in self.__dict__ and name in self.dataset_cfgs):
            return self.dataset_cfgs[name]
        return Global_Cfgs().get(name, default)

    def get_name(self):
        return self.dataset_name

    def get_output_modality_names(self, experiment_name):
        return self.experiments[experiment_name].get_output_modality_names()

    def get_input_modality_names(self, experiment_name):
        return self.experiments[experiment_name].get_input_modality_names()
