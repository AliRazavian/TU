from math import ceil
from collections import OrderedDict
from difflib import get_close_matches
import gc

from global_cfgs import Global_Cfgs
from scene import Scene
from Datasets.helpers import Dictionary_Generator
from UIs.console_UI import Console_UI
from file_manager import File_Manager
from GeneralHelpers import Singleton


class Scenario(metaclass=Singleton):

    def __init__(self, scenario_name: str):
        self.scenario_name = scenario_name
        fm = File_Manager()
        self.scenario_cfgs = fm.read_scenario_config(self.scenario_name)

        self.current_epoch = self.get_cfgs('current_epoch', default=0)
        self.optimizer_type = self.get_cfgs('optimizer_type')

        self.scenes = OrderedDict({})
        for scene_cfgs in self.get_cfgs('scenes'):
            scene_name = scene_cfgs['name']
            task_defaults = {}
            if 'task_defaults' in scene_cfgs:
                task_defaults = scene_cfgs['task_defaults']

            scene_cfgs['tasks'] = {
                t: fm.read_task_config(task_name=t, scenario=self, scene_defaults=task_defaults)
                for t in scene_cfgs['tasks']
            }
            self.scenes[scene_name] = scene_cfgs

        self.scenario_lengths = [{'name': key, 'len': len(s)} for (key, s) in self.scenes.items()]

        self.collect_dictionaries()

        self.__current_scene = None

    def __iter__(self):
        for (name, cfgs) in self.scenes.items():
            gc.collect()
            self.__current_scene = Scene(
                scenario_name=self.scenario_name,
                scenario_cfgs=self.scenario_cfgs,
                scene_name=name,
                scene_cfgs=cfgs,
            )

            yield self.__current_scene
            gc.collect()

    def __len__(self):
        return sum([v['len'] for v in self.scenario_lengths])

    def closing_credits(self):
        Console_UI().inform_user("That's it folks")

    def get_cfgs(self, name, default=None):
        if (name in self.scenario_cfgs):
            return self.scenario_cfgs[name]
        return Global_Cfgs().get(name, default)

    def get_name(self):
        return self.scenario_name

    def get_current_scene(self):
        return self.__current_scene

    def collect_dictionaries(self):
        """
        Check all the Datasets for common items, e.g. body part and then create
        a general dictionary for all of them.
        """
        datasets = []
        for scene in self.scenario_cfgs['scenes']:
            for task in scene['tasks'].values():
                if task['dataset_name'] not in datasets:
                    datasets.append(task['dataset_name'])

        configs = {}
        for dataset_name in datasets:
            configs[dataset_name] = File_Manager().read_dataset_config(dataset_name)

        modalities_with_dictionaries = [
            'one_vs_rest',
            'bipolar',
            'multi_bipolar',
        ]  # TODO: add 'hierarchical_label' but this has some fancy logic :-S

        dictionary_candidates = []
        for dataset_name in datasets:
            config = configs[dataset_name]
            try:
                for experiment in config['experiments'].values():
                    if isinstance(experiment['modalities'], dict):
                        [
                            dictionary_candidates.append(name)
                            for name, cfg in experiment['modalities'].items()
                            if cfg['type'].lower() in modalities_with_dictionaries and name not in dictionary_candidates
                        ]
            except Exception as e:
                raise Exception(f'Failed to get dictionary for {dataset_name}: {e}')

        # Store all the different values available for this modality into the dictionary singleton that
        # keeps track of the unique values
        dg = Dictionary_Generator()
        for modality_name in dictionary_candidates:
            for dataset_name in datasets:
                dg.append_suggested_dictionary(dataset_name=dataset_name, modality_name=modality_name)

                config = configs[dataset_name]
                for experiment in config['experiments'].values():
                    annotations = File_Manager().read_csv_annotations(
                        dataset_name,
                        annotations_rel_path=experiment['annotations_path'],
                        # Multi-view argument should be irrelevant for this
                    )
                    if annotations is None:
                        raise ValueError(
                            f'Could not find the dataset: {dataset_name} in {experiment["annotations_path"]}')

                    modalities = experiment['modalities']
                    if modalities == 'same_as_train_set':
                        modalities = config['experiments']['train_set']['modalities']

                    if modality_name in modalities:
                        if 'column_name' in modalities[modality_name]:
                            try:
                                colname = modalities[modality_name]['column_name']
                                dg.append_values(modality_name=modality_name, values=annotations[colname])
                            except KeyError as e:
                                Console_UI().warn_user(f'Got a key annotation exception for {colname}')
                                Console_UI().warn_user(modalities[modality_name])
                                Console_UI().warn_user(annotations.columns)
                                raise e
                            except Exception as e:
                                Console_UI().warn_user(f'Got an annotation exception for {colname}')
                                Console_UI().warn_user(modalities[modality_name])
                                Console_UI().warn_user(annotations)
                                raise e
                        elif 'columns' in modalities[modality_name]:
                            for column_name in modalities[modality_name]['columns']:
                                if isinstance(column_name, dict):
                                    assert 'csv_name' in column_name, \
                                        f'The column doesn\'t have the expected csv_name element, got: {column_name}'
                                    column_name = column_name['csv_name']
                                if column_name not in annotations:
                                    n = 3 if len(annotations.columns) < 10 else ceil(len(annotations.columns) / 3)
                                    closest = get_close_matches(
                                        word=column_name,
                                        possibilities=annotations.columns,
                                        n=n,
                                    )
                                    closest = ', '.join(closest)
                                    raise IndexError(f'The {column_name} from {modality_name} doesn\'t exist.' +
                                                     f' Closest matching are: {closest}')
                                dg.append_values(modality_name=modality_name, values=annotations[column_name])
                        else:
                            raise IndexError(f'Expected {modality_name} to have either columns or column_name defined')
