import os
import json
from yaml import load as yamlLoad, Loader as YamlLoader, dump as yamlDump
import re
import numpy as np
import pandas as pd
import torch
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen
from GeneralHelpers import Singleton
from UIs.console_UI import Console_UI
from typing import Dict

np.set_printoptions(precision=3, linewidth=100000, threshold=10000)


def get_base_dir(path: str):
    if os.path.isdir(path):
        return path

    super_path = re.sub("/[^/]+$", "", path)
    if len(super_path) == 0 or path == super_path:
        return None

    return get_base_dir(super_path)


class File_Manager(metaclass=Singleton):

    def __init__(
            self,
            annotations_root,
            scenario_log_root,
            tmp_root,
            model_zoo_root,
            resume_prefix,
            resume_scene,
            log_folder,
            global_cfgs,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.annotations_root = annotations_root
        self.scenario_log_root = scenario_log_root
        self.tmp_root = tmp_root
        self.model_zoo_root = model_zoo_root
        self.resume_prefix = resume_prefix
        self.resume_scene = resume_scene

        self.log_folder = log_folder
        Console_UI().set_log_folder(self.log_folder)
        self.log_configs = True

        # TODO - set iteration counter on init to last value (this should be saved in a iteration counter txt file)

        # We can't use the Singleton pattern here as the Global_Cfgs() imports and initiates File_Manager
        self.global_cfgs = global_cfgs

        self.__cache = {}

    def get_annotations_path(self, dataset_name: str):
        return os.path.join(self.annotations_root, dataset_name.lower())

    def read_graph_config(self, graph_name: str):
        config_path = os.path.join('..', 'configs', 'Graphs', '%s' % (graph_name.lower()))
        return self.read_config(config_path)

    def read_dataset_config(self, dataset_name: str):
        config_path = os.path.join('..', 'configs', 'Datasets', '%s' % (dataset_name.lower()))
        return self.read_config(config_path)

    def get_dataset_definitions(self):
        config_path = os.path.join('..', 'configs', 'Datasets')
        return [re.sub("\\.(json|yaml)$", "", fn) for fn in os.listdir(config_path) if re.search("\\.(json|yaml)$", fn)]

    def read_scenario_config(self, scenario_name):
        if self.resume_prefix is not None and self.global_cfgs.get('resume_config', False):
            print(f'Resuming scenario configs {scenario_name} from {self.resume_prefix}')
            scenario_cfgs = self.read_config(os.path.join(self.scenario_log_root, self.resume_prefix, 'scenario'))
            if (scenario_cfgs is not None):
                return scenario_cfgs

        config_dir = os.path.join('..', 'configs', 'Scenarios')
        config_path = os.path.join(config_dir, '%s' % (scenario_name.lower()))

        scenario = self.read_config(config_path)
        if scenario is None:
            available_configs = [
                f for f in os.listdir(config_dir)
                if os.path.isfile(os.path.join(config_dir, f)) and re.match(".+\\.(json|yaml)$", f)
            ]
            raise ImportError("Could not load file: %s files in that dir: \n - %s" %
                              (config_path, '\n - '.join(available_configs)))
        return scenario

    def get_task_config_path(self, path):
        config_path = os.path.join('..', 'configs', 'Tasks')
        for p in path.split('/'):
            config_path = os.path.join(config_path, p)
        return config_path

    def build_task(self, task: Dict, task_name, scenario, scene_defaults: Dict = {}):
        # Add general template definitions
        if 'template' in task:
            template = self.read_task_config(task_name=f'Template/{task["template"]}', scenario=None)
            task = {**template, **task}

        if scenario is not None:
            # Scenario defaults are overridden by task definitions
            scenario_defaults = scenario.get_cfgs('task_defaults')
            if isinstance(scenario_defaults, dict):
                task = {**scenario_defaults, **task}

        # Scene defaults overridde task definitions
        if isinstance(scene_defaults, dict):
            task = {**task, **scene_defaults}

        assert 'graph_name' in task or scenario.get_cfgs('graph_name', False), \
            f'Expected task "{task_name}" or scenario to have "grah_name"'
        assert 'dataset_name' in task, f'Expected {task_name} to have "dataset_name"'
        assert 'train_set_name' in task, f'Expected {task_name} to have "train_set_name"'
        assert 'apply' in task, f'Expected {task_name} to have "apply"'

        return task

    def read_task_config(self, task_name, scenario, scene_defaults: Dict = {}):
        if scenario is not None:
            scenario_tasks = scenario.get_cfgs('tasks')
            if scenario_tasks is not None and task_name in scenario_tasks:
                return self.build_task(task=scenario_tasks[task_name],
                                       task_name=task_name,
                                       scenario=scenario,
                                       scene_defaults=scene_defaults)

        # Convert Ankle/regulated to full path including the yaml
        config_path = self.get_task_config_path(path=task_name)

        task = self.read_config(config_path)
        if task is None:
            config_base = get_base_dir(config_path)
            available_configs = [
                f for f in os.listdir(config_base)
                if os.path.isfile(os.path.join(config_base, f)) and re.match(".+\\.(json|yaml)$", f)
            ]
            raise ImportError("Could not load file: %s files in that dir: \n - %s" %
                              (config_path, '\n - '.join(available_configs)))

        return self.build_task(task=task, task_name=task_name, scenario=scenario, scene_defaults=scene_defaults)

    def read_config(self, path):
        if not re.match(".+\\.(json|yaml)$", path):
            if os.path.exists(f'{path}.json'):
                path = f'{path}.json'
            elif os.path.exists(f'{path}.yaml'):
                path = f'{path}.yaml'

        config_data = None
        # The idea with none it could have a fallback default
        if os.path.exists(path):
            with open(path, 'r') as fstr:
                if re.match(".+\\.json$", path):
                    config_data = json.load(fstr)
                else:
                    config_data = yamlLoad(fstr, Loader=YamlLoader)

        if self.log_configs and config_data is not None:
            self.log_setup(data=config_data, name=path)

        return config_data

    def log_setup(self, data: dict, name: str):
        fn = os.path.basename(name)
        if not fn.endswith('.yaml'):
            if fn.endswith('.json'):
                fn = fn[:-len('json')] + 'yaml'
            else:
                fn = f'{fn}.yaml'

        # Scenario & Dataset configs may have the same name and
        # therefore we need to prefix according to the last dirname
        dirname = os.path.dirname(name)
        if len(dirname) > 3:
            subdir = dirname.rfind("/")
            if subdir >= 0:
                dirname = dirname[(subdir + 1):]
            fn = f'{dirname}_{fn}'

        dest_cfg_log_fn = os.path.join(self.log_folder, 'setup', fn)
        if not os.path.exists(dest_cfg_log_fn):
            self.make_sure_dir_exist(file_path=dest_cfg_log_fn)
            with open(dest_cfg_log_fn, 'w') as fstr:
                yamlDump(data, stream=fstr)

    def get_available_csvs(self, dataset_name: str):
        path = self.get_annotations_path(dataset_name)
        return [i[:-len('.csv')] for i in os.listdir(path) if re.search("\\.csv$", i)]

    def read_csv_annotations(
            self,
            dataset_name: str,
            annotations_rel_path: str,
            multi_view_per_sample: bool = False,
    ):
        annotations_path = os.path.join(self.get_annotations_path(dataset_name), annotations_rel_path)

        if os.path.exists(annotations_path):
            cache_path = f'csv:{annotations_path}'
            if cache_path in self.__cache:
                annotation = self.__cache[cache_path]
            else:
                annotation = pd.read_csv(annotations_path, low_memory=False)
                self.__cache[cache_path] = annotation

            if multi_view_per_sample and not isinstance(annotation.index, pd.MultiIndex):
                if 'index' not in annotation:
                    annotation['index'] = np.arange(len(annotation), dtype=int)
                else:
                    assert np.issubdtype(annotation['index'], np.dtype(int)), 'Index should be integers'
                    assert annotation['index'].min() == 0, 'The index has to be indexed from 0'

                if 'sub_index' not in annotation:
                    annotation['sub_index'] = np.zeros(len(annotation), dtype=int)
                else:
                    assert np.issubdtype(annotation['sub_index'], np.dtype(int)), 'Sub index should be integers'
                    assert annotation['sub_index'].max() > 0, 'You have provided a sub_index without purpose (max 0)'
                    assert annotation['sub_index'].min() == 0, 'The sub_index has to start from 0'

                annotation.set_index(['index', 'sub_index'], inplace=True)
            if 'num_views' not in annotation:
                annotation['num_views'] =\
                    [a for b in [[i] * i for i in annotation.groupby(level=0).size()] for a in b]

            return annotation

        Console_UI().warn_user(f'Failed to load file from disk: \'{annotations_path}\'')
        return None

    def write_csv_annotation(
            self,
            annotations: pd.DataFrame,
            dataset_name: str,
            experiment_file_name: str,
    ):
        annotations_path = os.path.join(self.log_folder, dataset_name, experiment_file_name)
        self.make_sure_dir_exist(annotations_path)
        annotations.to_csv(annotations_path, index=False)

    def read_dictionary(
            self,
            dataset_name: str,
            modality_name: str,
    ):
        """
        If we have a dictionary associated with the current weights we should use those.
        The fallback is the resume weight's dictionary and lastly the annotation's dictionary.
        """
        filename = f'{modality_name.lower()}_dictionary.csv'
        cachename = f'dictionary:{dataset_name}->{filename}'
        if cachename in self.__cache:
            return self.__cache[cachename]

        dictionary_path = os.path.join(self.log_folder, 'neural_nets', filename)

        if (not os.path.exists(dictionary_path) and self.resume_prefix is not None):
            dictionary_path = os.path.join(self.scenario_log_root, self.resume_prefix, 'neural_nets', filename)

        if not os.path.exists(dictionary_path):
            dictionary_path = os.path.join(self.get_annotations_path(dataset_name), filename)

        if os.path.exists(dictionary_path):
            try:
                dictionary = pd.read_csv(dictionary_path)
                self.__cache[cachename] = dictionary
                return dictionary
            except pd.errors.EmptyDataError:
                Console_UI().warn_user(f'The dictionary for {modality_name} is corrupt - see file {dictionary_path}')

        return None

    def write_dictionary(
            self,
            dictionary: pd.DataFrame,
            dataset_name: str,
            modality_name: str,
    ):
        """
        We save the dictionary with the annotations and the network weights. If we resume the weights
        then it is critical that the weights are interpreted with the same dictionary as used for those
        weights in case the order of the items change
        """
        filename = f'{modality_name.lower()}_dictionary.csv'
        dictionary_path = os.path.join(self.get_annotations_path(dataset_name), filename)

        self.make_sure_dir_exist(dictionary_path)
        dictionary.to_csv(dictionary_path, index=False)

        self.write_dictionary2logdir(dictionary=dictionary, modality_name=modality_name)

    def write_dictionary2logdir(
            self,
            dictionary: pd.DataFrame,
            modality_name: str,
    ):
        filename = f'{modality_name.lower()}_dictionary.csv'
        neural_net_dictionary_path = os.path.join(self.log_folder, 'neural_nets', filename)
        self.make_sure_dir_exist(neural_net_dictionary_path)
        dictionary.to_csv(neural_net_dictionary_path, index=False)

    def write_usage_profile(self, scene_name: str, task: str, memory_usage: pd.DataFrame):
        memory_path = os.path.join(self.log_folder, f'{scene_name}_{task.replace("/", "_")}_memory_usage.csv')
        memory_usage.to_csv(memory_path, index=True)

    def load_pytorch_neural_net(self, neural_net_name: str):
        current_run_last_save = self.get_network_full_path(neural_net_name=neural_net_name, scene_name='last')
        if os.path.exists(current_run_last_save):
            Console_UI().inform_user(f'Resuming current runs {neural_net_name:>90}::last network')
            return torch.load(current_run_last_save)['state_dict']

        if self.resume_prefix is not None:
            scene_name = self.global_cfgs.get('resume_scene')
            network_filename = self.get_network_filename(neural_net_name=neural_net_name, scene_name=scene_name)
            neural_net_path = os.path.join(self.scenario_log_root, self.resume_prefix, 'neural_nets', network_filename)
            if os.path.exists(neural_net_path):
                Console_UI().inform_user(f'Resuming from {self.resume_prefix} the network {network_filename}')
                return torch.load(neural_net_path)['state_dict']

        if self.model_zoo_root is not None:
            model_zoo_neural_net_path = os.path.join(self.model_zoo_root, '%s.t7' % (neural_net_name))
            if (os.path.exists(model_zoo_neural_net_path)):
                Console_UI().inform_user(f'loading from model_zoo {model_zoo_neural_net_path}')
                return torch.load(model_zoo_neural_net_path)['state_dict']

        if not self.global_cfgs.get('silent_init_info'):
            Console_UI().inform_user(f'{neural_net_name} does not exist, Initializing from scratch')

        return None

    def save_pytorch_neural_net(
            self,
            neural_net_name,
            neural_net,
            scene_name: str = 'last',
            full_network=False,
    ):
        neural_net_path = self.get_network_full_path(neural_net_name=neural_net_name, scene_name=scene_name)
        self.make_sure_dir_exist(neural_net_path)
        # This prints too much stuff to realy be useful
        # Console_UI().inform_user(f'Saving {neural_net_name:>90}::{scene_name} to: {os.path.dirname(neural_net_path)}')
        torch.save({
            'state_dict': neural_net.layers.state_dict(),
            'neural_net_cfgs': neural_net.neural_net_cfgs
        }, neural_net_path)

    def get_network_filename(self, neural_net_name: str, scene_name: str):
        return f'{neural_net_name}_{scene_name}.t7'

    def get_network_dir_path(self):
        return os.path.join(self.log_folder, 'neural_nets')

    def get_network_full_path(self, neural_net_name: str, scene_name: str):
        network_filename = self.get_network_filename(neural_net_name=neural_net_name, scene_name=scene_name)
        return os.path.join(self.get_network_dir_path(), network_filename)

    def write_description(
            self,
            file_path,
            description: str,
    ):
        self.make_sure_dir_exist(file_path)
        desc_file = open(file_path, 'w')
        desc_file.write(description)
        desc_file.close()

    def make_sure_dir_exist(self, file_path: str):
        dir_name = os.path.dirname(file_path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)

    def download_zip_file(self, url, dataset_name):
        """
        annotations has to be zipped with the following command:
        7z a -ppassword -mem=ZipCrypto imagenet.zip imagenet
        """

        url_content = urlopen(url)
        zipfile = ZipFile(BytesIO(url_content.read()))

        pswd = Console_UI().receive_password('Password for unzipping annotations of %s dataset:' % (dataset_name))
        zipfile.extractall(self.annotations_root, pwd=bytes(pswd, 'utf-8'))
