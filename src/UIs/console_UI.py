import os
from collections import defaultdict
import getpass
import numpy as np
import pandas as pd
import torchvision.utils as vutils
from datetime import datetime
from yaml import dump as yamlDump
from torch.utils.tensorboard import SummaryWriter
from GeneralHelpers import Singleton
from typing import Dict
pd.options.display.max_rows = 999


class Console_UI(metaclass=Singleton):
    pd.options.display.float_format = '{:,.02f}'.format

    def __init__(self, log_level: str, globalConfigs):
        log_levels = {"off": 0, "warning": 1, "info": 2, "debug": 3}
        self.log_level = log_levels[log_level]
        self.writers = {}
        self.last_reconstruction = defaultdict(dict)
        self.reset_overall()
        self.globalConfigs = globalConfigs

    def reset_overall(self):
        self.overall_epoch = None
        self.overall_repeat = None
        self.overall_total_epochs = None
        self.overall_total_repeats = None

    def set_log_folder(self, log_folder):
        self.log_folder = log_folder

    def costume_print(self, x, indent=0, force_print_dataframe=False):
        if isinstance(x, str):
            print('%s%s' % (''.join(['\t' for i in range(indent)]), x))
        elif isinstance(x, dict):
            print(yamlDump(x))
        elif isinstance(x, list):
            [self.costume_print(i, indent=indent + 1) for i in x]
        elif isinstance(x, pd.DataFrame):
            if not force_print_dataframe and (len(x.columns) > 10 or len(x.index) > 100):
                print(f'Skipped print of dataframe due to large size: {x.shape}')
            elif len(x.index) > 0:
                print(x.fillna(''))
            else:
                print('Empty dataframe?')
        else:
            print(x)

    def fix_types(self, o):
        if isinstance(o, np.int64):
            return int(o)
        if isinstance(o, np.ndarray):
            return list(o)
        raise TypeError

    def warn_user(self, message=None, info=None, debug=None):
        if self.log_level >= 1 and message is not None:
            self.costume_print(message, force_print_dataframe=True)
        self.inform_user(info, debug)

    def inform_user(self, info=None, debug=None):
        if self.log_level >= 2 and info is not None:
            self.costume_print(info)
        self.debug(debug)

    def debug(self, debug=None):
        if self.log_level >= 3 and debug is not None:
            self.costume_print(debug)

    def receive_regular_input(self, message: str = 'Input:[Y/n]', default: str = 'Y'):
        return input(message) or default

    def receive_password(self, message: str = 'Please enter password'):
        return getpass.getpass(message)

    def warn_user_and_ask_to_continue(self, message: str = 'Warning!!!'):
        self.warn_user(message)
        answer = self.receive_regular_input('continue? [y/N]', default='n')
        if answer.lower().startswith('n'):
            exit()

    def inform_user_and_ask_to_continue(self, message: str = 'Info!!'):
        self.inform_user(message)
        answer = self.receive_regular_input('continue? [Y/n]', default='y')
        if answer.lower().startswith('n'):
            exit()

    def add_epoch_results(self, summary: dict):
        s = pd.DataFrame(summary['modalities']).T
        columns2change = [c[len('pseudo_'):] for c in s.columns if c.startswith('pseudo_')]
        rename_cols = {c: f'teacher_{c}' for c in columns2change}
        s.rename(columns=rename_cols, inplace=True)

        result_name = self.get_result_name(
            ds=summary["dataset_name"],
            exp=summary['experiment_name'],
            task=summary['task_name'],
            graph=summary['graph_name'],
        )

        iteration = self.iteration
        if 'epoch_size' in summary:
            iteration -= summary['epoch_size'] // 2
        self.inform_user(s)

        self.add_scalar_to_tensorboard(s, result_name, iteration)
        self.add_reconstruction_to_tensorboard(result_name, iteration)

    def get_scene(self):
        # Import here as there otherwise is a cyclic dependency
        from scenario import Scenario
        return Scenario().get_current_scene()

    def get_result_name(self, ds: str, exp: str, task: str, graph: str):
        scene = self.get_scene()
        if scene is None:
            raise RuntimeError(f'Attempted retrieving {ds}_{exp}_{task}_{graph} before initiating scene!')

        scene_name = scene.get_name()
        return {
            'name': f'{exp}_{task}_{graph}_epoch',
            'id': f'{scene_name}_{ds}_{exp}_{task}_{graph}_epoch',
            'path': os.path.join(ds, scene_name),
        }

    def get_scenario_info(self):
        info = f'Scenario: {self.globalConfigs.get("scenario")} >> {self.globalConfigs.start_time}'

        resume = self.globalConfigs.get("resume", False)
        if resume:
            info += f' [resumed from {resume}::{self.globalConfigs.get("resume_scene")}]'

        return info

    def get_scene_info(self, batch: Dict):
        scene = self.get_scene()
        scenario = scene.get_name() if scene is not None else '?'

        graph = batch['graph_name']
        task = batch['task_name']

        info = f'Scene: {scenario}, graph: {graph}, task: {task}'
        if self.overall_epoch is not None and self.overall_repeat is not None:
            epoch_info = f'epoch: {self.overall_epoch} (of {self.overall_total_epochs})'
            repeat_info = f'repeat: {self.overall_repeat} (of {self.overall_total_repeats})'
            info += f' {epoch_info} {repeat_info}'

        return info

    def get_counter_info(self, batch):
        epoch = batch['epoch']
        index = batch['batch_index']
        size = batch['epoch_size']
        counter = batch['iteration_counter']
        batch_size = batch['batch_size']
        return f'Batch specifics: epoch[{epoch}][{index}/{size}], iteration[{counter}], batch_size: {batch_size}'

    def get_time_info(self, batch: Dict):
        time_spent = batch['time_stats']['end'].max(level=0) -\
                     batch['time_stats']['start'].min(level=0)  # noqa

        # When the software hangs it is nice to have some time info
        at_time = f'on {datetime.now():%Y-%m-%d %H:%M:%S}'
        total_ms = float(batch['time']['true_full_time']) * 1000
        batch_times = [f'{k}: {t*1000:4.0f}ms' for k, t in time_spent.to_dict().items()]

        time_spent_string = f'Time spent per batch: {total_ms:.0f}ms ({", ".join(batch_times)}) {at_time}'

        return time_spent_string

    def add_batch_results(self, batch: dict):
        self.iteration = batch['iteration_counter']
        if self.iteration > 5 and self.iteration % 10 != 0:
            return

        for _, v in batch['results'].items():
            nondisplay_items = ['output', 'pseudo_output', 'target', 'euclidean_distance']
            for item in nondisplay_items:
                if item in v:
                    v.pop(item)

        s = pd.DataFrame(batch['results']).T
        self.inform_user(s)
        self.inform_user('-----------------')
        self.debug(batch['time_stats'])
        self.debug('-----------------')

        info_string = f'{self.get_scenario_info()}\n{self.get_scene_info(batch)}\n{self.get_counter_info(batch)}'

        results_string = 'Batch summary: '
        results_string += ', '.join(['mean %s: %.2e' % (k, s[k].mean()) for k in s.keys()])

        self.inform_user(f'{info_string}\n{results_string}\n{self.get_time_info(batch)}\n=======')

        # The runtime becomes huge and lacks anything really interesting...
        # self.add_scalar_to_tensorboard(s, result_name, self.iteration)

    def add_scalar_to_tensorboard(self, x, result_name, iteration):

        id = result_name['id']
        if id not in self.writers:
            self.writers[id] = SummaryWriter(
                os.path.join(self.log_folder, 'tensorboard', result_name['path'], result_name['name']))
        writer = self.writers[id]

        for measurement in x:
            for name in x[measurement].index:
                scalar = x[measurement][name]
                if not isinstance(scalar, (list, np.ndarray)) and not np.isnan(scalar):
                    if measurement[-4:] == 'loss':
                        # cap the loss as it is not interesting to view if it explodes
                        scalar = min(scalar, 20)
                    elif measurement[-3:] == 'auc':
                        # Values below 0.5 are meaningless
                        scalar = max(scalar, 0.5)

                    writer.add_scalar('%s/%s' % (measurement, name), scalar, iteration)

    def add_last_reconstructed_input(self, batch):
        if ('decoder_image' in batch):
            last_reconst = vutils.make_grid(batch['decoder_image'].clone().detach(), normalize=True, scale_each=True)
            last_image = vutils.make_grid(batch['encoder_image'].view(
                -1, *batch['encoder_image'].shape[-3:]).clone().detach(),
                                          normalize=True,
                                          scale_each=True)

            result_name = self.get_result_name(ds=batch['dataset_name'],
                                               exp=batch['experiment_name'],
                                               task=batch['task_name'],
                                               graph=batch['graph_name'])
            id = result_name['id']
            self.last_reconstruction[id]['reconst'] = last_reconst.cpu().clone().detach()
            self.last_reconstruction[id]['image'] = last_image.cpu().clone().detach()

    def add_reconstruction_to_tensorboard(self, result_name: dict, iteration: int):
        id = result_name['id']
        writer = self.writers[id]
        if (id in self.last_reconstruction):
            writer.add_image(
                'Visualization/Reconstruction',
                self.last_reconstruction[id]['reconst'],
                iteration,
            )
            writer.add_image(
                'Visualization/Image',
                self.last_reconstruction[id]['image'],
                iteration,
            )
