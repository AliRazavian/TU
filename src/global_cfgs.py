import os
import sys
import argparse

from datetime import datetime
import pytz
from glob import iglob
import shutil
from UIs.console_UI import Console_UI
from Utils.retrieve_dir import retrieve_dir
from GeneralHelpers import Singleton
from file_manager import File_Manager
"""
This file stores variables that are constant during the runtime but will change
from runtime to tuntime
"""


def positive_float(string):
    value = float(string)
    if value < 0:
        msg = "%r is less than 0" % string
        raise argparse.ArgumentTypeError(msg)
    return value


class Global_Cfgs(metaclass=Singleton):

    def __init__(self, test_mode=False, test_init_values={}):
        if test_mode:
            self.cfgs = test_init_values

            Console_UI(self.get('log_level', 'warning'), globalConfigs=self)
            self.sub_log_path = self.get('sub_log_path', 'sub_log_not_set')

            File_Manager(annotations_root=self.get('annotation_root'),
                         log_folder=self.get('log_folder'),
                         scenario_log_root=self.get('scenario_log_root'),
                         resume_prefix=self.get('resume'),
                         resume_scene=self.get('resume_scene'),
                         tmp_root=self.get('tmp_root'),
                         model_zoo_root=self.get('model_zoo_root'),
                         global_cfgs=self)
            return

        args = self.parse_argument()
        self.cfgs = args.__dict__

        Console_UI(self.get('log_level', 'info'), globalConfigs=self)
        self.read_environment_variables()

        self.start_time = datetime.now(pytz.timezone('Europe/Stockholm')).strftime('%Y%m%d/%H.%M')
        self.sub_log_path = os.path.join(self.get('scenario'), self.start_time)

        if self.get('resume') is not None:
            self.prep_resume()

        fm = File_Manager(scenario_log_root=self.scenario_log_root,
                          log_folder=self.log_folder,
                          annotations_root=self.get('annotation_root'),
                          resume_prefix=self.get('resume'),
                          resume_scene=self.get('resume_scene'),
                          tmp_root=self.get('tmp_root'),
                          model_zoo_root=self.get('model_zoo_root'),
                          global_cfgs=self)

        setup_data = {'call': ' '.join(sys.argv), 'setup': self.cfgs}

        fm.log_setup(data=setup_data, name='base_setup.yaml')

        self.__forward_noise = 0

    @property
    def log_folder(self):
        return os.path.join(self.get('log_root'), self.sub_log_path)

    @property
    def scenario_log_root(self):
        return os.path.join(self.get('log_root'), self.get('scenario'))

    def set_forward_noise(self, forward_noise=1e-1):
        self.__forward_noise = forward_noise

    @property
    def forward_noise(self):
        return self.__forward_noise

    def parse_argument(self):
        parser = argparse.ArgumentParser("DeepMed")

        parser.add_argument(
            '--log_level',
            dest='log_level',
            default='Info',
            type=str.lower,
            choices=['off', 'warning', 'info', 'debug'],
            help='Different levels of logging',
        )

        parser.add_argument(
            '-scenario',
            dest='scenario',
            default='all_xray',
            type=str.lower,
            help='Scenario to run',
        )

        parser.add_argument(
            '-resume',
            dest='resume',
            default=None,
            type=str.lower,
            help='Resume prefix path. It should be in the form of YYYYMMDD/HH.MM',
        )

        parser.add_argument(
            '-lw',
            '--loss_weight_type',
            dest='loss_weight_type',
            default='basic',
            type=str.lower,
            choices=['basic', 'max', 'sqrt'],
            help='Different types of loss weights where basic is proportional to the actual number',
        )

        parser.add_argument(
            '-rl',
            '--resume_last',
            dest='resume',
            action='store_const',
            const='last',
            help='find the last run and resume from there',
        )

        parser.add_argument(
            '-rs',
            '--resume_scene',
            dest='resume_scene',
            default='last',
            type=str,
            help='Resume a specific scene - defaults to last',
        )

        parser.add_argument(
            '-rc',
            '--resume_config',
            dest='resume_config',
            action='store_true',
            help='Resume a the last config',
        )

        parser.add_argument(
            '-at',
            '--start_at_scene',
            dest='start_scene',
            default=None,
            help='The name of the scenario that we should start from.',
        )

        parser.add_argument(
            '--skip_tb_copy',
            action='store_true',
            dest='skip_tensorboard',
            help='Skip tensorboard copy if you are resuming previious run',
        )

        parser.add_argument(
            '-batch_size',
            action='store',
            dest='batch_size',
            default=128,
            type=int,
            help='Batch size. Note that if you want to shrink the size during different scenes you can use the config' +
            'batch_size_multipler for task or scene config',
        )

        parser.add_argument(
            '-learning_rate',
            action='store',
            dest='learning rate',
            default=1e-1,
            type=float,
            help='Learning Rate',
        )

        parser.add_argument(
            '-min_channel',
            action='store',
            dest='min_channel',
            default=2,
            type=int,
            help='Minimum representation size for when classes are smaller than a specific dim',
        )

        # More than 2 workers seem to cause shared memory error in Docker and the speed is hardly much better
        parser.add_argument(
            '-num_workers',
            action='store',
            dest='num_workers',
            default=2,
            type=int,
            help='Number of workers to use in PyTorch DataLoader',
        )

        parser.add_argument(
            '-test_run',
            dest='test_run',
            action='store_true',
            help='Runs a test run through the code with scene using minimal epoch_size, epochs & repeat',
        )

        parser.add_argument(
            '-silent_init',
            dest='silent_init_info',
            action='store_true',
            help='Skip all the initialization info',
        )

        parser.add_argument('--pseudo_mask_margin',
                            dest='pseudo_mask_margin',
                            default=0.5,
                            type=positive_float,
                            help='''
                            The margin used in margin loss for setting the boundary for when to accept predictions for
                            the pseudo labels (i.e. teacher\'s labels).
                            ''')

        parser.add_argument('--pseudo_loss_factor',
                            dest='pseudo_loss_factor',
                            default=0.5,
                            type=positive_float,
                            help='''
                            The factor to multiply the pseudo loss with. If we use a higher margin it can possibly be
                            of interest to increase this factor as the labels will have less noise in them.
                            ''')

        parser.add_argument('--version', action='version', version='%(prog)s 0.6')

        return parser.parse_args()

    def get(self, key, default=None):
        if (key in self.cfgs):
            return self.cfgs[key]

        return default

    def read_environment_variables(self):
        required_environment_variables = ['TENSOR_BACKEND', 'DEVICE_BACKEND', 'IMAGE_BACKEND']
        for var in required_environment_variables:
            if var not in os.environ:
                raise IndexError(f'You forgot to set {var} in your environment, i.e. {var}="your value"')
            self.cfgs[var] = os.environ[var]

        optional_environment_variables = [
            'ANNOTATION_ROOT',
            'LOG_ROOT',
            'TMP_ROOT',
            'MODEL_ZOO_ROOT',
            'LOG_ROOT',
            'TMP_ROOT',
            'MODEL_ZOO_ROOT',
            'IMAGENET_ROOT',
            'CIFAR10_ROOT',
            'XRAY_ROOT',
            'COCO_ROOT',
        ]
        for var in optional_environment_variables:
            if var in os.environ:
                self.cfgs[var.lower()] = os.environ[var]
            else:
                self.cfgs[var.lower()] = None

    def prep_resume(self):
        ui = Console_UI()
        resume_prefix = self.get('resume')
        resume_scene = self.get('resume_scene')
        if resume_scene is not None and resume_prefix is None:
            raise ValueError('You must provide resume prefix if you have set a resume scene')

        # for debug mode uncomment:
        # scenario_log_root = "/media/max/SSD_1TB/log/"
        if resume_prefix.lower() == 'last':
            dirs = sorted([d for d in iglob(f'{self.scenario_log_root}/*/*/neural_nets')])
            dirs = [d for d in dirs if len([f for f in iglob(f'{d}/*{resume_scene}.t7')]) > 0]
            if len(dirs) == 0:
                raise Exception(f'No previous runs found in \'{self.scenario_log_root}\' with *{resume_scene}.t7')
            resume_prefix = dirs[-1].lstrip(self.scenario_log_root).rstrip('/neural_nets')

            ui.inform_user(f'Resuming run from {resume_prefix}')
        elif resume_prefix is not None:
            resume_prefix = retrieve_dir(path=resume_prefix, base_path=self.scenario_log_root, expected_depth=1)
            ui.inform_user(f'Resuming run from {resume_prefix}')

        self.cfgs['resume'] = resume_prefix
        # for debug mode uncomment:
        # self.cfgs['resume'] = "../%s" % self.cfgs['resume']
        if not self.cfgs['skip_tensorboard']:
            dst_tensorboard_path = os.path.join(self.log_folder, 'tensorboard')
            if os.path.exists(dst_tensorboard_path):
                ui.inform_user(f'Removing previous tensorboard catalogue: {dst_tensorboard_path}')
                shutil.rmtree(dst_tensorboard_path)

            ui.inform_user('Copying the previous tensorboard data')
            shutil.copytree(
                src=os.path.join(self.scenario_log_root, resume_prefix, 'tensorboard'),
                dst=dst_tensorboard_path,
            )


"""
parser.add_argument('-s', action='store', dest='simple_value',
                    help='Store a simple value')

parser.add_argument('-c', action='store_const', dest='constant_value',
                    const='value-to-store',
                    help='Store a constant value')

parser.add_argument('-t', action='store_true', default=False,
                    dest='boolean_switch',
                    help='Set a switch to true')
parser.add_argument('-f', action='store_false', default=False,
                    dest='boolean_switch',
                    help='Set a switch to false')
"""
