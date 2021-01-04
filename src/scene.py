import re
import numpy as np

from global_cfgs import Global_Cfgs
from file_manager import File_Manager
from UIs.console_UI import Console_UI
from Tasks.training_task import Training_Task


class Scene:

    def __init__(
            self,
            scenario_name,
            scenario_cfgs,
            scene_name,
            scene_cfgs,
    ):
        self.scenario_name = scenario_name
        self.scenario_cfgs = scenario_cfgs
        self.scene_name = scene_name
        self.scene_cfgs = scene_cfgs
        self.epochs = self.get_cfgs('epochs')
        self.repeat = self.get_cfgs('repeat', default=1)

        self.learning_rate = self.get_cfgs('learning_rate')

        self.stochastic_weight_averaging = self.get_cfgs('stochastic_weight_averaging', default=False)
        self.stochastic_weight_averaging_last = self.get_cfgs('stochastic_weight_averaging_last',
                                                              default=self.stochastic_weight_averaging)

        self.tasks = {}
        for task_name, task_cfgs in self.get_cfgs('tasks').items():
            if task_cfgs['task_type'].lower() == 'training'.lower():
                self.tasks[task_name] = Training_Task(task_name=task_name,
                                                      task_cfgs=task_cfgs,
                                                      scenario_name=scenario_name,
                                                      scenario_cfgs=scenario_cfgs,
                                                      scene_cfgs=self.scene_cfgs,
                                                      has_pseudo_labels=self.get_cfgs('has_pseudo_labels',
                                                                                      default=False))
            else:
                raise BaseException('Unknown task type %s' % (task_cfgs['task_type']))

        self.main_task = self.get_cfgs('main_task')
        if self.main_task not in self.tasks:
            regex_main_matches = list(filter(re.compile(self.main_task).search, self.tasks))
            if (len(regex_main_matches) > 0):
                self.main_task = regex_main_matches[0]
            else:
                self.main_task = list(self.tasks.keys())[0]
        self.epoch_size = len(self.tasks[self.main_task])

        if self.get_cfgs('test_run'):
            self.epochs = 2
            if self.repeat > 3:
                self.repeat = 3

        self.task_load_balancer = {}  # Keeps track of the number of runs per task

    def __len__(self):
        return self.epochs * self.repeat

    iteration_counter = 0

    def should_task_run(self, task_name, task):
        """
        If certain task are much smaller than the current task then we should
        skip that task a few times or small datasets risk overfitting
        """
        if task_name != self.main_task and self.epoch_size > len(task):
            if task_name in self.task_load_balancer:
                self.task_load_balancer[task_name] += 1
                if self.task_load_balancer[task_name] * len(task) % self.epoch_size > len(task):
                    return False
            else:
                self.task_load_balancer[task_name] = 0
        return True

    def run_scene(self, start_epoch=0):
        logged_memory_usage = False
        ui = Console_UI()
        ui.overall_total_epochs = self.epochs
        ui.overall_total_repeats = self.repeat

        Global_Cfgs().set_forward_noise(self.get_cfgs('forward_noise', default=0))
        for r in range(0, self.repeat):
            ui.overall_repeat = r
            if (self.stochastic_weight_averaging and r > 0):
                self.tasks[self.main_task].stochastic_weight_average()

            for e in range(0, self.epochs):
                ui.overall_epoch = e
                if start_epoch > e + r * self.epochs:
                    Scene.iteration_counter += self.epoch_size
                else:
                    for task in self.tasks.values():
                        task.update_learning_rate(self.get_learning_rate(e))

                    for _ in range(self.epoch_size):
                        for key, task in self.tasks.items():
                            if self.should_task_run(task_name=key, task=task):
                                task.step(iteration_counter=Scene.iteration_counter, scene_name=self.scene_name)
                        Scene.iteration_counter += 1

                        if logged_memory_usage is False:
                            for key in self.tasks.keys():
                                task = self.tasks[key]
                                memory_usage = task.get_memory_usage_profile()
                                File_Manager().write_usage_profile(
                                    scene_name=self.scene_name,
                                    task=key,
                                    memory_usage=memory_usage,
                                )
                                ui.inform_user(f'\n Memory usage for {self.scene_name}::{key}\n')
                                ui.inform_user(memory_usage)
                            logged_memory_usage = True

                    for task in self.tasks.values():
                        task.save(scene_name='last')
                        # Not really helping with just emptying cache - we need to add something more
                        # removing as this may be the cause for errors
                        # torch.cuda.empty_cache()
        ui.reset_overall()

        # Note that the evaluation happens after this step and therefore averaging may hur the performance
        if self.stochastic_weight_averaging_last:
            self.tasks[self.main_task].stochastic_weight_average()
            for task in self.tasks.values():
                task.save(scene_name='last')

        for task in self.tasks.values():
            task.validate(iteration_counter=Scene.iteration_counter, scene_name=self.scene_name)
            task.test(iteration_counter=Scene.iteration_counter, scene_name=self.scene_name)

        # Save all tasks before enterering the next scene
        for task in self.tasks.values():
            task.save(scene_name=self.scene_name)
            [g.dropModelNetworks() for g in task.graphs.values()]
            # Not really helping with just emptying cache - we need to add something more
            # removing as this may be the cause for errors
            # torch.cuda.empty_cache()

    def get_name(self):
        return self.scene_name

    def get_cfgs(self, name, default=None):
        if name in self.scene_cfgs:
            return self.scene_cfgs[name]
        if name in self.scenario_cfgs:
            return self.scenario_cfgs[name]
        return Global_Cfgs().get(name, default=default)

    def get_learning_rate(self, epoch):
        if isinstance(self.learning_rate, float):
            return self.learning_rate
        elif isinstance(self.learning_rate, dict):
            assert('type' in self.learning_rate),\
                'learning rate is defined as dictionary but' +\
                'does not have a "type" field '
            assert(self.learning_rate['type'].lower() in ['decay']),\
                'learning rate should be in ["decay"] but found %s' \
                % self.learning_rate['type'].lower()

            if self.learning_rate['type'].lower() == 'decay':
                if self.learning_rate['function'].lower() == 'cosine':
                    lsp = np.linspace(0, np.pi / 2, self.epochs + 1, dtype='float32')
                    return np.cos(lsp)[epoch] * self.learning_rate['starting_value']
                elif self.learning_rate['function'].lower() == 'linear':
                    lsp = np.linspace(1, 0, self.epochs + 1, dtype='float32')
                    return lsp[epoch] * self.learning_rate['starting_value']
                else:
                    raise BaseException('Unknown learning rate function %s' % self.learning_rate['function'])
            else:
                raise BaseException('Unknown learning rate type %s' % self.learning_rate['type'])
        else:
            raise BaseException('Unknown learning rate %s' % self.learning_rate)
