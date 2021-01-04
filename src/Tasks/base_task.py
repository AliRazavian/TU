from abc import ABCMeta, abstractmethod

from global_cfgs import Global_Cfgs


class Base_Task(metaclass=ABCMeta):

    def __init__(
            self,
            task_name,
            task_cfgs,
            scenario_name,
            scenario_cfgs,
            scene_cfgs,
            has_pseudo_labels,
    ):
        self.task_name = task_name
        self.task_cfgs = task_cfgs
        self.scenario_name = scenario_name
        self.scenario_cfgs = scenario_cfgs
        self.scene_cfgs = scene_cfgs
        self.has_pseudo_labels = has_pseudo_labels
        self.epoch = 0
        self.graphs = {}

    def get_cfgs(self, name, default=None):
        # TODO: The get_cfgs probably be merged into one spot
        if name in self.task_cfgs:
            return self.task_cfgs[name]
        if name in self.task_cfgs['apply']:
            return self.task_cfgs['apply'][name]
        if name in self.scene_cfgs:
            return self.scene_cfgs[name]
        if name in self.scenario_cfgs:
            return self.scenario_cfgs[name]
        return Global_Cfgs().get(name, default=default)

    def get_name(self):
        return self.task_name

    def get_scenario_name(self):
        return self.scenario_name

    @abstractmethod
    def save(self, scene_name='last'):
        pass
