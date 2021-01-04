import torch
import torch.nn as nn
from abc import ABCMeta, abstractmethod

from global_cfgs import Global_Cfgs


class Base_Loss(nn.Module, metaclass=ABCMeta):

    def __init__(
            self,
            experiment_set,
            graph,
            graph_cfgs: dict,
            loss_name: str,
            loss_cfgs: dict,
            task_cfgs: dict,
            scene_cfgs: dict,
            scenario_cfgs: dict,
    ):

        super().__init__()
        self.experiment_set = experiment_set
        self.graph_cfgs = graph_cfgs
        self.graph = graph
        self.loss_name = loss_name
        self.loss_cfgs = loss_cfgs
        self.task_cfgs = task_cfgs
        self.scene_cfgs = scene_cfgs
        self.scenario_cfgs = scenario_cfgs
        self.modality_name = self.get_cfgs('modality_name')
        self.modality = self.experiment_set.get_modality(self.modality_name)
        self.tensor_shape = self.modality.get_tensor_shape()

        self.neural_net = None
        self.initial_learning_rate = self.get_cfgs('learning_rate')
        self.coef = self.get_cfgs('loss_coef', default=1.)
        self.use_cuda = self.get_cfgs('DEVICE_BACKEND').lower() == 'cuda'.lower()

    @abstractmethod
    def forward(self, batch):
        pass

    def pool_and_reshape_output(self, output, num_views=None):
        output = output.view([-1, *self.tensor_shape])
        return output

    def pool_and_reshape_target(self, target, num_views=None):
        target = target.view([-1, *self.tensor_shape])
        return target

    def get_name(self):
        return self.loss_name

    def get_cfgs(self, name, default=None):
        if name in self.loss_cfgs:
            return self.loss_cfgs[name]
        if name in self.graph_cfgs:
            return self.graph_cfgs[name]
        if name in self.task_cfgs['apply']:
            return self.task_cfgs['apply'][name]
        if name in self.scene_cfgs:
            return self.scene_cfgs[name]
        if name in self.scenario_cfgs:
            return self.scenario_cfgs[name]
        return Global_Cfgs().get(name, default=default)

    def zero_grad(self):
        if self.neural_net is not None:
            self.neural_net.zero_grad()

    def step(self):
        if self.neural_net is not None:
            self.neural_net.step()

    def train(self):
        if self.neural_net is not None:
            self.neural_net.train()

    def eval(self):
        if self.neural_net is not None:
            self.neural_net.eval()

    def save(self, scene_name):
        if self.neural_net is not None:
            self.neural_net.save(scene_name)

    def update_learning_rate(self, learning_rate):
        if self.neural_net is not None:
            self.neural_net.update_learning_rate(learning_rate)
        else:
            self.initial_learning_rate = learning_rate

    def update_stochastic_weighted_average_parameters(self):
        if self.neural_net is not None:
            self.neural_net.update_stochastic_weighted_average_parameters()

    def prepare_for_batchnorm_update(self):
        if self.neural_net is not None:
            self.neural_net.prepare_for_batchnorm_update()

    def update_batchnorm(self, batch):
        if self.neural_net is not None:
            self.neural_net.update_batchnorm(batch)

    def finish_batchnorm_update(self):
        if self.neural_net is not None:
            self.neural_net.finish_batchnorm_update()

    def convert_batch_results_to_detached_cpu(self, batch: dict):
        """
        Doesn't seem to save any space but in theory it can be called after forward() and
        moves all tensor results to cpu and with detach to save space on the GPU
        """
        if 'results' not in batch:
            return batch

        modId = self.modality.get_name()  # Just shorter and easier to read
        if modId not in batch['results']:
            return batch

        for key in batch['results'][modId].keys():
            if isinstance(batch['results'][modId][key], torch.Tensor):
                batch['results'][modId][key] = batch['results'][modId][key].clone().cpu().detach()

        return batch
