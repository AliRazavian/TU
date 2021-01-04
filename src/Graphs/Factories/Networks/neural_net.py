import torch
from torch import nn
import torch.optim as optim
import numpy as np

from global_cfgs import Global_Cfgs
from .helpers import summarizeModelSize
from file_manager import File_Manager
from UIs.console_UI import Console_UI


class Neural_Net(nn.Module):

    def __init__(
            self,
            neural_net_name,
            neural_net_cfgs,
            layers,
            optimizer_type: str = 'sgd',
            input_name: str = '',
            output_name: str = '',
            input_shape: list = [],
            output_shape: list = [],
            load_from_batch=True,
            add_noise=False,
    ):
        super().__init__()
        self.neural_net_name = neural_net_name
        self.neural_net_cfgs = neural_net_cfgs
        self.input_name = input_name
        self.output_name = output_name
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.layers = layers
        self.optimizer_type = optimizer_type
        self.add_noise = add_noise
        self.load_from_batch = load_from_batch
        self.weighted_average_parameters = None
        self.weighted_average_parameters_counter = 0
        self.batch_norm_update_counter = 0
        self.momenta = {}

        if self.load_from_batch:
            self.forward = self.forward_from_batch
        else:
            self.forward = self.forward_data

        if Global_Cfgs().get('DEVICE_BACKEND') == 'cuda':
            self.layers.cuda()
            self.layers = nn.DataParallel(layers)

        self.network_memory_usage = None
        try:
            self.network_memory_usage = summarizeModelSize(
                model=layers,
                input_size=(*self.input_shape,),
                device=Global_Cfgs().get('DEVICE_BACKEND'),
            )
        except Exception as e:
            Console_UI().warn_user(f'Failed to get size for {neural_net_name}: {e}')
            pass

        Console_UI().debug(self.layers)
        self.optimizer = None
        self.optimizer = self.get_optimizer()

        self.load()

    def save(self, scene_name='last'):
        File_Manager().save_pytorch_neural_net(self.get_name(), self, scene_name)

    def get_forward_noise(self):
        if not self.training:
            return 0.0
        if np.random.rand() < 0.5:
            return 0.0
        return Global_Cfgs().forward_noise * np.random.rand()

    def load(self):
        neural_net_name = self.get_name()
        state_dict = File_Manager().load_pytorch_neural_net(neural_net_name=neural_net_name)
        if state_dict is not None:
            try:
                # As the save is done at the layers level: neural_net.layers.state_dict()
                # we need to load it from the layers
                self.layers.load_state_dict(state_dict)
            except RuntimeError as e:
                raise RuntimeError(f'Failed to load dictionary for {neural_net_name} \nError message: {e}')

    def get_name(self):
        return self.neural_net_name

    def forward_data(self, x):
        x = x.view([-1, *self.input_shape])
        noise = self.get_forward_noise() * torch.randn_like(x, device=x.device)
        if x.min() >= 0:  # TODO: is the minimum expensive to calculate?
            noise = nn.functional.relu(noise)

        y = self.layers(x + noise)
        return y.view([-1, *self.output_shape])

    def forward_from_batch(self, batch):
        x = batch[self.input_name.lower()]
        y = self.forward_data(x)

        if (self.output_name.lower() in batch):
            # This merge is only happening in morph_visual_morph...
            batch[self.output_name.lower()] += y
        else:
            batch[self.output_name.lower()] = y

    def get_optimizer(self):
        if (len(list(self.parameters())) > 0):
            if (self.optimizer_type.lower() == 'Adam'.lower()):
                return optim.Adam(self.parameters(), weight_decay=1e-6)
            elif (self.optimizer_type.lower() == 'SGD'.lower()):
                return optim.SGD(self.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-4)
            else:
                raise BaseException('Unknown Optimizer type %s' % self.optimizer_type)
        self.zero_grad()
        return None

    def update_learning_rate(self, learning_rate):
        if self.optimizer:
            if (self.optimizer_type.lower() == 'Adam'.lower()):
                if (learning_rate > 1e-3):
                    Console_UI().warn_user(f'learning rate {learning_rate:.2e} for Adam is too big.' +
                                           ' We recommend a learning rate of less than 1e-3')
            elif (self.optimizer_type.lower() == 'SGD'.lower()):
                if (learning_rate > 1e-1):
                    Console_UI().warn_user(f'learning rate {learning_rate:.2e} for Adam is too big.' +
                                           ' We recommend a learning rate of less than 1e-1')

            for param_group in self.optimizer.param_groups:
                param_group['lr'] = learning_rate
                param_group['weight_decay'] = learning_rate / 200

    def zero_grad(self):
        if self.optimizer:
            self.optimizer.zero_grad()

    def step(self):
        if self.optimizer:
            try:
                self.optimizer.step()
            except RuntimeError as e:
                Console_UI().warn_user(f'Failed to optimize {self.get_name()}- a masking issue? {e}')
                pass

    def update_stochastic_weighted_average_parameters(self):
        """
        This allows us to find a wider local minima
        """
        # Before saving the parameters we remove the backprop gradient
        self.zero_grad()
        self.weighted_average_parameters_counter += 1

        weight_has_been_updated = False
        if self.weighted_average_parameters is not None:
            alpha = 1. / self.weighted_average_parameters_counter
            for self_params, prev_params in zip(self.parameters(), self.weighted_average_parameters):
                # TODO: change to numpy or move to cpu
                self_params.data = (1. - alpha) * prev_params.to(self_params.data.device) +\
                                         alpha * self_params.data  # noqa - align ok
            weight_has_been_updated = True

        # Save the updated parameters
        self.weighted_average_parameters = [p.data.clone().detach().cpu() for p in self.parameters()]

        return weight_has_been_updated

    def __check_bn(self, module, flag):
        if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
            flag[0] = True

    def check_bn(self):
        flag = [False]
        self.apply(lambda module: self.__check_bn(module, flag))
        return flag[0]

    def reset_bn(self, module):
        if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
            module.running_mean = torch.zeros_like(module.running_mean)
            module.running_var = torch.ones_like(module.running_var)

    def __get_momenta(self, module, momenta):
        if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
            momenta[module] = module.momentum

    def __set_momenta(self, module, momenta):
        if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
            module.momentum = momenta[module]

    def prepare_for_batchnorm_update(self):

        # this function is not complete yet

        if not self.check_bn():
            return
        self.train()

        self.apply(self.reset_bn)
        self.apply(lambda module: self.__get_momenta(module, self.momenta))
        self.batch_norm_update_counter = 0
        return self.momenta

    def finish_batchnorm_update(self):
        self.apply(lambda module: self.__set_momenta(module, self.momenta))

    def update_batchnorm(self, x):

        if (isinstance(x, dict)):
            batch_size = x[self.input_name.lower()].shape[0]
        else:
            batch_size = x.shape[0]

        momentum = batch_size / (self.batch_norm_update_counter + batch_size)
        for module in self.momenta.keys():
            module.momentum = momentum

        self.batch_norm_update_counter += batch_size
