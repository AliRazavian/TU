import time

from global_cfgs import Global_Cfgs
from Graphs.Factories.cascade_factory import Cascade_Factory
from Graphs.Factories.fully_connected_factory import Fully_Connected_Factory
from Graphs.Factories.fork_factory import Fork_Factory
from Graphs.Factories.morph_factory import Morph_Factory


class Model:

    def __init__(
            self,
            graph,
            graph_cfgs,
            scene_cfgs,
            task_cfgs,
            scenario_cfgs,
            experiment_set,
            model_name,
            model_cfgs,
    ):
        self.graph = graph
        self.graph_cfgs = graph_cfgs
        self.model_name = model_name
        self.model_cfgs = model_cfgs
        self.task_cfgs = task_cfgs
        self.scene_cfgs = scene_cfgs
        self.scenario_cfgs = scenario_cfgs

        self.experiment_set = experiment_set
        self.heads = model_cfgs['heads']
        self.tails = model_cfgs['tails']
        self.encoder = None
        self.decoder = None
        self.factory = self.get_factory()

        self.update_modality_dims()

    def encode(self, batch):
        start_time = time.time()
        encoder = self.get_encoder()
        encoder(batch)
        batch['time']['encode'][self.get_name()] = {'start': start_time, 'end': time.time()}

    def get_encoder(self):
        if self.encoder is None:
            self.encoder = self.init_neural_net('encoder')
        return self.encoder

    def decode(self, batch):
        start_time = time.time()
        decoder = self.get_decoder()
        decoder(batch)
        batch['time']['decode'][self.get_name()] = {'start': start_time, 'end': time.time()}

    def get_decoder(self):
        if self.decoder is None:
            self.decoder = self.init_neural_net('decoder')
        return self.decoder

    def dropNetworks(self):
        self.decoder = None
        self.encoder = None

    def init_neural_net(self, neural_net_type: str = 'encoder'):
        neural_net = self.factory.get_neural_net(
            model_cfgs=self.model_cfgs,
            heads={h.lower(): self.graph.nodes[h.lower()] for h in self.heads},
            tails={t.lower(): self.graph.nodes[t.lower()] for t in self.tails},
            neural_net_type=neural_net_type,
            optimizer_type=self.get_cfgs('optimizer_type'),
        )

        neural_net.update_learning_rate(self.initial_learning_rate)
        return neural_net

    def get_factory(self):
        neural_net_cfgs = self.get_cfgs('neural_net_cfgs')
        net_type = neural_net_cfgs['neural_net_type'].lower()
        if (net_type == 'Cascade'.lower()):
            return Cascade_Factory()
        elif (net_type == 'Fully_Connected'.lower()):
            return Fully_Connected_Factory()
        elif (net_type == 'Fork'.lower()):
            return Fork_Factory()
        elif (net_type == 'Morph'.lower()):
            return Morph_Factory()
        else:
            raise BaseException(f'Unsupported neural_net_type "{net_type}"')

    def zero_grad(self):
        try:
            if self.encoder is not None:
                self.encoder.zero_grad()
            if self.decoder is not None:
                self.decoder.zero_grad()
        except Exception as e:
            raise RuntimeError(f'Failed zero_grad for {self.get_name()}: {e}')

    def step(self):
        if self.encoder is not None:
            self.encoder.step()
        if self.decoder is not None:
            self.decoder.step()

    def update_learning_rate(self, learning_rate):
        if self.encoder is not None:
            self.encoder.update_learning_rate(learning_rate)
        else:
            self.initial_learning_rate = learning_rate
        if self.decoder is not None:
            self.decoder.update_learning_rate(learning_rate)
        else:
            self.initial_learning_rate = learning_rate

    def update_stochastic_weighted_average_parameters(self):
        has_run_average = False
        if self.encode is not None:
            encoder_has_run_average = self.encoder.update_stochastic_weighted_average_parameters()
            if encoder_has_run_average:
                has_run_average = True
        if self.decoder is not None:
            decoder_has_run_average = self.decoder.update_stochastic_weighted_average_parameters()
            if decoder_has_run_average:
                has_run_average = True

        return has_run_average

    def prepare_for_batchnorm_update(self):
        if self.encode is not None:
            self.encoder.prepare_for_batchnorm_update()
        if self.decoder is not None:
            self.decoder.prepare_for_batchnorm_update()

    def update_batchnorm(self, batch):
        if self.encode is not None:
            self.encoder.update_batchnorm(batch)
        if self.decoder is not None:
            self.decoder.update_batchnorm(batch)

    def finish_batchnorm_update(self):
        if self.encode is not None:
            self.encoder.finish_batchnorm_update()
        if self.decoder is not None:
            self.decoder.finish_batchnorm_update()

    def get_name(self):
        return self.model_name

    def get_cfgs(self, name, default=None):
        if name in self.model_cfgs:
            return self.model_cfgs[name]
        if name in self.graph_cfgs:
            return self.graph_cfgs[name]
        if name in self.task_cfgs:
            return self.task_cfgs[name]
        if name in self.task_cfgs['apply']:
            return self.task_cfgs['apply'][name]
        if name in self.scene_cfgs:
            return self.scene_cfgs[name]
        if name in self.scenario_cfgs:
            return self.scenario_cfgs[name]
        return Global_Cfgs().get(name, default)

    def train(self):
        """
        Activate train mode
        """
        if self.encoder is not None:
            self.encoder.train()
        if self.decoder is not None:
            self.decoder.train()

    def eval(self):
        if self.encoder is not None:
            self.encoder.eval()
        if self.decoder is not None:
            self.decoder.eval()

    def save(self, scene_name):
        if self.encoder is not None:
            self.encoder.save(scene_name)
        if self.decoder is not None:
            self.decoder.save(scene_name)

    def update_modality_dims(self):
        neural_net_cfgs = self.model_cfgs['neural_net_cfgs']
        self.factory.update_modality_dims(neural_net_cfgs=neural_net_cfgs,
                                          heads=self.heads,
                                          tails=self.tails,
                                          graph=self.graph)
