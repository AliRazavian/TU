class Neural_Net_Set():

    def __init__(self, neural_nets):
        self.neural_nets = neural_nets

    def __call__(self, batch):
        self.forward(batch)

    def forward(self, batch):
        for neural_net in self.neural_nets:
            neural_net.forward(batch)

    def save(self, scene_name='last'):
        for neural_net in self.neural_nets:
            neural_net.save(scene_name)

    def load(self):
        for neural_net in self.neural_nets:
            neural_net.load()

    def zero_grad(self):
        for neural_net in self.neural_nets:
            neural_net.zero_grad()

    def step(self):
        for neural_net in self.neural_nets:
            neural_net.step()

    def train(self):
        for neural_net in self.neural_nets:
            neural_net.train()

    def eval(self):
        for neural_net in self.neural_nets:
            neural_net.eval()

    def update_learning_rate(self, learning_rate):
        for neural_net in self.neural_nets:
            neural_net.update_learning_rate(learning_rate)

    def update_stochastic_weighted_average_parameters(self):
        for neural_net in self.neural_nets:
            neural_net.update_stochastic_weighted_average_parameters()

    def prepare_for_batchnorm_update(self):
        for neural_net in self.neural_nets:
            neural_net.prepare_for_batchnorm_update()

    def update_batchnorm(self, batch):
        for neural_net in self.neural_nets:
            neural_net.update_batchnorm(batch)

    def finish_batchnorm_update(self):
        for neural_net in self.neural_nets:
            neural_net.finish_batchnorm_update()
