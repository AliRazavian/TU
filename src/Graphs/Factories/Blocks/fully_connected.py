from collections import OrderedDict
import numpy as np
from torch import nn


class Fully_Connected(nn.Module):
    """
    FC block (fully connected block creates a multilayer (or single layer)
    neural network with dropout.
    layer_shapes simply tells what the input, hidden and output layers should
    look like, th
    """

    def __init__(
            self,
            input_shape,
            output_shape,
            num_hidden_layers=1,
            add_relu=True,
    ):
        super().__init__()

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.add_relu = add_relu

        self.input_volume = np.prod(np.array(input_shape))
        self.output_volume = np.prod(np.array(output_shape))

        hidden_volumes = np.linspace(np.log2(self.input_volume), np.log2(self.output_volume),
                                     num_hidden_layers + 2, dtype='float32')[1:-1]
        hidden_volumes = 2**np.round(hidden_volumes)
        self.layer_volumes = [self.input_volume] + \
            list(hidden_volumes) + [self.output_volume]

        self.layers = self.get_layer()

    def get_layer(self):
        layers = OrderedDict({})
        for i in range(len(self.layer_volumes) - 1):
            if self.add_relu:
                layers['relu_%d' % (i)] = nn.ReLU(inplace=True)
            layers['linear_%d' % (i)] = \
                nn.Linear(in_features=int(self.layer_volumes[i]),
                          out_features=int(self.layer_volumes[i + 1]),
                          bias=True)
            self.add_relu = True
        return nn.Sequential(layers)

    def forward(self, x):
        batch_size = x.shape[0]
        y = self.layers(x.view(batch_size, self.input_volume))
        return y.view(batch_size, *self.output_shape)
