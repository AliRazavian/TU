from collections import OrderedDict
from torch import nn


class Basic_Conv_Block(nn.Module):
    """
    Basic block is just a convolution, followed by batchnorm and relu.
    This is the basic block of networks like VGG net
    """

    def __init__(
            self,
            input_c,
            output_c,
            kernel_size,
            consistency="2D".lower(),
            add_relu=True,
            groups=1,
    ):
        super().__init__()
        self.input_c = input_c
        self.output_c = output_c
        self.kernel_size = kernel_size
        self.consistency = consistency
        self.add_relu = add_relu

        if self.consistency.lower() == "1D".lower():
            self.Conv = nn.Conv1d
            self.Norm = nn.BatchNorm1d
        elif self.consistency.lower() == "2D".lower():
            self.Conv = nn.Conv2d
            self.Norm = nn.BatchNorm2d
        elif self.consistency.lower() == "3D".lower():
            self.Conv = nn.Conv3d
            self.Norm = nn.BatchNorm3d
        else:
            raise BaseException('Unknown consistency :%s' % self.consistency)
        self.layers = self.get_layer()

    def get_layer(self):
        layers = OrderedDict({})
        if self.add_relu:
            layers['relu'] = nn.ReLU()
        num_groups = int(self.input_c)
        if num_groups % 32 == 0:
            num_groups = 32
        # layers['batch_norm'] = nn.GroupNorm(num_groups,self.input_c)
        layers['batch_norm'] = self.Norm(self.input_c)
        layers['conv_%dx%dx%s' % (self.input_c,
                                  self.output_c,
                                  self.consistency)] =\
            self.Conv(self.input_c,
                      self.output_c,
                      kernel_size=self.kernel_size,
                      stride=1,
                      padding=self.kernel_size // 2,
                      bias=True)

        return nn.Sequential(layers)

    def forward(self, x):
        return self.layers(x)
