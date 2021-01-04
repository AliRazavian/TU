from torch import nn

from .basic_conv_block import Basic_Conv_Block


class ResNet_Basic_Block(nn.Module):
    """
    This is the bottleneck block for residual network
    input_c is the #input channels
    output_c is the #output channels
    group is the parameter added in resnext
    """

    def __init__(
            self,
            input_c,
            output_c,
            kernel_size=3,
            consistency="2D".lower(),
            add_relu=True,
            groups=1,
    ):
        super().__init__()
        self.input_c = input_c
        self.output_c = output_c
        self.kernel_size = kernel_size
        self.consistency = consistency
        self.groups = groups
        self.rescale = self.get_rescale()
        self.layers = self.get_layers()

    def get_rescale(self):
        if self.input_c == self.output_c:
            return None
        return Basic_Conv_Block(input_c=self.input_c,
                                output_c=self.output_c,
                                kernel_size=self.kernel_size,
                                consistency=self.consistency)

    def get_layers(self):
        hidden_size = max(8, self.input_c // 4)
        return nn.Sequential(
            Basic_Conv_Block(input_c=self.input_c,
                             output_c=hidden_size,
                             kernel_size=1,
                             consistency=self.consistency,
                             groups=self.groups),
            Basic_Conv_Block(input_c=hidden_size,
                             output_c=hidden_size,
                             kernel_size=self.kernel_size,
                             consistency=self.consistency,
                             groups=self.groups),
            Basic_Conv_Block(
                input_c=hidden_size,
                output_c=self.output_c,
                kernel_size=1,
                consistency=self.consistency,
                groups=self.groups,
            ))

    def forward(self, x):
        residual = x
        if self.rescale is not None:
            residual = self.rescale(x)
        out = self.layers(x)
        out += residual
        return out
