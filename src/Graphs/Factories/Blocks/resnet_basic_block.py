from torch import nn

from .basic_conv_block import Basic_Conv_Block


class ResNet_Basic_Block(nn.Module):
    """
    This is the basic block for residual network
    input_c is the #input channels
    output_c is the #output channels
    resnext refers to the next generation of resnet:
    stride indicates the downsample ratio.
    if input is WxH then with stride 2, output would be W/2 x H/2
    negative stride means upsampling instead of downsampling.
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
        self.add_relu = add_relu
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
                                consistency=self.consistency,
                                add_relu=self.add_relu)

    def get_layers(self):
        return nn.Sequential(
            Basic_Conv_Block(input_c=self.input_c,
                             output_c=self.input_c,
                             kernel_size=self.kernel_size,
                             consistency=self.consistency,
                             groups=self.groups,
                             add_relu=self.add_relu),
            Basic_Conv_Block(input_c=self.input_c,
                             output_c=self.output_c,
                             kernel_size=self.kernel_size,
                             consistency=self.consistency,
                             groups=self.groups,
                             add_relu=self.add_relu))

    def forward(self, x):
        residual = x
        if self.rescale is not None:
            residual = self.rescale(x)
        out = self.layers(x)
        out += residual
        return out
