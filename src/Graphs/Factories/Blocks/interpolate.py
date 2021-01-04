from torch import nn


class Interpolate(nn.Module):
    """
    Possible alternative to the Upsampling that creates warnings:
    https://discuss.pytorch.org/t/using-nn-function-interpolate-inside-nn-sequential/23588
    https://discuss.pytorch.org/t/which-function-is-better-for-upsampling-upsampling-or-interpolate/21811/12
    """

    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=None):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        x = self.interp(x, size=self.size, mode=self.mode, align_corners=self.align_corners)
        return x
