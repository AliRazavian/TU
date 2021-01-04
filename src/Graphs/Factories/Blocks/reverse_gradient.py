import torch.nn as nn


class Reverse_Gradient(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Apparently, pytorch compiler or interpreter bypasses
        # modules do not change the input. having a code that
        # pretend we are changing the input in the forward pass
        # keeps the backward method in the loop
        return x.view_as(x)

    def backward(self, grad_output):
        return grad_output * -1
