from torch import nn


class Squeeze_and_Excite_2D(nn.Module):
    """
    Squeeze and Excite plays as some sort of non-binary switches
    on the feature maps.
    This block does not change the shape of the tensor
    """

    def __init__(
            self,
            input_c,
            reduction=16,
    ):
        super().__init__()
        self.input_c = input_c
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.hidden_size = max(self.input_c // reduction, 4)
        self.layer = self.get_layer()
        self.relu = nn.ReLU()

    def get_layer(self):
        return nn.Sequential(nn.Linear(self.input_c, self.hidden_size), nn.ReLU(inplace=True),
                             nn.Linear(self.hidden_size, self.input_c), nn.Sigmoid())

    def forward(self, x):
        batch_size = x.shape[0]
        avg_features = self.avg_pool(x).view(batch_size, self.input_c)
        y = self.layer(avg_features).view(batch_size, self.input_c, 1, 1)
        return self.relu(x * y)
