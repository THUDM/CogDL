import jittor
from jittor import nn,Module

class SELayer(Module):
    """Squeeze-and-excitation networks"""

    def __init__(self, in_channels, se_channels):
        super(SELayer, self).__init__()

        self.in_channels = in_channels
        self.se_channels = se_channels

        self.encoder_decoder = nn.Sequential(
            nn.Linear(in_channels, se_channels), nn.ELU(), nn.Linear(se_channels, in_channels), nn.Sigmoid(),
        )

        # self.reset_parameters()

    def execute(self, x):
        """"""
        # Aggregate input representation
        x_global = jittor.mean(x, dim=0)
        # Compute reweighting vector s
        s = self.encoder_decoder(x_global)

        return x * s
