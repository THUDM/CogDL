import torch
import torch.nn as nn

from ..inits import glorot, zeros


class SELayer(nn.Module):
    """Squeeze-and-excitation networks"""

    def __init__(self, in_channels, se_channels):
        super(SELayer, self).__init__()

        self.in_channels = in_channels
        self.se_channels = se_channels

        self.encoder_decoder = nn.Sequential(
            nn.Linear(in_channels, se_channels),
            nn.ELU(),
            nn.Linear(se_channels, in_channels),
            nn.Sigmoid(),
        )

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.encoder_decoder[0].weight)
        zeros(self.encoder_decoder[0].bias)
        glorot(self.encoder_decoder[2].weight)
        zeros(self.encoder_decoder[2].bias)

    def forward(self, x):
        """"""
        # Aggregate input representation
        x_global = torch.mean(x, dim=0)
        # Compute reweighting vector s
        s = self.encoder_decoder(x_global)

        return x * s
