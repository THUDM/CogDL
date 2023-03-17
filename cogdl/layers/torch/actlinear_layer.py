import torch.nn as nn

from actnn.conf import config
from actnn.qscheme import QScheme

from cogdl.operators.linear import linear


class QLinear(nn.Linear):
    num_layers = 0

    def __init__(self, input_features, output_features, bias=True, group=0, rp_ratio=2):
        super(QLinear, self).__init__(input_features, output_features, bias)
        if config.adaptive_conv_scheme:
            self.scheme = QScheme(self, group=group)
        else:
            self.scheme = None
        self.rp_ratio = rp_ratio

    def forward(self, input):
        if config.training:
            return linear.apply(input, self.weight, self.bias, self.scheme, self.rp_ratio)
        else:
            return super(QLinear, self).forward(input)
