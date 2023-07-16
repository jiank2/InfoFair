import math

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class MLPLayer(nn.Module):
    """
    a typical fully connected layer
    """

    def __init__(self, in_features, out_features, bias=True):
        super(MLPLayer, self).__init__()
        self.in_features = in_features  # input feature dimension
        self.out_features = out_features  # output feature dimension
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))  # weight params
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))  # bias
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.out_features)
        self.weight.data.normal_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.normal_(-stdv, stdv)

    def forward(self, input):
        output = torch.mm(input, self.weight)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )
