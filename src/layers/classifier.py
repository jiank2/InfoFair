import torch.nn as nn
import torch.nn.functional as F


class CLF(nn.Module):
    """
    n-layer MLP as classifier, can be used as critic function
    """

    def __init__(self, nfeat, nhids, nclass, droprate):
        super(CLF, self).__init__()
        self.nlayers = len(nhids)  # number of hidden layers
        self.layers = nn.ModuleList()  # list of hidden layers
        self.in_features = nfeat  # input feature dimension
        for i in range(self.nlayers):
            self.layers.append(
                nn.Linear(
                    in_features=self.in_features, out_features=nhids[i], bias=True
                )
            )
            self.in_features = nhids[i]

        self.layers.append(
            nn.Linear(in_features=self.in_features, out_features=nclass, bias=False)
        )
        self.droprate = droprate

    def forward(self, x):
        for i in range(self.nlayers):
            x = F.relu(self.layers[i](x))
            if self.droprate != 0.0 and i != (self.nlayers - 1):
                x = F.dropout(x, self.droprate, training=self.training)
        x = F.log_softmax(self.layers[-1](x), dim=1)
        return x
