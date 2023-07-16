import torch.nn as nn
import torch.nn.functional as F


class FeatureExtractor(nn.Module):
    """
    n-layer MLP for feature extraction
    """

    def __init__(self, nfeat, nhids, droprate):
        super(FeatureExtractor, self).__init__()
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
        self.droprate = droprate

    def forward(self, x):
        for i in range(self.nlayers):
            x = F.relu(self.layers[i](x))
            if self.droprate != 0.0:
                x = F.dropout(x, self.droprate, training=self.training)
        return x
