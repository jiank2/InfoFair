import torch
import torch.nn as nn

from utils import utils


class DensityRatioEstimator(nn.Module):
    """
    logistic regression as estimator
    """

    def __init__(self, nfeat, nsensitive):
        super(DensityRatioEstimator, self).__init__()
        self.nsensitive = nsensitive  # number of sensitive attribute
        self.linear_emb = nn.Linear(nfeat, 1, bias=False)  # linear layer for embedding
        self.linear_sensi = nn.Linear(nsensitive, 1, bias=False)  # linear layer for sensitive attribute
        self.sigmoid_layer = nn.Sigmoid()
        
    def forward(self, emb, sensitive_group, sensitive_gumbel, device):
        emb_wx = self.linear_emb(emb)

        # logits for positive points: embedding with true sensitive group label
        pos_sensi_label = torch.FloatTensor(
            utils.label_to_one_hot(sensitive_group, self.nsensitive)
        ).to(device)
        pos_logits = emb_wx + self.linear_sensi(pos_sensi_label)

        # logits for negative points: embedding with predicted sensitive group label (estimated using gumbel softmax)
        neg_logits = emb_wx + self.linear_sensi(sensitive_gumbel)

        return torch.mean(pos_logits + neg_logits) / 2
