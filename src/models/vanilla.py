import torch.nn as nn

from layers.classifier import CLF
from layers.feature_extractor import FeatureExtractor


class Vanilla(nn.Module):
    """
    a vanilla MLP with same initialization way as InfoFair
    """

    def __init__(self, nfeat, nhids, nclass, droprate):
        super(Vanilla, self).__init__()
        # initialize feature extractor
        self.feature_extractor = FeatureExtractor(
            nfeat=nfeat, nhids=nhids["feature_extractor"], droprate=droprate
        )
        # initialize target predictor
        if len(nhids["feature_extractor"]) > 0:
            self.target_predictor = CLF(
                nfeat=nhids["feature_extractor"][-1],
                nhids=nhids["classifier"],
                nclass=nclass,
                droprate=droprate,
            )
        else:
            self.target_predictor = CLF(
                nfeat=nfeat, nhids=nhids["classifier"], nclass=nclass, droprate=droprate
            )

    def forward(self, feature):
        emb = self.feature_extractor(feature)
        class_log_prob = self.target_predictor(emb)
        return class_log_prob
