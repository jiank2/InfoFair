import torch.nn as nn
import torch.nn.functional as F

from layers.classifier import CLF
from layers.density_ratio_estimator import DensityRatioEstimator
from layers.feature_extractor import FeatureExtractor


class InfoFair(nn.Module):
    """
    our proposed method, InfoFair
    """

    def __init__(
        self, nfeat: int, nhids: dict, nclass: int, nsensitive: int, droprate: float
    ):
        super(InfoFair, self).__init__()
        # initialize feature extractor
        self.feature_extractor = FeatureExtractor(
            nfeat=nfeat, nhids=nhids["feature_extractor"], droprate=droprate
        )

        # initialize target predictor, sensitive attribute predictor and density ratio estimator
        if len(nhids["feature_extractor"]) > 0:
            self.target_predictor = CLF(
                nfeat=nhids["feature_extractor"][-1],
                nhids=nhids["classifier"],
                nclass=nclass,
                droprate=droprate,
            )
            self.sensitive_predictor = CLF(
                nfeat=nhids["feature_extractor"][-1],
                nhids=nhids["sensitive_classifier"],
                nclass=nsensitive,
                droprate=droprate,
            )
            self.density_ratio_estimator = DensityRatioEstimator(
                nfeat=nhids["feature_extractor"][-1], nsensitive=nsensitive
            )
        else:
            self.target_predictor = CLF(
                nfeat=nfeat, nhids=nhids["classifier"], nclass=nclass, droprate=droprate
            )
            self.sensitive_predictor = CLF(
                nfeat=nfeat,
                nhids=nhids["sensitive_classifier"],
                nclass=nsensitive,
                droprate=droprate,
            )
            self.density_ratio_estimator = DensityRatioEstimator(
                nfeat=nfeat, nsensitive=nsensitive
            )

    def forward(
        self,
        feature,
        sensitive_label,
        temperature=0.5,
        is_hard_gumbel_softmax=False,
        device="cuda",
    ):
        emb = self.feature_extractor(feature)
        class_log_prob = self.target_predictor(emb)
        sensitive_log_prob = self.sensitive_predictor(emb)
        sensitive_gumbel = F.gumbel_softmax(
            sensitive_log_prob, tau=temperature, hard=is_hard_gumbel_softmax
        )
        density_ratio_mean = self.density_ratio_estimator(
            emb, sensitive_label, sensitive_gumbel, device
        )
        return class_log_prob, sensitive_log_prob, density_ratio_mean
