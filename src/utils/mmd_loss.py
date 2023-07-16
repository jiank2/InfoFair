import torch
import torch.nn as nn


class MMDLoss(nn.Module):
    """
    Unbiased estimation of MMD
    Adopted from https://github.com/josipd/torch-two-sample/blob/master/torch_two_sample/statistics_diff.py
    """

    def __init__(self, bandwidth=2.0):
        super(MMDLoss, self).__init__()
        self.bandwidth = bandwidth
        self.alpha_lst = [1.0 / bandwidth]

    def _pairwise_dist(self, sample_1, sample_2, norm=2.0, eps=1e-5):
        """
        compute the matrix of all squared pairwise distances.
        :param sample_1: torch tensor of shape (n_1, d)
        :param sample_2: torch tensor of shape (n_2, d)
        :param norm: Lp norm of distance
        :param eps: small constant

        :return: squared pairwise distance between two data points
        """
        n_1, n_2 = sample_1.size(0), sample_2.size(0)
        if norm == 2.0:
            norms_1 = torch.sum(sample_1**2, dim=1, keepdim=True)
            norms_2 = torch.sum(sample_2**2, dim=1, keepdim=True)
            norms = norms_1.expand(n_1, n_2) + norms_2.transpose(0, 1).expand(n_1, n_2)
            distances_squared = norms - 2 * sample_1.mm(sample_2.t())
            return torch.sqrt(eps + torch.abs(distances_squared))
        else:
            dim = sample_1.size(1)
            expanded_1 = sample_1.unsqueeze(1).expand(n_1, n_2, dim)
            expanded_2 = sample_2.unsqueeze(0).expand(n_1, n_2, dim)
            differences = torch.abs(expanded_1 - expanded_2) ** norm
            inner = torch.sum(differences, dim=2, keepdim=False)
            return (eps + inner) ** (1.0 / norm)

    def forward(self, x_0, x_1, alpha_lst=None):
        # get number of samples in each demographic groups
        n_0, n_1 = x_0.size()[0], x_1.size()[0]
        # calculate coefficients for MMD estimation
        try:
            a_00 = 1.0 / (n_0 * (n_0 - 1))
        except:
            a_00 = 0.0
        try:
            a_11 = 1.0 / (n_1 * (n_1 - 1))
        except:
            a_11 = 0.0
        try:
            a_01 = -1.0 / (n_0 * n_1)
        except:
            a_01 = 0.0

        # get pariwise distance
        pdist = self._pairwise_dist(x_0, x_1, norm=2)

        # compute gram matrix
        kernels = None
        alpha_lst = alpha_lst if alpha_lst else [1.0 / self.bandwidth]
        for alpha in alpha_lst:
            kernels_a = torch.exp(-alpha * pdist**2)
            if kernels is None:
                kernels = kernels_a
            else:
                kernels = kernels + kernels_a

        # get submatrix in the gram matrix
        k_1 = kernels[:n_0, :n_0]
        k_2 = kernels[n_0:, n_0:]
        k_12 = kernels[:n_0, n_0:]

        # estimate MMD
        mmd = (
            2 * a_01 * k_12.sum()
            + a_00 * (k_1.sum() - torch.trace(k_1))
            + a_11 * (k_2.sum() - torch.trace(k_2))
        )
        return mmd
