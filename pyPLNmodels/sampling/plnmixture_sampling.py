# pylint: skip-file
import torch
from ._utils import _get_coef


class PlnMixtureSampler:
    def __init__(
        self, nb_cluster=3, dim=20, n_samples=100, ecart_clusters=1.0, add_const=True
    ):
        self.nb_cluster = nb_cluster
        self.dim = dim
        self.n_samples = n_samples
        self.cluster_probs = torch.rand(nb_cluster)
        self.cluster_probs /= self.cluster_probs.sum()
        self.coefs = {}
        for i in range(self.nb_clusters):
            self.coefs[i] = _get_coef(
                nb_cov=nb_cov, dim=dim, mean=i, add_const=add_const
            )
