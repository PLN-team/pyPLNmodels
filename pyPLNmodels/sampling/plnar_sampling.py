import torch

from .pln_sampling import PlnSampler


class PlnARSampler(PlnSampler):
    def __init__(
        self,
        n_samples: int = 100,
        dim: int = 20,
        *,
        nb_cov: int = 1,
        add_const: bool = True,
        add_offsets: bool = False,
        marginal_mean_mean: int = 2,
        seed: int = 0,
    ):  # pylint: disable=too-many-arguments
        self.autoregressive_matrix = torch.eye(dim) / 2
