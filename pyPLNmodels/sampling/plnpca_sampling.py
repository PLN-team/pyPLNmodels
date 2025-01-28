import torch


from .pln_sampling import PlnSampler
from ._utils import _components_from_covariance


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class PlnPCASampler(PlnSampler):
    """Sampler for Poisson Log-Normal model. The parameters of the model are generated randomly."""

    def __init__(
        self,
        n_samples: int = 100,
        dim: int = 20,
        rank: int = 5,
        *,
        nb_cov: int = 1,
        add_const: bool = True,
        use_offsets: bool = False,
        marginal_mean_mean: int = 2,
    ):  # pylint: disable=too-many-arguments
        self._rank = rank
        super().__init__(
            n_samples=n_samples,
            dim=dim,
            nb_cov=nb_cov,
            use_offsets=use_offsets,
            marginal_mean_mean=marginal_mean_mean,
            add_const=add_const,
        )

    def _get_components(self):
        return _components_from_covariance(self._params["covariance"], self._rank)

    @property
    def rank(self):
        """Rank of the covariance matrix."""
        return self._rank

    @property
    def _dim_latent(self):
        return self._rank

    @property
    def covariance(self) -> torch.Tensor:
        """Covariance matrix."""
        components = self._get_components()
        return (components @ components.T).cpu()
