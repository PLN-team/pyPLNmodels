import torch

from pyPLNmodels._utils import _add_doc

from ._base_sampler import _BaseSampler
from ._utils import _get_exog, _get_coef, _get_covariance, _get_offsets


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class _BasePlnSampler(_BaseSampler):
    def __init__(
        self,
        *,
        n_samples: int,
        exog: torch.Tensor,
        offsets: torch.Tensor,
        coef: torch.Tensor,
        covariance: torch.Tensor,
    ):  # pylint: disable=too-many-arguments
        """
        Instantiate the model with the data given.

        Parameters
        ----------
        n_samples : int (keyword-only argument)
            Number of samples.
        exog : torch.Tensor (keyword-only argument)
            Covariates with size (n, d).
        offsets : torch.Tensor (keyword-only argument)
            Offsets with size (n, p).
        """
        params = {"coef": coef, "covariance": covariance}
        dim = covariance.shape[0]
        super().__init__(
            n_samples=n_samples, dim=dim, exog=exog, offsets=offsets, params=params
        )

    def _get_gaussians(self):
        centered_unit_gaussian = torch.randn(self._n_samples, self._dim_latent).to(
            DEVICE
        )
        components = self._get_components()
        mean = self._marginal_mean + self._offsets
        return torch.matmul(centered_unit_gaussian, components.T) + mean

    def _get_components(self):
        return torch.linalg.cholesky(self._params["covariance"])

    @property
    def _marginal_mean(self):
        if self._exog is None:
            return 0
        return torch.matmul(self._exog, self._params["coef"])

    @property
    def dict_model_true_parameters(self):
        """Alias for the parameters."""
        return self.params


class PlnSampler(_BasePlnSampler):
    """Sampler for Poisson Log-Normal model.
    The parameters of the model are generated randomly but have a specific structure."""

    @_add_doc(_BasePlnSampler)
    def __init__(
        self,
        n_samples: int = 100,
        dim: int = 50,
        *,
        nb_cov: int = 1,
        use_offsets: bool = False,
        marginal_mean: int = 2,
    ):  # pylint: disable=too-many-arguments
        exog = _get_exog(n_samples, nb_cov)
        offsets = _get_offsets(n_samples, dim, use_offsets)
        coef = _get_coef(nb_cov, dim, marginal_mean)
        covariance = _get_covariance(dim)
        super().__init__(
            n_samples=n_samples,
            exog=exog,
            offsets=offsets,
            coef=coef,
            covariance=covariance,
        )

    @property
    def _dim_latent(self):
        return self._dim
