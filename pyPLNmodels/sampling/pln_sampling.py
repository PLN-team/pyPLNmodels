import torch


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
    ):  # pylint: disable=too-many-arguments, too-many-positional-arguments
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
        centered_unit_gaussian = torch.randn(self._n_samples, self._dim).to(DEVICE)
        components = torch.linalg.cholesky(self._params["covariance"])
        mean = torch.matmul(self._exog, self._params["coef"]) + self._offsets
        return torch.matmul(centered_unit_gaussian, components.T) + mean


class PlnSampler(_BasePlnSampler):
    """Sampler for Poisson Log-Normal model. The parameters of the model are generated randomly."""

    def __init__(
        self,
        n_samples: int = 100,
        dim: int = 50,
        *,
        nb_cov: int = 1,
        use_offsets: bool = False,
        mean_gaussian: int = 2,
    ):  # pylint: disable=too-many-arguments, too-many-positional-arguments
        exog = _get_exog(n_samples, nb_cov)
        offsets = _get_offsets(n_samples, dim, use_offsets)
        coef = _get_coef(nb_cov, dim, mean_gaussian)
        covariance = _get_covariance(dim)
        super().__init__(
            n_samples=n_samples,
            exog=exog,
            offsets=offsets,
            coef=coef,
            covariance=covariance,
        )
