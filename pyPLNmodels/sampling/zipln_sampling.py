import torch

from pyPLNmodels._utils import _add_doc

from ._base_sampler import _BaseSampler
from .pln_sampling import PlnSampler
from ._utils import _get_exog, _get_coef


class ZIPlnSampler(PlnSampler):
    """Sampler for Zero-Inflated Poisson Log-Normal model.
    The parameters of the model are generated
    randomly but have a specific structure.
    """

    bernoulli: torch.Tensor

    def __init__(
        self,
        n_samples: int = 100,
        dim: int = 50,
        *,
        nb_cov: int = 1,
        nb_cov_inflation: int = 1,
        use_offsets: bool = False,
        marginal_mean_mean: int = 2,
        marginal_mean_inflation_mean: int = 0.5,
    ):  # pylint: disable=too-many-arguments
        super().__init__(
            n_samples,
            dim,
            nb_cov=nb_cov,
            use_offsets=use_offsets,
            marginal_mean_mean=marginal_mean_mean,
        )
        if nb_cov_inflation == 0:
            raise ValueError("Number of covariates should should be positive.")
        self._exog_inflation = _get_exog(n_samples, nb_cov_inflation)
        coef_inflation = _get_coef(nb_cov_inflation, dim, marginal_mean_inflation_mean)
        self._params["coef_inflation"] = coef_inflation

    @property
    def _marginal_mean_inflation(self):
        return torch.matmul(self._exog_inflation, self._params["coef_inflation"])

    @_add_doc(_BaseSampler)
    def sample(self, seed: int = 0) -> torch.Tensor:
        endog_not_inflated = super().sample()
        self.bernoulli = torch.bernoulli(torch.sigmoid(self._marginal_mean_inflation))
        return endog_not_inflated * (1 - self.bernoulli)

    @property
    def exog_inflation(self):
        """Exogenous variables (i.e. covariates) of the zero inflation part ."""
        return self._exog_inflation.cpu()

    @property
    def coef_inflation(self) -> torch.Tensor:
        """Coefficient matrix for the zero inflation part."""
        return self._params.get("coef_inflation")
