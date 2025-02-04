import torch

from pyPLNmodels._utils import _add_doc
from pyPLNmodels._data_handler import _add_constant_to_exog

from ._base_sampler import _BaseSampler
from .pln_sampling import PlnSampler
from ._utils import _get_exog, _get_coef


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ZIPlnSampler(PlnSampler):
    """Sampler for Zero-Inflated Poisson Log-Normal model.
    The parameters of the model are generated randomly but have a specific structure.


    Examples
    --------
    >>> from pyPLNmodels import ZIPlnSampler, ZIPln
    >>> sampler = ZIPlnSampler()
    >>> endog = sampler.sample()
    >>> zi = ZIPln(endog, exog = sampler.exog, exog_inflation = sampler.exog_inflation, add)
    >>> zi.fit()
    >>> estimated_cov = zi.covariance
    >>> true_covariance = sampler.covariance
    >>> latent_probabilities = zi.latent_prob
    >>> true_latent_probabilites = sampler.bernoulli
    """

    bernoulli: torch.Tensor

    def __init__(
        self,
        n_samples: int = 100,
        dim: int = 20,
        *,
        nb_cov: int = 1,
        nb_cov_inflation: int = 1,
        add_const: bool = True,
        add_const_inflation: bool = True,
        use_offsets: bool = False,
        marginal_mean_mean: int = 2,
        marginal_mean_inflation_mean: int = 0.5,
    ):  # pylint: disable=too-many-arguments
        super().__init__(
            n_samples,
            dim,
            nb_cov=nb_cov,
            add_const=add_const,
            use_offsets=use_offsets,
            marginal_mean_mean=marginal_mean_mean,
        )
        if nb_cov_inflation == 0 and add_const_inflation is False:
            raise ValueError("Number of covariates should be positive.")
        self._exog_inflation = _get_exog(
            n_samples=n_samples,
            nb_cov=nb_cov_inflation,
            will_add_const=add_const_inflation,
        )
        if add_const_inflation is True:
            self._exog_inflation = _add_constant_to_exog(
                self._exog_inflation, self.n_samples
            )

        coef_inflation = _get_coef(
            nb_cov_inflation,
            dim,
            marginal_mean_inflation_mean,
            add_const=add_const_inflation,
        )
        self._params["coef_inflation"] = coef_inflation

    @property
    def _marginal_mean_inflation(self):
        return torch.matmul(self._exog_inflation, self._params["coef_inflation"])

    @_add_doc(_BaseSampler)
    def sample(self, seed: int = 0) -> torch.Tensor:
        endog_not_inflated = super().sample()
        self.bernoulli = torch.bernoulli(
            torch.sigmoid(self._marginal_mean_inflation)
        ).to("cpu")
        return endog_not_inflated * (1 - self.bernoulli)

    @property
    def exog_inflation(self):
        """Exogenous variables (i.e. covariates) of the zero inflation part ."""
        return self._exog_inflation.cpu()

    @property
    def coef_inflation(self) -> torch.Tensor:
        """Coefficient matrix for the zero inflation part."""
        return self._params.get("coef_inflation")
