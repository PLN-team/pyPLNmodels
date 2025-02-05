import torch
from pyPLNmodels._utils import _add_doc

from .pln_sampling import _BasePlnSampler
from ._utils import _get_exog, _get_coef, _get_covariance, _get_offsets


class PlnDiagSampler(_BasePlnSampler):
    """Sampler for Poisson Log-Normal model with diagonal covariance.
    The parameters of the model are generated randomly but have a specific structure.

    Examples
    --------
    >>> from pyPLNmodels import PlnDiagSampler, PlnDiag
    >>> sampler = PlnDiagSampler()
    >>> endog = sampler.sample()
    >>> pln = PlnDiag(endog, exog = sampler.exog, add_const = False)
    >>> pln.fit()
    >>> estimated_cov = pln.covariance
    >>> true_covariance = sampler.covariance
    >>> estimated_latent_var = pln.latent_variables
    >>> true_latent_var = sampler.latent_variables
    """

    @_add_doc(_BasePlnSampler)
    def __init__(
        self,
        n_samples: int = 100,
        dim: int = 20,
        *,
        nb_cov: int = 1,
        add_const: bool = True,
        use_offsets: bool = False,
        marginal_mean_mean: int = 2,
    ):  # pylint: disable=too-many-arguments
        exog = _get_exog(n_samples=n_samples, nb_cov=nb_cov, will_add_const=add_const)
        offsets = _get_offsets(n_samples=n_samples, dim=dim, use_offsets=use_offsets)
        coef = _get_coef(
            nb_cov=nb_cov, dim=dim, mean=marginal_mean_mean, add_const=add_const
        )
        covariance = torch.diag(
            torch.diag(_get_covariance(dim) + torch.randn(dim) ** 2 / 4)
        )
        super().__init__(
            n_samples=n_samples,
            exog=exog,
            add_const=add_const,
            offsets=offsets,
            coef=coef,
            covariance=covariance,
        )

    @property
    def _dim_latent(self):
        return self.dim
