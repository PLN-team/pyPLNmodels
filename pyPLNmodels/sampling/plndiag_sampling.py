import torch
from pyPLNmodels.utils._utils import _add_doc

from ._base_sampler import _BaseSampler
from .pln_sampling import PlnSampler
from ._utils import _get_diag_covariance

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class PlnDiagSampler(PlnSampler):
    """
    Sampler for Poisson Log-Normal model with diagonal covariance.
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

    See also
    --------
    :class:`pyPLNmodels.PlnDiag`
    """

    @_add_doc(
        PlnSampler,
        example="""
        >>> from pyPLNmodels import PlnDiagSampler
        >>> sampler = PlnDiagSampler()
        >>> endog = sampler.sample()
        """,
    )
    def __init__(
        self,
        n_samples: int = 100,
        dim: int = 20,
        *,
        nb_cov: int = 1,
        add_const: bool = True,
        add_offsets: bool = False,
        marginal_mean_mean: int = 2,
        seed=0,
    ):  # pylint: disable=too-many-arguments
        super().__init__(
            n_samples=n_samples,
            dim=dim,
            nb_cov=nb_cov,
            add_const=add_const,
            add_offsets=add_offsets,
            marginal_mean_mean=marginal_mean_mean,
            seed=seed,
        )

    def _get_covariance(self, dim, seed):
        return _get_diag_covariance(dim, seed=seed)

    @property
    def _dim_latent(self):
        return self.dim

    def _get_gaussians(self, seed):
        torch.manual_seed(seed)
        centered_unit_gaussian = torch.randn(self.n_samples, self._dim_latent).to(
            DEVICE
        )
        mean = self._marginal_mean
        return centered_unit_gaussian * torch.sqrt(self._params["covariance"]) + mean

    @_add_doc(
        _BaseSampler,
        example="""
        >>> from pyPLNmodels import PlnDiagSampler
        >>> sampler = PlnDiagSampler()
        >>> endog = sampler.sample()
        """,
    )
    def sample(self, seed: int = 0) -> torch.Tensor:
        return super().sample(seed=seed)

    def _get_components(self):
        return torch.sqrt(self.covariance)
