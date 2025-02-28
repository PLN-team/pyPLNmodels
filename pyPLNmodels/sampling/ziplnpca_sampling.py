import torch
from pyPLNmodels.utils._utils import _add_doc

from .zipln_sampling import ZIPlnSampler
from ._utils import _components_from_covariance
from ._base_sampler import _BaseSampler


class ZIPlnPCASampler(ZIPlnSampler):
    """
    Sampler for Zero-Inflated Poisson Log-Normal PCA model.
    The parameters of the model are generated randomly but have a specific structure.


    Examples
    --------
    >>> from pyPLNmodels import ZIPlnPCASampler, ZIPlnPCA
    >>> sampler = ZIPlnPCASampler()
    >>> endog = sampler.sample()
    >>> zipca = ZIPlnPCA(
    >>>        endog,
    >>>        exog=sampler.exog,
    >>>        exog_inflation=sampler.exog_inflation,
    >>>        add_const=False,
    >>>        add_const_inflation=False,
    >>>        rank=4
    >>>    )
    >>> zipca.fit()
    >>> estimated_cov = zipca.covariance
    >>> true_covariance = sampler.covariance
    >>> latent_probabilities = zipca.latent_prob
    >>> true_latent_probabilites = sampler.bernoulli

    See also
    --------
    :class:`pyPLNmodels.ZIPlnPCA`
    :class:`pyPLNmodels.ZIPln`
    :class:`pyPLNmodels.PlnPCA`
    """

    def __init__(
        self,
        n_samples: int = 100,
        dim: int = 20,
        *,
        nb_cov: int = 1,
        nb_cov_inflation: int = 1,
        add_const: bool = True,
        add_const_inflation: bool = True,
        add_offsets: bool = False,
        marginal_mean_mean: int = 2,
        marginal_mean_inflation_mean: int = 0.5,
        seed: int = 0,
        rank: int = 5,
    ):  # pylint: disable=too-many-arguments
        """
        Initialize the sampling class.

        Parameters
        ----------
        n_samples : int, optional
            Number of samples to generate (default is 100).
        dim : int, optional
            Dimensionality of the data (default is 20).
        nb_cov : int, optional, keyword-only
            Number of covariates (default is 1).
        add_const : bool, optional, keyword-only
            Whether to add a constant term to the covariates (default is True).
        add_offsets : bool, optional, keyword-only
            Whether to add offsets to the data (default is False).
        marginal_mean_mean : int, optional, keyword-only
            Mean of the marginal means (default is 2).
        seed : int, optional, keyword-only
            Random seed for reproducibility (default is 0).
        rank : int, optional, keyword-only
            The rank of the covariance matrix (default is 5).

        Examples
        --------
        >>> from pyPLNmodels import ZIPlnPCASampler
        >>> sampler = ZIPlnPCASampler()
        >>> endog = sampler.sample()
        """
        self.rank = rank
        super().__init__(
            n_samples=n_samples,
            nb_cov_inflation=nb_cov_inflation,
            dim=dim,
            nb_cov=nb_cov,
            add_const=add_const,
            add_const_inflation=add_const_inflation,
            add_offsets=add_offsets,
            marginal_mean_mean=marginal_mean_mean,
            marginal_mean_inflation_mean=marginal_mean_inflation_mean,
            seed=seed,
        )

    def _get_components(self):
        return _components_from_covariance(self._params["covariance"], self.rank)

    @property
    def covariance(self) -> torch.Tensor:
        """Covariance matrix."""
        components = self._get_components()
        return (components @ components.T).cpu()

    @_add_doc(
        _BaseSampler,
        example="""
        >>> from pyPLNmodels import ZIPlnPCASampler
        >>> sampler = ZIPlnPCASampler()
        >>> endog = sampler.sample()
        """,
    )
    def sample(self, seed: int = 0) -> torch.Tensor:
        return super().sample(seed=seed)

    @property
    def _dim_latent(self):
        return self.rank
