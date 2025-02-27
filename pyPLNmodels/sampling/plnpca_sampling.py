import torch

from pyPLNmodels.utils._utils import _add_doc

from ._base_sampler import _BaseSampler
from .pln_sampling import PlnSampler
from ._utils import _components_from_covariance


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class PlnPCASampler(PlnSampler):
    """Sampler for Poisson Log-Normal PCA model.
    The parameters of the model are generated randomly.

    Examples
    --------
    >>> from pyPLNmodels import PlnPCASampler, PlnPCA
    >>> sampler = PlnPCASampler()
    >>> endog = sampler.sample()
    >>> pca = PlnPCA(endog, exog = sampler.exog, add_const = False)
    >>> pca.fit()
    >>> estimated_cov = pca.covariance
    >>> true_covariance = sampler.covariance
    >>> estimated_latent_var = pca.latent_variables
    >>> true_latent_var = sampler.latent_variables

    See also
    --------
    :class:`pyPLNmodels.PlnPCA`
    :class:`pyPLNmodels.ZIPlnPCA`
    """

    def __init__(
        self,
        n_samples: int = 100,
        dim: int = 20,
        *,
        nb_cov: int = 1,
        add_const: bool = True,
        add_offsets: bool = False,
        marginal_mean_mean: int = 2,
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
        >>> from pyPLNmodels import PlnPCASampler
        >>> sampler = PlnPCASampler()
        >>> endog = sampler.sample()
        """
        self.rank = rank
        super().__init__(
            n_samples=n_samples,
            dim=dim,
            nb_cov=nb_cov,
            add_offsets=add_offsets,
            marginal_mean_mean=marginal_mean_mean,
            add_const=add_const,
        )

    def _get_components(self):
        return _components_from_covariance(self._params["covariance"], self.rank)

    @property
    def _dim_latent(self):
        return self.rank

    @property
    def covariance(self) -> torch.Tensor:
        """Covariance matrix."""
        components = self._get_components()
        return (components @ components.T).cpu()

    @_add_doc(
        _BaseSampler,
        example="""
        >>> from pyPLNmodels import PlnPCASampler
        >>> sampler = PlnPCASampler()
        >>> endog = sampler.sample()
        """,
    )
    def sample(self, seed: int = 0) -> torch.Tensor:
        return super().sample(seed=seed)
