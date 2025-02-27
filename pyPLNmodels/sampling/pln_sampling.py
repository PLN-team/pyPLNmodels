from abc import abstractmethod, ABC

import torch

from pyPLNmodels.utils._utils import _add_doc

from ._base_sampler import _BaseSampler
from ._utils import _get_exog, _get_coef, _get_full_covariance, _get_offsets


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class _BasePlnSampler(_BaseSampler, ABC):

    def __init__(
        self, *, n_samples, exog, add_const, offsets, coef, covariance
    ):  # pylint:disable = too-many-instance-attributes,too-many-arguments
        """
        Initalize the data and parameters of the model.

        Parameters
        ----------
        n_samples : int
            Number of samples to generate.
        exog : torch.Tensor
            Exogenous variables (covariates) tensor of size(n_samples, nb_cov).
        add_const : bool
            Whether to add a constant term to the covariates.
        offsets : torch.Tensor
            Offsets to be added to the data of size (n_samples, dim).
        coef : torch.Tensor
            Coefficients for the covariates of size (nb_cov, dim).
        covariance : torch.Tensor
            Covariance matrix of the data of size (dim, dim).
        """
        params = {"coef": coef, "covariance": covariance}
        dim = covariance.shape[0]
        super().__init__(
            n_samples=n_samples,
            dim=dim,
            exog=exog,
            add_const=add_const,
            offsets=offsets,
            params=params,
        )

    def _get_gaussians(self, seed):
        torch.manual_seed(seed)
        centered_unit_gaussian = torch.randn(self.n_samples, self._dim_latent).to(
            DEVICE
        )
        components = self._get_components()
        mean = self._marginal_mean
        return torch.matmul(centered_unit_gaussian, components.T) + mean

    @property
    @abstractmethod
    def _dim_latent(self):
        """Dimension that is sampling in the latent space."""

    def _get_components(self):
        return torch.linalg.cholesky(self._params["covariance"])

    @property
    def _marginal_mean(self):
        if self._exog is None:
            return 0
        return torch.matmul(self._exog, self._params["coef"])

    @property
    def marginal_mean(self):
        """
        Marginal mean of the latent variables, not knowing the endog variables.
        """
        if self._exog is None:
            return 0
        return self._marginal_mean.cpu()

    @property
    def covariance(self) -> torch.Tensor:
        """Covariance matrix."""
        return self._params["covariance"].cpu()

    @property
    def coef(self) -> torch.Tensor:
        """Coefficient matrix."""
        coef = self._params.get("coef")
        if coef is None:
            return None
        return coef.cpu()


class PlnSampler(_BasePlnSampler):
    """
    Sampler for Poisson Log-Normal model.
    The parameters of the model are generated randomly but have a specific structure.

    Examples
    --------
    >>> from pyPLNmodels import PlnSampler, Pln
    >>> sampler = PlnSampler()
    >>> endog = sampler.sample()
    >>> pln = Pln(endog, exog = sampler.exog, add_const = False)
    >>> pln.fit()
    >>> estimated_cov = pln.covariance
    >>> true_covariance = sampler.covariance
    >>> estimated_latent_variables = pln.latent_variables
    >>> true_latent_variables = sampler.latent_variables

    See also
    --------
    :class:`pyPLNmodels.Pln`
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
        seed: int = 0,
    ):  # pylint: disable=too-many-arguments
        """
        Initialize the sampling class.

        Parameters
        ----------
        n_samples : int, optional
            Number of samples to generate (default is 100).
        dim : int, optional
            Dimensionality of the data (default is 20).
        nb_cov : int, optional
            Number of covariates (default is 1).
        add_const : bool, optional
            Whether to add a constant term to the covariates (default is True).
        add_offsets : bool, optional
            Whether to add offsets to the data (default is False).
        marginal_mean_mean : int, optional
            Mean of the marginal means (default is 2).
        seed : int, optional
            Random seed for reproducibility (default is 0).
        Examples
        --------
        >>> from pyPLNmodels import PlnSampler
        >>> sampler = PlnSampler()
        >>> endog = sampler.sample()
        """
        exog = self._get_exog(
            n_samples=n_samples, nb_cov=nb_cov, will_add_const=add_const, seed=seed
        )
        offsets = _get_offsets(
            n_samples=n_samples, dim=dim, add_offsets=add_offsets, seed=seed
        )
        coef = self._get_coef(
            nb_cov=nb_cov,
            dim=dim,
            mean=marginal_mean_mean,
            add_const=add_const,
            seed=seed,
        )
        covariance = self._get_covariance(dim, seed=seed)
        super().__init__(
            n_samples=n_samples,
            exog=exog,
            add_const=add_const,
            offsets=offsets,
            coef=coef,
            covariance=covariance,
        )

    def _get_coef(
        self, *, nb_cov, dim, mean, add_const, seed
    ):  # pylint: disable = too-many-arguments
        return _get_coef(
            nb_cov=nb_cov, dim=dim, mean=mean, add_const=add_const, seed=seed
        )

    def _get_exog(self, *, n_samples, nb_cov, will_add_const, seed):
        return _get_exog(
            n_samples=n_samples, nb_cov=nb_cov, will_add_const=will_add_const, seed=seed
        )

    def _get_covariance(self, dim, seed):
        return _get_full_covariance(dim, seed=seed)

    @property
    def _dim_latent(self):
        return self.dim

    @_add_doc(
        _BaseSampler,
        example="""
        >>> from pyPLNmodels import PlnSampler
        >>> sampler = PlnSampler()
        >>> endog = sampler.sample()
        """,
    )
    def sample(self, seed: int = 0) -> torch.Tensor:
        return super().sample(seed=seed)
