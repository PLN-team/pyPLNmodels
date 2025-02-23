import torch

from pyPLNmodels._utils import _add_doc

from ._base_sampler import _BaseSampler

from .plndiag_sampling import PlnDiagSampler


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class PlnARSampler(PlnDiagSampler):
    """
    PLN model sampler with simple autoregressive model on the latent variables. This basically
    assumes the latent variable of sample i depends on the latent variable on sample i-1.
    This assumes the dataset given in the initialization is ordered ! The covariance is assumed
    diagonal.
    See ?? for more details.

    Examples
    --------
    >>> from pyPLNmodels import PlnARSampler, PlnAR
    >>> sampler = PlnARSampler()
    >>> endog = sampler.sample()
    >>> pln = PlnAR(endog, exog = sampler.exog, add_const = False)
    >>> pln.fit()
    >>> estimated_cov = pln.covariance
    >>> true_covariance = sampler.covariance
    >>> estimated_latent_var = pln.latent_variables
    >>> true_latent_var = sampler.latent_variables
    >>> print('Autoreg matrix:', pln.autoreg_matrix)

    See also
    --------
    :class:`pyPLNmodels.PlnDiag`
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
        self.autoreg_matrix = torch.ones(dim) / 2
        super().__init__(
            n_samples=n_samples,
            dim=dim,
            nb_cov=nb_cov,
            add_const=add_const,
            add_offsets=add_offsets,
            marginal_mean_mean=marginal_mean_mean,
            seed=seed,
        )

    def _get_gaussians(self, seed):
        torch.manual_seed(seed)
        centered_gaussian = torch.randn(self.n_samples, self._dim_latent).to(DEVICE)
        mean = self._marginal_mean
        components = self._get_components()
        gaussians = torch.zeros(self.n_samples, self.dim).to(DEVICE)
        Z_i = centered_gaussian[0] * components + mean[0]
        gaussians[0] = Z_i
        covariance = self._params["covariance"]
        autoregressive_covariance = covariance - self.autoreg_matrix**2 * covariance
        autoregressive_components = torch.sqrt(autoregressive_covariance)
        for i in range(1, self.n_samples):
            mean_noise = mean[i] - self.autoreg_matrix * mean[i - 1]
            noise = centered_gaussian[i] * autoregressive_components + mean_noise
            gaussians[i] = self.autoreg_matrix * gaussians[i - 1] + noise
        return gaussians

    @_add_doc(
        _BaseSampler,
        example="""
        >>> from pyPLNmodels import PlnARSampler
        >>> sampler = PlnARSampler()
        >>> endog = sampler.sample()
        >>> autoreg_matrix = sampler.autoreg_matrix
        >>> print(autoreg_matrix)
        """,
    )
    def sample(self, seed: int = 0) -> torch.Tensor:
        return super().sample(seed=seed)
