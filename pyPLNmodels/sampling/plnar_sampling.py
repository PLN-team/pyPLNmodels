import torch

from pyPLNmodels.utils._utils import _add_doc

from ._base_sampler import _BaseSampler
from .pln_sampling import PlnSampler
from ._utils import _get_diag_covariance, _get_full_covariance, _get_covariance_ortho


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class PlnARSampler(PlnSampler):
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
    >>> print('Autoreg matrix:', pln.ar_coef)

    See also
    --------
    :class:`pyPLNmodels.PlnAR`
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
        ar_type: {"spherical", "diagonal", "full"} = "full",
    ):  # pylint: disable=too-many-arguments
        self.ar_type = ar_type
        dumb_covariance = _get_full_covariance(dim, seed)
        self.ortho_components, _ = torch.linalg.qr(dumb_covariance)
        super().__init__(
            n_samples=n_samples,
            dim=dim,
            nb_cov=nb_cov,
            add_const=add_const,
            add_offsets=add_offsets,
            marginal_mean_mean=marginal_mean_mean,
            seed=seed,
        )
        if self.ar_type == "diagonal":
            ar_coef = torch.ones(dim) * 0.8
        elif self.ar_type == "spherical":
            ar_coef = torch.Tensor([0.5])
        else:
            diag_ar_coef = torch.abs(torch.randn(self.dim)).to(DEVICE)
            diag_ar_coef = diag_ar_coef / (torch.max(diag_ar_coef) + 0.1)
            ar_coef = self.ortho_components * diag_ar_coef @ (self.ortho_components.T)
        self._params["ar_coef"] = ar_coef.to(DEVICE)

    @property
    def ar_coef(self):
        """
        Autoregressive coefficient of the model.
        If `ar_type` is "diagonal", this is a vector of size `dim`.
        If `ar_type` is "spherical", this is a scalar.
        """
        return self._params["ar_coef"].cpu()

    def _get_covariance(self, dim, seed):
        if self.ar_type == "diagonal":
            return _get_diag_covariance(dim, seed) + 0.5
        if self.ar_type == "spherical":
            return _get_full_covariance(dim, seed)
        return _get_covariance_ortho(self.ortho_components, seed)

    def _get_gaussians(self, seed):
        if self.ar_type == "diagonal":
            return self._get_gaussians_diag_ar(seed)
        if self.ar_type == "spherical":
            return self._get_gaussians_spherical_ar(seed)
        return self._get_gaussians_full_ar(seed)

    def _get_gaussians_spherical_ar(self, seed):
        torch.manual_seed(seed)
        centered_gaussian = torch.randn(self.n_samples, self._dim_latent).to(DEVICE)
        ar_coef = self._params["ar_coef"]
        components = self._get_components()
        gaussians = torch.zeros(self.n_samples, self.dim).to(DEVICE)
        gaussians[0] = centered_gaussian[0] @ components + self._get_mean_i(0)
        covariance = self._params["covariance"]
        autoregressive_covariance = covariance - ar_coef**2 * covariance
        autoregressive_components = torch.linalg.cholesky(autoregressive_covariance)
        for i in range(1, self.n_samples):
            mean_noise = self._get_mean_i(i) - ar_coef * self._get_mean_i(i - 1)
            noise = centered_gaussian[i] @ autoregressive_components + mean_noise
            gaussians[i] = ar_coef * gaussians[i - 1] + noise
        return gaussians

    def _get_mean_i(self, i):
        if self._exog is None:
            return 0
        return self._marginal_mean[i]

    def _get_components(self):
        if self.ar_type == "diagonal":
            return torch.sqrt(self._params["covariance"])
        return torch.linalg.cholesky(self._params["covariance"])

    def _get_gaussians_diag_ar(self, seed):
        torch.manual_seed(seed)
        centered_gaussian = torch.randn(self.n_samples, self._dim_latent).to(DEVICE)
        ar_coef = self._params["ar_coef"]
        components = self._get_components()
        gaussians = torch.zeros(self.n_samples, self.dim).to(DEVICE)
        gaussians[0] = centered_gaussian[0] * components + self._get_mean_i(0)
        covariance = self._params["covariance"]
        autoregressive_covariance = covariance - ar_coef**2 * covariance
        autoregressive_components = torch.sqrt(autoregressive_covariance)
        for i in range(1, self.n_samples):
            mean_noise = self._get_mean_i(i) - ar_coef * self._get_mean_i(i - 1)
            noise = centered_gaussian[i] * autoregressive_components + mean_noise
            gaussians[i] = ar_coef * gaussians[i - 1] + noise
        return gaussians

    def _get_gaussians_full_ar(self, seed):
        torch.manual_seed(seed)
        centered_gaussian = torch.randn(self.n_samples, self._dim_latent).to(DEVICE)
        ar_coef = self._params["ar_coef"]
        components = self._get_components()
        gaussians = torch.zeros(self.n_samples, self.dim).to(DEVICE)
        gaussians[0] = centered_gaussian[0] @ components + self._get_mean_i(0)
        covariance = self._params["covariance"]
        autoregressive_covariance = covariance - ar_coef @ covariance @ (ar_coef.T)
        autoregressive_components = torch.linalg.cholesky(autoregressive_covariance)
        for i in range(1, self.n_samples):
            mean_backward = (
                0 if self._exog is None else ar_coef @ self._get_mean_i(i - 1)
            )
            mean_noise = self._get_mean_i(i) - mean_backward
            noise = centered_gaussian[i] @ autoregressive_components + mean_noise
            gaussians[i] = gaussians[i - 1] @ ar_coef + noise
        return gaussians

    @_add_doc(
        _BaseSampler,
        example="""
        >>> from pyPLNmodels import PlnARSampler
        >>> sampler = PlnARSampler()
        >>> endog = sampler.sample()
        >>> ar_coef = sampler.ar_coef
        >>> print(ar_coef)
        """,
    )
    def sample(self, seed: int = 0) -> torch.Tensor:
        return super().sample(seed=seed)
