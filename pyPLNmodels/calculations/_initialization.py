import math
from typing import Optional
import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from pyPLNmodels.utils._utils import _log_stirling

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def compute_poissreg_log_like(
    endog: torch.Tensor, offsets: torch.Tensor, exog: torch.Tensor, beta: torch.Tensor
) -> torch.Tensor:
    """
    Compute the log likelihood of a Poisson regression model.

    Parameters
    ----------
    endog : torch.Tensor
        The dependent variable of shape (n_samples, dim).
    offsets : torch.Tensor
        The offsets term of shape (n_samples, dim).
    exog : torch.Tensor
        The exog of shape (n_samples, n_exog).
    beta : torch.Tensor
        The regression coefficients of shape (n_exog, dim).

    Returns
    -------
    torch.Tensor
        The log likelihood of the Poisson regression model.
    """
    marginal_mean = torch.matmul(exog.unsqueeze(1), beta.unsqueeze(0)).squeeze()
    return torch.sum(
        -torch.exp(offsets + marginal_mean) + endog * (offsets + marginal_mean)
    )


def _init_coef(  # pylint: disable=too-many-arguments
    *,
    endog: torch.Tensor,
    exog: torch.Tensor,
    offsets: torch.Tensor,
    itermax: int = 70,
    tol: float = 0.001,
    lr: float = 0.005,
    verbose: bool = False,
) -> Optional[torch.Tensor]:
    """
    Initialize the coefficient for the `Pln` model using the Poisson regression model.

    Parameters
    ----------
    endog : torch.Tensor
        Samples with size (n, p)
    exog : torch.Tensor
        Covariates, size (n, d)
    offsets : torch.Tensor
        Offset, size (n, p)
    itermax : int, keyword-only, optional
        The maximum number of iterations (default is 70).
    tol : float, keyword-only, optional
        The tolerance for convergence (default is 0.001).
    lr : float, keyword-only, optional
        The learning rate (default is 0.005).
    verbose : bool, keyword-only, optional
        Whether to print intermediate information during fitting (default is False).

    Returns
    -------
    torch.Tensor or None
        Coefficient of size (d, p) or None if `exog` is None.
    """
    if exog is None:
        return None

    beta = torch.rand(
        (exog.shape[1], endog.shape[1]), requires_grad=True, device=DEVICE
    )
    optimizer = torch.optim.Rprop([beta], lr=lr)
    i = 0
    grad_norm = 2 * tol  # Criterion

    while i < itermax and grad_norm > tol:
        loss = -compute_poissreg_log_like(endog, offsets, exog, beta)
        loss.backward()
        optimizer.step()
        grad_norm = torch.norm(beta.grad)
        beta.grad.zero_()
        i += 1
        if verbose is True:
            if i % 10 == 0:
                print("log like : ", -loss)
                print("grad_norm : ", grad_norm)
    if verbose is True:
        if i < itermax:
            print(f"Tolerance reached in {i} iterations.")
        else:
            print("Maximum number of iterations reached.")

    return beta


def _log_transform(endog: torch.Tensor) -> torch.Tensor:
    return torch.log(endog + (endog == 0) * math.exp(-2))


def _perform_pca_logy(log_y: torch.Tensor, max_dim: int) -> np.ndarray:
    pca = PCA(n_components=max_dim)
    pca.fit(log_y.cpu().detach())
    return pca.components_.T * np.sqrt(pca.explained_variance_)


def _add_random_components(
    *, pca_components: np.ndarray, endog: torch.Tensor, rank: int, max_dim: int
) -> np.ndarray:
    if rank > max_dim:
        nb_missing = rank - max_dim
        random_components = np.random.randn(endog.shape[1], nb_missing) / rank
        pca_components = np.concatenate((pca_components, random_components), axis=1)
    return pca_components


def _init_components(endog: torch.Tensor, rank: int) -> torch.Tensor:
    """
    Initialize components for the `PlnPCA` model.

    Parameters
    ----------
    endog : torch.Tensor
        Samples with size (n, p)
    rank : int
        The dimension of the latent space, i.e., the reduced dimension.

    Returns
    -------
    torch.Tensor
        Initialization of components of size (p, rank)
    """
    log_y = _log_transform(endog)
    max_dim = min(rank, endog.shape[0])
    pca_components = _perform_pca_logy(log_y, max_dim)
    pca_components = _add_random_components(
        pca_components=pca_components, endog=endog, rank=rank, max_dim=max_dim
    )
    return torch.from_numpy(pca_components).float()


def _init_components_prec(endog):
    nan_mask = torch.isnan(endog)
    mean_value = torch.nanmean(endog, dim=0)
    endog[nan_mask] = mean_value[nan_mask.nonzero(as_tuple=True)[1]]

    log_y = _log_transform(endog)
    log_y = log_y - log_y.mean(dim=0)
    covariance = torch.mm(log_y.T, log_y) / (log_y.shape[0] - 1)
    precision = torch.linalg.pinv(covariance)
    eigenvalues, eigenvectors = torch.linalg.eigh(precision)
    eigenvalues = torch.maximum(eigenvalues, torch.Tensor([0]).to(endog.device))
    return eigenvectors @ torch.diag(torch.sqrt(eigenvalues))


def compute_log_posterior(  # pylint: disable=too-many-arguments
    *,
    endog: torch.Tensor,
    exog: torch.Tensor,
    offsets: torch.Tensor,
    latent_mean: torch.Tensor,
    components: torch.Tensor,
    coef: torch.Tensor,
    log_stirling_log_endog: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the log posterior of the Poisson Log-Normal (`Pln`) model.

    Parameters
    ----------
    endog : torch.Tensor
        Samples with size (n_samples, dim)
    exog : torch.Tensor or None
        Covariates, size (n_samples, nb_cov)
    offsets : torch.Tensor
        Offset, size (n_samples, dim)
    latent_mean : torch.Tensor
        Posterior mean with size (n_samples, rank)
    components : torch.Tensor
        Components with size (dim, rank)
    coef : torch.Tensor
        Coefficient with size (nb_cov, dim)
    log_stirling_log_endog : torch.Tensor, keyword-only
        Precomputed log Stirling approximation

    Returns
    -------
    torch.Tensor
        Log posterior of size n_samples.
    """
    rank = latent_mean.shape[-1]
    components_latent_mean = torch.matmul(
        components.unsqueeze(0), latent_mean.unsqueeze(2)
    ).squeeze()

    if exog is None:
        marginal_mean = 0
    else:
        marginal_mean = torch.matmul(exog, coef)

    log_lambda = offsets + components_latent_mean + marginal_mean
    prior_term = (
        -rank / 2 * math.log(2 * math.pi) - 1 / 2 * torch.norm(latent_mean, dim=-1) ** 2
    )
    likelihood_term = torch.sum(
        -torch.exp(log_lambda) + log_lambda * endog - log_stirling_log_endog, axis=-1
    )
    return prior_term + likelihood_term


def _init_latent_mean_pca(  # pylint: disable=too-many-arguments
    *,
    endog: torch.Tensor,
    exog: torch.Tensor,
    offsets: torch.Tensor,
    coef: torch.Tensor,
    components: torch.Tensor,
    max_iterations: int = 40,
    learning_rate: float = 0.01,
    tolerance: float = 7e-3,
) -> torch.Tensor:
    """
    Initialize the latent mean for the variational parameter.

    Parameters
    ----------
    endog : torch.Tensor
        Samples with size (n_samples, dim)
    exog : torch.Tensor
        Covariates, size (n_samples, nb_cov)
    offsets : torch.Tensor
        Offset, size (n_samples, dim)
    coef : torch.Tensor
        Coefficient of size (nb_cov, dim)
    components : torch.Tensor
        Components of size (dim, rank)
    max_iterations : int, keyword-only, optional
        The maximum number of iterations in the gradient ascent. Default is 40.
    learning_rate : float, keyword-only, optional
        The learning rate of the optimizer. Default is 0.01.
    tolerance : float, keyword-only, optional
        The tolerance. The algorithm will stop as soon as the criterion is lower than the tolerance.
        Default is 7e-3.

    Returns
    -------
    torch.Tensor
        The initialized latent mean with size (n_samples, rank)
    """
    log_stirling_log_endog = _log_stirling(endog)
    latent_mean = torch.randn(endog.shape[0], components.shape[1], device=DEVICE)
    latent_mean.requires_grad_(True)
    optimizer = torch.optim.Rprop([latent_mean], lr=learning_rate)
    criterion = 2 * tolerance
    old_latent_mean = torch.clone(latent_mean)
    iteration = 0

    while iteration < max_iterations and criterion > tolerance:
        loss = -torch.mean(
            compute_log_posterior(
                endog=endog,
                exog=exog,
                offsets=offsets,
                latent_mean=latent_mean,
                components=components,
                coef=coef,
                log_stirling_log_endog=log_stirling_log_endog,
            )
        )
        loss.backward()
        optimizer.step()
        criterion = torch.max(torch.abs(latent_mean - old_latent_mean))
        optimizer.zero_grad()
        old_latent_mean = torch.clone(latent_mean)
        iteration += 1

    return latent_mean.detach()


def _init_latent_sqrt_variance_pca(marginal_mean, offsets, components, mode):
    exp_term = torch.exp(offsets + marginal_mean + torch.matmul(mode, components.T))
    first = components.T.unsqueeze(0) * exp_term.unsqueeze(1)
    hessian = torch.matmul(first, components.unsqueeze(0)) + torch.eye(
        mode.shape[1]
    ).unsqueeze(0).to(DEVICE)
    cov_matrix = torch.inverse(hessian)
    return torch.sqrt(torch.diagonal(cov_matrix, dim1=1, dim2=2))


def _init_coef_coef_inflation(*, endog, exog, exog_inflation, offsets):
    zip_model = ZIP(
        endog=endog, exog=exog, exog_inflation=exog_inflation, offsets=offsets
    )
    zip_model.fit()
    coef = zip_model.coef.detach()
    coef = None if coef.shape[0] == 0 else coef
    return (coef, zip_model.coef_inflation.detach())


def _init_latent_pln(endog):
    n_samples, dim = endog.shape
    latent_mean = torch.log(endog + (endog == 0)).to(DEVICE)
    latent_sqrt_variance = 1 / 2 * torch.ones((n_samples, dim)).to(DEVICE)
    return latent_mean, latent_sqrt_variance


class ZIP:  # pylint: disable=too-many-instance-attributes
    """
    Simple Zero Inflated Poisson model for initialization of the `ZIPln` model.
    """

    def __init__(self, *, endog, exog, exog_inflation, offsets):
        """
        Simple initialization of the Zero Inflated Poisson model. Coefficients are
        initialized randomly.
        """
        self._log_endog = endog
        if exog is not None:
            self._exog = exog
        else:
            self._exog = None
        self._exog_inflation = exog_inflation
        self._offsets = offsets

        self._r0 = torch.mean((self._log_endog == 0).double(), axis=0)
        self._ybarre = torch.mean(self._log_endog, axis=0)

        self._n_samples = self._log_endog.shape[0]
        dim = self._log_endog.shape[1]
        nb_cov = exog.shape[1] if exog is not None else 0
        nb_cov_infla = exog_inflation.shape[1]

        self._coef_inflation = (
            torch.randn(nb_cov_infla, dim).to(DEVICE).requires_grad_(True)
        )
        self._coef = torch.randn(nb_cov, dim).to(DEVICE).requires_grad_(True)

    @property
    def _marginal_mean(self):
        if self._exog is None:
            return 0
        return self._exog @ self._coef

    @property
    def _mean_poisson(self):
        return torch.exp(self._offsets + self._marginal_mean)

    @property
    def _mean_inflation(self):
        mean = self._exog_inflation @ self._coef_inflation
        return torch.sigmoid(mean)

    def loglike(self, lam, pi):
        """
        Computes the log likelihood of a Zero Inflated Poisson regression model.
        """
        first_term = (
            self._n_samples * self._r0 * torch.log(pi + (1 - pi) * torch.exp(-lam))
        )
        second_term = self._n_samples * (1 - self._r0) * (
            torch.log(1 - pi) - lam
        ) + self._n_samples * self._ybarre * torch.log(lam)
        return first_term + second_term

    def fit(self, maxiter=150):  # pylint: disable=missing-function-docstring
        optim = torch.optim.Rprop([self._coef, self._coef_inflation])
        for _ in range(maxiter):
            loss = -torch.mean(self.loglike(self._mean_poisson, self._mean_inflation))
            loss.backward()
            optim.step()
            optim.zero_grad()

    @property
    def coef(self):
        """Coefficient of the mean for the ZI Poisson regression model."""
        return self._coef

    @property
    def coef_inflation(self):
        """Coefficient for the inflation part of the ZI Poisson regression model."""
        return self._coef_inflation


def _init_gmm(latent_positions, n_cluster, seed=0):
    gmm = GaussianMixture(
        n_components=n_cluster, covariance_type="diag", random_state=seed
    )
    gmm.fit(latent_positions)
    cluster_bias = torch.from_numpy(gmm.means_).to(DEVICE)
    sqrt_covariances = torch.sqrt(torch.from_numpy(gmm.covariances_)).to(DEVICE)
    weights = torch.from_numpy(gmm.weights_).to(DEVICE)
    latent_prob = torch.from_numpy(gmm.predict_proba(latent_positions)).to(DEVICE)
    return cluster_bias, sqrt_covariances, weights, latent_prob
