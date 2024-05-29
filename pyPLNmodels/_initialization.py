import torch
import math
from typing import Optional
from pyPLNmodels._utils import _log_stirling
import time
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np


if torch.cuda.is_available():
    DEVICE = "cuda:0"
else:
    DEVICE = "cpu"


def _init_covariance(endog: torch.Tensor, exog: torch.Tensor) -> torch.Tensor:
    """
    Initialization for the covariance for the Pln model. Take the log of endog
    (careful when endog=0), and computes the Maximum Likelihood
    Estimator in the gaussian case.

    Parameters
    ----------
    endog : torch.Tensor
        Samples with size (n,p)
    offsets : torch.Tensor
        Offset, size (n,p)
    exog : torch.Tensor
        Covariates, size (n,d)
    coef : torch.Tensor
        Coefficient of size (d,p)

    Returns
    -------
    torch.Tensor
        Covariance matrix of size (p,p)
    """
    log_y = torch.log(endog + (endog == 0) * math.exp(-2))
    log_y_centered = log_y - torch.mean(log_y, axis=0)
    n_samples = endog.shape[0]
    sigma_hat = 1 / (n_samples - 1) * (log_y_centered.T) @ log_y_centered
    return sigma_hat


def _init_components(endog: torch.Tensor, rank: int) -> torch.Tensor:
    """
    Initialization for components for the Pln model. Get a first guess for covariance
    that is easier to estimate and then takes the rank largest eigenvectors to get components.

    Parameters
    ----------
    endog : torch.Tensor
        Samples with size (n,p)
    rank : int
        The dimension of the latent space, i.e. the reduced dimension.

    Returns
    -------
    torch.Tensor
        Initialization of components of size (p,rank)
    """
    t = time.time()
    log_y = torch.log(endog + (endog == 0) * math.exp(-2))
    max_dim = min(rank, endog.shape[0])
    pca = PCA(n_components=max_dim)
    pca.fit(log_y.cpu().detach())
    pca_comp = pca.components_.T * np.sqrt(pca.explained_variance_)
    if rank > max_dim:
        nb_missing = rank - max_dim
        adding = np.random.randn(endog.shape[1], nb_missing) / rank
        pca_comp = np.concatenate((pca_comp, adding), axis=1)
    return torch.from_numpy(pca_comp)


def _init_latent_mean(
    endog: torch.Tensor,
    exog: torch.Tensor,
    offsets: torch.Tensor,
    coef: torch.Tensor,
    components: torch.Tensor,
    n_iter_max=40,
    lr=0.01,
    eps=7e-3,
) -> torch.Tensor:
    """
    Initialization for the variational parameter latent_mean.
    Basically, the mode of the log_posterior is computed.

    Parameters
    ----------
    endog : torch.Tensor
        Samples with size (n,p)
    offsets : torch.Tensor
        Offset, size (n,p)
    exog : torch.Tensor
        Covariates, size (n,d)
    coef : torch.Tensor
        Coefficient of size (d,p)
    components : torch.Tensor
        Components of size (p,rank)
    n_iter_max : int, optional
        The maximum number of iterations in the gradient ascent. Default is 40.
    lr : float, optional
        The learning rate of the optimizer. Default is 0.01.
    eps : float, optional
        The tolerance. The algorithm will stop as soon as the criterion is lower than the tolerance.
        Default is 7e-1.

    Returns
    -------
    torch.Tensor
        The initialized latent mean with size (n,rank)
    """
    device = endog.device
    components = components.to(device)
    if coef is not None:
        coef = coef.to(device)
    mode = torch.randn(endog.shape[0], components.shape[1], device=device)
    mode.requires_grad_(True)
    optimizer = torch.optim.Rprop([mode], lr=lr)
    crit = 2 * eps
    old_mode = torch.clone(mode)
    keep_condition = True
    i = 0
    while i < n_iter_max and keep_condition:
        batch_loss = log_posterior(endog, exog, offsets, mode, components, coef)
        loss = -torch.mean(batch_loss)
        loss.backward()
        optimizer.step()
        crit = torch.max(torch.abs(mode - old_mode))
        optimizer.zero_grad()
        if crit < eps and i > 2:
            keep_condition = False
        old_mode = torch.clone(mode)
        i += 1
    return mode


def _components_from_covariance(covariance: torch.Tensor, rank: int) -> torch.Tensor:
    """
    Get the PCA with rank components of covariance.

    Parameters
    ----------
    covariance : torch.Tensor
        Covariance matrix of size (p, p)
    rank : int
        The number of columns wanted for components

    Returns
    -------
    torch.Tensor
        Requested components of size (p, rank) containing the rank eigenvectors
        with largest eigenvalues.
    """
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    requested_components = eigenvectors[:, -rank:] @ torch.diag(
        torch.sqrt(eigenvalues[-rank:])
    )
    return requested_components


def _init_coef(
    endog: torch.Tensor, exog: torch.Tensor, offsets: torch.Tensor
) -> torch.Tensor:
    """
    Initialize the coefficient for the Pln model using Poisson regression model.

    Parameters
    ----------
    endog : torch.Tensor
        Samples with size (n, p)
    exog : torch.Tensor
        Covariates, size (n, d)
    offsets : torch.Tensor
        Offset, size (n, p)

    Returns
    -------
    torch.Tensor or None
        Coefficient of size (d, p) or None if exog is None.
    """
    if exog is None:
        return None

    poiss_reg = _PoissonReg()
    poiss_reg.fit(endog, exog, offsets)
    return poiss_reg.beta.to(DEVICE)


def _init_coef_coef_inflation(
    endog: torch.Tensor,
    exog: torch.Tensor,
    exog_inflation: torch.Tensor,
    offsets: torch.Tensor,
    zero_inflation_formula: str,
) -> torch.Tensor:
    """
    Initialize the coefficient for the ZIPln model using
    Zero Inflated Poisson regression model.

    Parameters
    ----------
    endog : torch.Tensor
        Samples with size (n, p)
    exog : torch.Tensor
        Covariates, size (n, d)
    exog_infla : torch.Tensor
        Covariates for the inflation, size (n, d)
    offsets : torch.Tensor
        Offset, size (n, p)
    zero_inflation_formula: str {"column-wise", "row-wise", "global"}
        The modelling of the zero_inflation. Either "column-wise", "row-wise"
        or "global".
    Returns
    -------
    tuple (torch.Tensor or None, torch.Tensor) or None
        Coefficient of size (d, p) or None if exog is None.
        torch.Tensor of size (d,p)
    """
    if exog is None:
        if zero_inflation_formula == "global":
            coef_infla = torch.tensor([0.0])
        elif zero_inflation_formula == "row-wise":
            coef_infla = torch.randn(endog.shape[0], exog_inflation.shape[0])
        else:
            coef_infla = torch.randn(exog_inflation.shape[1], endog.shape[1])
        return None, coef_infla.to(DEVICE), None
    zip = ZIP(endog, exog, exog_inflation, offsets, zero_inflation_formula)
    zip.fit()

    return (
        zip.coef.detach().to(DEVICE),
        zip.coef_inflation.detach().to(DEVICE),
        zip.rec_error,
    )


def log_posterior(
    endog: torch.Tensor,
    exog: torch.Tensor,
    offsets: torch.Tensor,
    posterior_mean: torch.Tensor,
    components: torch.Tensor,
    coef: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the log posterior of the Poisson Log-Normal (Pln) model.

    Parameters
    ----------
    endog : torch.Tensor
        Samples with size (batch_size, p)
    exog : torch.Tensor or None
        Covariates, size (batch_size, d) or (d)
    offsets : torch.Tensor
        Offset, size (batch_size, p)
    posterior_mean : torch.Tensor
        Posterior mean with size (N_samples, N_batch, rank) or (batch_size, rank)
    components : torch.Tensor
        Components with size (p, rank)
    coef : torch.Tensor
        Coefficient with size (d, p)

    Returns
    -------
    torch.Tensor
        Log posterior of size n_samples.
    """
    rank = posterior_mean.shape[-1]
    components_posterior_mean = torch.matmul(
        components.unsqueeze(0), posterior_mean.unsqueeze(2)
    ).squeeze()

    if exog is None:
        XB = 0
    else:
        XB = torch.matmul(exog, coef)

    log_lambda = offsets + components_posterior_mean + XB
    first_term = (
        -rank / 2 * math.log(2 * math.pi)
        - 1 / 2 * torch.norm(posterior_mean, dim=-1) ** 2
    )
    second_term = torch.sum(
        -torch.exp(log_lambda) + log_lambda * endog - _log_stirling(endog), axis=-1
    )
    return first_term + second_term


class ZIP:
    def __init__(self, endog, exog, exog_inflation, offsets, zero_inflation_formula):
        self.endog = endog
        self.exog = exog
        self.exog_inflation = exog_inflation
        self.offsets = offsets
        self.zero_inflation_formula = zero_inflation_formula

        self.r0 = torch.mean((endog == 0).double(), axis=0)
        self.ybarre = torch.mean(endog, axis=0)

        self.dim = self.endog.shape[1]
        self.d = exog.shape[1]
        self.n_samples = endog.shape[0]

        if self.zero_inflation_formula == "column-wise":
            self.d0 = exog_inflation.shape[1]
            self.coef_inflation = torch.randn(self.d0, self.dim).requires_grad_(True)
        elif self.zero_inflation_formula == "row-wise":
            self.d0 = exog_inflation.shape[0]
            self.coef_inflation = torch.randn(self.n_samples, self.d0).requires_grad_(
                True
            )
        else:
            self.d0 = None
            self.coef_inflation = torch.Tensor([0.0]).requires_grad_(True)
        self.coef = torch.randn(self.d, self.dim).requires_grad_(True)

    @property
    def mean_poisson(self):
        return torch.exp(self.offsets + self.exog @ self.coef)

    @property
    def mean_inflation(self):
        if self.zero_inflation_formula == "column-wise":
            mean = self.exog_inflation @ self.coef_inflation
        elif self.zero_inflation_formula == "row-wise":
            mean = self.coef_inflation @ self.exog_inflation
        else:
            mean = self.coef_inflation
        return torch.sigmoid(mean)

    def loglike(self, lam, pi):
        first_term = (
            self.n_samples * self.r0 * torch.log(pi + (1 - pi) * torch.exp(-lam))
        )
        second_term = self.n_samples * (1 - self.r0) * (
            torch.log(1 - pi) - lam
        ) + self.n_samples * self.ybarre * torch.log(lam)
        return first_term + second_term

    def fit(self, nb_iter=150):
        optim = torch.optim.Rprop([self.coef, self.coef_inflation])
        for i in range(nb_iter):
            mean_poisson = self.mean_poisson
            mean_inflation = self.mean_inflation
            loss = -torch.mean(self.loglike(self.mean_poisson, self.mean_inflation))
            loss.backward()
            optim.step()
            optim.zero_grad()
        pred = (1 - self.mean_inflation) * self.mean_poisson
        self.rec_error = torch.sqrt(torch.mean((pred - self.endog) ** 2)).item()


class _PoissonReg:
    """
    Poisson regression model.

    Attributes
    ----------
    beta : torch.Tensor
        The learned regression coefficients.

    Methods
    -------
    fit(Y, exog, O, Niter_max=70, tol=0.001, lr=0.005, verbose=False)
        Fit the Poisson regression model to the given data.

    """

    def __init__(self) -> None:
        self.beta: Optional[torch.Tensor] = None

    def fit(
        self,
        Y: torch.Tensor,
        exog: torch.Tensor,
        offsets: torch.Tensor,
        Niter_max: int = 70,
        tol: float = 0.001,
        lr: float = 0.005,
        verbose: bool = False,
    ) -> None:
        """
        Fit the Poisson regression model to the given data.

        Parameters
        ----------
        Y : torch.Tensor
            The dependent variable of shape (n_samples, n_features).
        exog : torch.Tensor
            The exog of shape (n_samples, n_exog).
        offsets : torch.Tensor
            The offset term of shape (n_samples, n_features).
        Niter_max : int, optional
            The maximum number of iterations (default is 70).
        tol : float, optional
            The tolerance for convergence (default is 0.001).
        lr : float, optional
            The learning rate (default is 0.005).
        verbose : bool, optional
            Whether to print intermediate information during fitting (default is False).

        """
        self.device = Y.device
        beta = torch.rand(
            (exog.shape[1], Y.shape[1]), requires_grad=True, device=self.device
        )
        optimizer = torch.optim.Rprop([beta], lr=lr)
        i = 0
        grad_norm = 2 * tol  # Criterion
        while i < Niter_max and grad_norm > tol:
            loss = -compute_poissreg_log_like(Y, offsets, exog, beta)
            loss.backward()
            optimizer.step()
            grad_norm = torch.norm(beta.grad)
            beta.grad.zero_()
            i += 1
            if verbose:
                if i % 10 == 0:
                    print("log like : ", -loss)
                    print("grad_norm : ", grad_norm)
                if i < Niter_max:
                    print(f"Tolerance reached in {i} iterations")
                else:
                    print("Maximum number of iterations reached")
        self.beta = beta


def compute_poissreg_log_like(
    Y: torch.Tensor, O: torch.Tensor, exog: torch.Tensor, beta: torch.Tensor
) -> torch.Tensor:
    """
    Compute the log likelihood of a Poisson regression model.

    Parameters
    ----------
    Y : torch.Tensor
        The dependent variable of shape (n_samples, n_features).
    O : torch.Tensor
        The offset term of shape (n_samples, n_features).
    exog : torch.Tensor
        The exog of shape (n_samples, n_exog).
    beta : torch.Tensor
        The regression coefficients of shape (n_exog, n_features).

    Returns
    -------
    torch.Tensor
        The log likelihood of the Poisson regression model.

    """
    XB = torch.matmul(exog.unsqueeze(1), beta.unsqueeze(0)).squeeze()
    return torch.sum(-torch.exp(O + XB) + torch.multiply(Y, O + XB))
