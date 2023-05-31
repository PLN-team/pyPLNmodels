import torch
import math
from typing import Optional
from ._utils import _log_stirling

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


def _init_covariance(
    counts: torch.Tensor, covariates: torch.Tensor, coef: torch.Tensor
) -> torch.Tensor:
    """
    Initialization for the covariance for the Pln model. Take the log of counts
    (careful when counts=0), and computes the Maximum Likelihood
    Estimator in the gaussian case.

    Parameters
    ----------
    counts : torch.Tensor
        Samples with size (n,p)
    offsets : torch.Tensor
        Offset, size (n,p)
    covariates : torch.Tensor
        Covariates, size (n,d)
    coef : torch.Tensor
        Coefficient of size (d,p)

    Returns
    -------
    torch.Tensor
        Covariance matrix of size (p,p)
    """
    log_y = torch.log(counts + (counts == 0) * math.exp(-2))
    log_y_centered = log_y - torch.mean(log_y, axis=0)
    n_samples = counts.shape[0]
    sigma_hat = 1 / (n_samples - 1) * (log_y_centered.T) @ log_y_centered
    return sigma_hat


def _init_components(
    counts: torch.Tensor, covariates: torch.Tensor, coef: torch.Tensor, rank: int
) -> torch.Tensor:
    """
    Initialization for components for the Pln model. Get a first guess for covariance
    that is easier to estimate and then takes the rank largest eigenvectors to get components.

    Parameters
    ----------
    counts : torch.Tensor
        Samples with size (n,p)
    offsets : torch.Tensor
        Offset, size (n,p)
    covariates : torch.Tensor
        Covariates, size (n,d)
    coef : torch.Tensor
        Coefficient of size (d,p)
    rank : int
        The dimension of the latent space, i.e. the reduced dimension.

    Returns
    -------
    torch.Tensor
        Initialization of components of size (p,rank)
    """
    sigma_hat = _init_covariance(counts, covariates, coef).detach()
    components = _components_from_covariance(sigma_hat, rank)
    return components


def _init_latent_mean(
    counts: torch.Tensor,
    covariates: torch.Tensor,
    offsets: torch.Tensor,
    coef: torch.Tensor,
    components: torch.Tensor,
    n_iter_max=500,
    lr=0.01,
    eps=7e-3,
) -> torch.Tensor:
    """
    Initialization for the variational parameter latent_mean.
    Basically, the mode of the log_posterior is computed.

    Parameters
    ----------
    counts : torch.Tensor
        Samples with size (n,p)
    offsets : torch.Tensor
        Offset, size (n,p)
    covariates : torch.Tensor
        Covariates, size (n,d)
    coef : torch.Tensor
        Coefficient of size (d,p)
    components : torch.Tensor
        Components of size (p,rank)
    n_iter_max : int, optional
        The maximum number of iterations in the gradient ascent. Default is 500.
    lr : float, optional
        The learning rate of the optimizer. Default is 0.01.
    eps : float, optional
        The tolerance. The algorithm will stop as soon as the criterion is lower than the tolerance.
        Default is 7e-3.

    Returns
    -------
    torch.Tensor
        The initialized latent mean with size (n,rank)
    """
    mode = torch.randn(counts.shape[0], components.shape[1], device=DEVICE)
    mode.requires_grad_(True)
    optimizer = torch.optim.Rprop([mode], lr=lr)
    crit = 2 * eps
    old_mode = torch.clone(mode)
    keep_condition = True
    i = 0
    while i < n_iter_max and keep_condition:
        batch_loss = log_posterior(counts, covariates, offsets, mode, components, coef)
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
    counts: torch.Tensor, covariates: torch.Tensor, offsets: torch.Tensor
) -> torch.Tensor:
    """
    Initialize the coefficient for the Pln model using Poisson regression model.

    Parameters
    ----------
    counts : torch.Tensor
        Samples with size (n, p)
    covariates : torch.Tensor
        Covariates, size (n, d)
    offsets : torch.Tensor
        Offset, size (n, p)

    Returns
    -------
    torch.Tensor or None
        Coefficient of size (d, p) or None if covariates is None.
    """
    if covariates is None:
        return None

    poiss_reg = _PoissonReg()
    poiss_reg.fit(counts, covariates, offsets)
    return poiss_reg.beta


def log_posterior(
    counts: torch.Tensor,
    covariates: torch.Tensor,
    offsets: torch.Tensor,
    posterior_mean: torch.Tensor,
    components: torch.Tensor,
    coef: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the log posterior of the Poisson Log-Normal (Pln) model.

    Parameters
    ----------
    counts : torch.Tensor
        Samples with size (batch_size, p)
    covariates : torch.Tensor or None
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

    if covariates is None:
        XB = 0
    else:
        XB = torch.matmul(covariates, coef)

    log_lambda = offsets + components_posterior_mean + XB
    first_term = (
        -rank / 2 * math.log(2 * math.pi)
        - 1 / 2 * torch.norm(posterior_mean, dim=-1) ** 2
    )
    second_term = torch.sum(
        -torch.exp(log_lambda) + log_lambda * counts - _log_stirling(counts), axis=-1
    )
    return first_term + second_term


class _PoissonReg:
    """
    Poisson regression model.

    Attributes
    ----------
    beta : torch.Tensor
        The learned regression coefficients.

    Methods
    -------
    fit(Y, covariates, O, Niter_max=300, tol=0.001, lr=0.005, verbose=False)
        Fit the Poisson regression model to the given data.

    """

    def __init__(self) -> None:
        self.beta: Optional[torch.Tensor] = None

    def fit(
        self,
        Y: torch.Tensor,
        covariates: torch.Tensor,
        offsets: torch.Tensor,
        Niter_max: int = 300,
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
        covariates : torch.Tensor
            The covariates of shape (n_samples, n_covariates).
        offsets : torch.Tensor
            The offset term of shape (n_samples, n_features).
        Niter_max : int, optional
            The maximum number of iterations (default is 300).
        tol : float, optional
            The tolerance for convergence (default is 0.001).
        lr : float, optional
            The learning rate (default is 0.005).
        verbose : bool, optional
            Whether to print intermediate information during fitting (default is False).

        """
        beta = torch.rand(
            (covariates.shape[1], Y.shape[1]), device=DEVICE, requires_grad=True
        )
        optimizer = torch.optim.Rprop([beta], lr=lr)
        i = 0
        grad_norm = 2 * tol  # Criterion
        while i < Niter_max and grad_norm > tol:
            loss = -compute_poissreg_log_like(Y, offsets, covariates, beta)
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
    Y: torch.Tensor, O: torch.Tensor, covariates: torch.Tensor, beta: torch.Tensor
) -> torch.Tensor:
    """
    Compute the log likelihood of a Poisson regression model.

    Parameters
    ----------
    Y : torch.Tensor
        The dependent variable of shape (n_samples, n_features).
    O : torch.Tensor
        The offset term of shape (n_samples, n_features).
    covariates : torch.Tensor
        The covariates of shape (n_samples, n_covariates).
    beta : torch.Tensor
        The regression coefficients of shape (n_covariates, n_features).

    Returns
    -------
    torch.Tensor
        The log likelihood of the Poisson regression model.

    """
    XB = torch.matmul(covariates.unsqueeze(1), beta.unsqueeze(0)).squeeze()
    return torch.sum(-torch.exp(O + XB) + torch.multiply(Y, O + XB))
