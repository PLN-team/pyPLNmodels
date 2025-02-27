import torch
from pyPLNmodels.utils._utils import _phi


def _closed_formula_coef(exog: torch.Tensor, latent_mean: torch.Tensor) -> torch.Tensor:
    """
    Compute the closed-form coefficient for the M step of the Pln model.

    Parameters
    ----------
    exog : torch.Tensor
        Covariates with size (`n_samples`, `nb_cov`).
    latent_mean : torch.Tensor
        Variational parameter with size (`n_samples`, `dim`).

    Returns
    -------
    Optional[torch.Tensor]
        The closed-form coefficient with size (`nb_cov`, `dim`) or `None` if `exog` is `None`.
    """
    if exog is None:
        return None
    return torch.inverse(exog.T @ exog) @ exog.T @ latent_mean


def _closed_formula_covariance(
    marginal_mean: torch.Tensor,
    latent_mean: torch.Tensor,
    latent_sqrt_variance: torch.Tensor,
    n_samples: int,
) -> torch.Tensor:
    """
    Compute the closed-form covariance for the M step of the Pln model.

    Parameters
    ----------
    marginal_mean: torch.Tensor
        The marginal mean of the latent variables, given by `X @ B`, where
        `X` is the `exog` (or covariates) and `B` is the `coef` (or regression parameter).
    latent_mean : torch.Tensor
        Variational parameter with size (`n_samples`, `dim`).
    latent_sqrt_variance : torch.Tensor
        Variational parameter with size (`n_samples`, `dim`).
    n_samples : int
        Number of samples.

    Returns
    -------
    torch.Tensor
        The closed-form covariance with size (`dim`, `dim`).
    """
    residuals = latent_mean - marginal_mean
    closed = residuals.T @ residuals + torch.diag(
        torch.sum(torch.square(latent_sqrt_variance), dim=0)
    )
    return closed / n_samples


def _closed_formula_diag_covariance(
    marginal_mean: torch.Tensor,
    latent_mean: torch.Tensor,
    latent_sqrt_variance: torch.Tensor,
    n_samples: int,
):
    residuals = latent_mean - marginal_mean
    closed = torch.sum(residuals**2, dim=0) + torch.sum(
        torch.square(latent_sqrt_variance), dim=0
    )
    return closed / n_samples


def _closed_formula_latent_prob(
    marginal_mean, offsets, marginal_mean_infla, cov, dirac
):
    """
    Closed formula for the latent probability using the lambert function.
    """
    diag = torch.diag(cov)
    full_diag = diag.expand(dirac.shape[0], -1)
    return (
        torch.sigmoid(
            marginal_mean_infla - torch.log(_phi(marginal_mean + offsets, full_diag))
        )
        * dirac
    )
