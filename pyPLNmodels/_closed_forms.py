from typing import Optional
from ._utils import phi

import torch  # pylint:disable=[C0114]


def _closed_formula_covariance(
    exog: torch.Tensor,
    latent_mean: torch.Tensor,
    latent_sqrt_var: torch.Tensor,
    coef: torch.Tensor,
    n_samples: int,
) -> torch.Tensor:
    """
    Compute the closed-form covariance for the M step of the Pln model.

    Parameters:
    ----------
    exog : torch.Tensor
        Covariates with size (n, d).
    latent_mean : torch.Tensor
        Variational parameter with size (n, p).
    latent_sqrt_var : torch.Tensor
        Variational parameter with size (n, p).
    coef : torch.Tensor
        Model parameter with size (d, p).
    n_samples : int
        Number of samples (n).

    Returns:
    -------
    torch.Tensor
        The closed-form covariance with size (p, p).
    """
    if exog is None:
        XB = 0
    else:
        XB = exog @ coef
    m_minus_xb = latent_mean - XB
    closed = m_minus_xb.T @ m_minus_xb + torch.diag(
        torch.sum(torch.square(latent_sqrt_var), dim=0)
    )
    return closed / n_samples


def _closed_formula_coef(
    exog: torch.Tensor, latent_mean: torch.Tensor
) -> Optional[torch.Tensor]:
    """
    Compute the closed-form coef for the M step of the Pln model.

    Parameters:
    ----------
    exog : torch.Tensor
        Covariates with size (n, d).
    latent_mean : torch.Tensor
        Variational parameter with size (n, p).

    Returns:
    -------
    Optional[torch.Tensor]
        The closed-form coef with size (d, p) or None if exog is None.
    """
    if exog is None:
        return None
    return torch.inverse(exog.T @ exog) @ exog.T @ latent_mean


def _closed_formula_pi(
    offsets: torch.Tensor,
    latent_mean: torch.Tensor,
    latent_sqrt_var: torch.Tensor,
    dirac: torch.Tensor,
    exog: torch.Tensor,
    _coef_inflation: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the closed-form pi for the M step of the noPCA model.

    Parameters:
    ----------
    offsets : torch.Tensor
        Offset with size (n, p).
    latent_mean : torch.Tensor
        Variational parameter with size (n, p).
    latent_sqrt_var : torch.Tensor
        Variational parameter with size (n, p).
    dirac : torch.Tensor
        Dirac tensor.
    exog : torch.Tensor
        Covariates with size (n, d).
    _coef_inflation : torch.Tensor
        Inflation coefficient tensor.

    Returns:
    -------
    torch.Tensor
        The closed-form pi with the same size as dirac.
    """
    poiss_param = torch.exp(offsets + latent_mean + 0.5 * torch.square(latent_sqrt_var))
    return torch._sigmoid(poiss_param + torch.mm(exog, _coef_inflation)) * dirac


def _closed_formula_latent_prob(exog, coef, coef_infla, cov, dirac):
    if exog is not None:
        XB = exog @ coef
        XB_zero = exog @ coef_infla
    else:
        XB_zero = 0
        XB = 0
    XB_zero = exog @ coef_infla
    pi = torch.sigmoid(XB_zero)
    diag = torch.diag(cov)
    full_diag = diag.expand(exog.shape[0], -1)
    return torch.sigmoid(XB_zero - torch.log(phi(XB, full_diag))) * dirac
