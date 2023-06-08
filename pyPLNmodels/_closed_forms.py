from typing import Optional

import torch  # pylint:disable=[C0114]


def _closed_formula_covariance(
    covariates: torch.Tensor,
    latent_mean: torch.Tensor,
    latent_sqrt_var: torch.Tensor,
    coef: torch.Tensor,
    n_samples: int,
) -> torch.Tensor:
    """
    Compute the closed-form covariance for the M step of the Pln model.

    Parameters:
    ----------
    covariates : torch.Tensor
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
    if covariates is None:
        XB = 0
    else:
        XB = covariates @ coef
    m_minus_xb = latent_mean - XB
    closed = m_minus_xb.T @ m_minus_xb + torch.diag(
        torch.sum(torch.square(latent_sqrt_var), dim=0)
    )
    return closed / n_samples


def _closed_formula_coef(
    covariates: torch.Tensor, latent_mean: torch.Tensor
) -> Optional[torch.Tensor]:
    """
    Compute the closed-form coef for the M step of the Pln model.

    Parameters:
    ----------
    covariates : torch.Tensor
        Covariates with size (n, d).
    latent_mean : torch.Tensor
        Variational parameter with size (n, p).

    Returns:
    -------
    Optional[torch.Tensor]
        The closed-form coef with size (d, p) or None if covariates is None.
    """
    if covariates is None:
        return None
    return torch.inverse(covariates.T @ covariates) @ covariates.T @ latent_mean


def _closed_formula_pi(
    offsets: torch.Tensor,
    latent_mean: torch.Tensor,
    latent_sqrt_var: torch.Tensor,
    dirac: torch.Tensor,
    covariates: torch.Tensor,
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
    covariates : torch.Tensor
        Covariates with size (n, d).
    _coef_inflation : torch.Tensor
        Inflation coefficient tensor.

    Returns:
    -------
    torch.Tensor
        The closed-form pi with the same size as dirac.
    """
    poiss_param = torch.exp(offsets + latent_mean + 0.5 * torch.square(latent_sqrt_var))
    return torch._sigmoid(poiss_param + torch.mm(covariates, _coef_inflation)) * dirac
