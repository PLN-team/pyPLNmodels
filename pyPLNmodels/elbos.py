import torch

from pyPLNmodels._closed_forms import _closed_formula_coef, _closed_formula_covariance
from pyPLNmodels._utils import _log_stirling


def profiled_elbo_pln(
    endog: torch.Tensor,
    exog: torch.Tensor,
    offsets: torch.Tensor,
    latent_mean: torch.Tensor,
    latent_sqrt_var: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the ELBO (Evidence Lower Bound) for the Pln model with profiled
    model parameters (i.e the model parameters are derived directly from the
    latent parameters).

    Parameters:
    ----------
    endog : torch.Tensor
        Counts with size (n, p).
    exog : torch.Tensor
        Covariates with size (n, d).
    offsets : torch.Tensor
        Offsets with size (n, p).
    latent_mean : torch.Tensor
        Variational parameter with size (n, p).
    latent_sqrt_var : torch.Tensor
        Variational parameter with size (n, p).

    Returns:
    -------
    torch.Tensor
        The ELBO (Evidence Lower Bound) with size 1.
    """
    n_samples, _ = endog.shape
    s_squared = torch.square(latent_sqrt_var)
    offsets_plus_mean = offsets + latent_mean
    closed_coef = _closed_formula_coef(exog, latent_mean)
    marginal_mean = exog @ closed_coef if exog is not None else 0
    closed_covariance = _closed_formula_covariance(
        marginal_mean, latent_mean, latent_sqrt_var, n_samples
    )
    elbo = -0.5 * n_samples * torch.logdet(closed_covariance)
    elbo += torch.sum(
        endog * offsets_plus_mean
        - torch.exp(offsets_plus_mean + s_squared / 2)
        + 0.5 * torch.log(s_squared)
    )
    elbo -= torch.sum(_log_stirling(endog))
    return elbo / n_samples
