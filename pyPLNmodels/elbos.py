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
        Counts with size (n_samples, dim).
    exog : torch.Tensor
        Covariates with size (n_samples, nb_cov).
    offsets : torch.Tensor
        Offsets with size (n_samples, dim).
    latent_mean : torch.Tensor
        Variational parameter with size (n_samples, dim).
    latent_sqrt_var : torch.Tensor
        Variational parameter with size (n_samples, dim).

    Returns:
    -------
    torch.Tensor
        The ELBO (Evidence Lower Bound) with size 1.
    """
    n_samples, _ = endog.shape
    latent_var = torch.square(latent_sqrt_var)
    offsets_plus_mean = offsets + latent_mean
    coef = _closed_formula_coef(exog, latent_mean)
    marginal_mean = exog @ coef if exog is not None else 0
    covariance = _closed_formula_covariance(
        marginal_mean, latent_mean, latent_sqrt_var, n_samples
    )

    elbo = -0.5 * n_samples * torch.logdet(covariance)
    elbo += torch.sum(
        endog * offsets_plus_mean
        - torch.exp(offsets_plus_mean + latent_var / 2)
        + 0.5 * torch.log(latent_var)
    )
    elbo -= torch.sum(_log_stirling(endog))

    return elbo


def elbo_plnpca(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    endog: torch.Tensor,
    marginal_mean: torch.Tensor,
    offsets: torch.Tensor,
    latent_mean: torch.Tensor,
    latent_sqrt_var: torch.Tensor,
    components: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the ELBO (Evidence Lower Bound) for the Pln model
    with PCA parametrization.

    Parameters:
    ----------
    endog : torch.Tensor
        Counts with size (n_samples, dim).
    marginal_mean : torch.Tensor
        The matrix product exog @ coef, of size (n_samples, dim)
    offsets : torch.Tensor
        Offset with size (n_samples, dim).
    latent_mean : torch.Tensor
        Variational parameter with size (n_samples, dim).
    latent_sqrt_var : torch.Tensor
        Variational parameter with size (n_samples, dim). More precisely it is the unsigned
        square root of the variational variance.
    components : torch.Tensor
        Model parameter with size (dim, rank).

    Returns:
    -------
    torch.Tensor
        The ELBO (Evidence Lower Bound) with size 1, with a gradient.
    """
    n_samples = endog.shape[0]
    rank = components.shape[1]
    log_intensity = offsets + marginal_mean + latent_mean @ components.T
    latent_var = torch.square(latent_sqrt_var)

    elbo = torch.sum(endog * log_intensity)
    elbo += torch.sum(
        -torch.exp(log_intensity + 0.5 * latent_var @ (components * components).T)
    )
    elbo += 0.5 * torch.sum(torch.log(latent_var))
    elbo -= 0.5 * torch.sum(torch.square(latent_mean) + torch.square(latent_sqrt_var))
    elbo -= torch.sum(_log_stirling(endog))
    elbo += 0.5 * n_samples * rank

    return elbo
