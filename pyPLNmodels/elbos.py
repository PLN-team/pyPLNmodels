import torch

from pyPLNmodels._closed_forms import _closed_formula_coef, _closed_formula_covariance
from pyPLNmodels._utils import _log_stirling, _trunc_log, _log1pexp


def profiled_elbo_pln(
    *,
    endog: torch.Tensor,
    exog: torch.Tensor,
    offsets: torch.Tensor,
    latent_mean: torch.Tensor,
    latent_sqrt_variance: torch.Tensor,
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
    latent_sqrt_variance : torch.Tensor
        Variational parameter with size (n_samples, dim).

    Returns:
    -------
    torch.Tensor
        The ELBO (Evidence Lower Bound) with size 1.
    """
    n_samples, _ = endog.shape
    latent_var = torch.square(latent_sqrt_variance)
    offsets_plus_mean = offsets + latent_mean
    coef = _closed_formula_coef(exog, latent_mean)
    marginal_mean = exog @ coef if exog is not None else 0
    covariance = _closed_formula_covariance(
        marginal_mean, latent_mean, latent_sqrt_variance, n_samples
    )

    elbo = -0.5 * n_samples * torch.logdet(covariance)
    elbo += torch.sum(
        endog * offsets_plus_mean
        - torch.exp(offsets_plus_mean + latent_var / 2)
        + 0.5 * torch.log(latent_var)
    )
    elbo -= torch.sum(_log_stirling(endog))

    return elbo


def elbo_plnpca(  # pylint: disable=too-many-arguments
    *,
    endog: torch.Tensor,
    marginal_mean: torch.Tensor,
    offsets: torch.Tensor,
    latent_mean: torch.Tensor,
    latent_sqrt_variance: torch.Tensor,
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
    latent_sqrt_variance : torch.Tensor
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
    latent_var = torch.square(latent_sqrt_variance)

    elbo = torch.sum(endog * log_intensity)
    elbo += torch.sum(
        -torch.exp(log_intensity + 0.5 * latent_var @ (components * components).T)
    )
    elbo += 0.5 * torch.sum(torch.log(latent_var))
    elbo -= 0.5 * torch.sum(
        torch.square(latent_mean) + torch.square(latent_sqrt_variance)
    )
    elbo -= torch.sum(_log_stirling(endog))
    elbo += 0.5 * n_samples * rank

    return elbo


# pylint: disable=too-many-arguments,too-many-locals
def elbo_zipln(
    *,
    endog,
    marginal_mean,
    offsets,
    latent_mean,
    latent_sqrt_variance,
    latent_prob,
    covariance,
    marginal_mean_inflation,
    dirac,
):
    """
    Compute the ELBO (Evidence Lower Bound) for the ZIPln model.

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
    latent_sqrt_variance : torch.Tensor
        Variational parameter with size (n_samples, dim). More precisely it is the unsigned
        square root of the variational variance.
    latent_prob : torch.Tensor
        Variational parameter for the latent probability with size (n_samples, dim).
    covariance : torch.Tensor
        The model covariance of size (dim,dim).
    marginal_mean_inflation : torch.Tensor
        The matrix product exog_inflation @ coef, of size (n_samples, dim)
    dirac : torch.Tensor
        Vector with 0s and 1s only, indicating whether endog is null or not.
        Size is (n_samples, dim).

    Returns:
    -------
    torch.Tensor
        The ELBO (Evidence Lower Bound) with size 1, with a gradient.
    """
    if torch.norm(latent_prob * dirac - latent_prob) > 1e-8:
        raise RuntimeError("Latent probability error.")

    n_samples, dim = endog.shape
    latent_var = latent_sqrt_variance**2
    offsets_plus_latent_mean = offsets + latent_mean
    mean_diff = latent_mean - marginal_mean

    poisson_mean = torch.exp(offsets_plus_latent_mean + latent_var / 2)
    poisson_term = torch.sum(
        (1 - latent_prob)
        * (endog * offsets_plus_latent_mean - poisson_mean - _log_stirling(endog))
    )

    omega = torch.inverse(covariance)
    mean_diff_outer = torch.mm(mean_diff.T, mean_diff)
    quadratic_term = torch.sum(
        -0.5 * omega * (mean_diff_outer + torch.diag(torch.sum(latent_var, axis=0)))
    )

    inflation_term = torch.sum(
        latent_prob * marginal_mean_inflation - _log1pexp(marginal_mean_inflation)
    )

    entropy_term = torch.sum(0.5 * torch.log(latent_var))

    kl_divergence_term = torch.sum(
        -latent_prob * _trunc_log(latent_prob)
        - (1 - latent_prob) * _trunc_log(1 - latent_prob)
    )

    logdet_term = -0.5 * n_samples * torch.logdet(covariance)

    elbo = (
        poisson_term
        + quadratic_term
        + inflation_term
        + entropy_term
        + kl_divergence_term
        + logdet_term
        + 0.5 * n_samples * dim
    )
    return elbo
