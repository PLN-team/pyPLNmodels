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


def elbo_pln(
    *,  # pylint: disable=too-many-arguments
    endog: torch.Tensor,
    marginal_mean: torch.Tensor,
    offsets: torch.Tensor,
    latent_mean: torch.Tensor,
    latent_sqrt_variance: torch.Tensor,
    covariance: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the ELBO (Evidence Lower Bound) for the Pln model.

    Parameters:
    ----------
    endog : torch.Tensor
        Counts with size (n, p).
    marginal_mean : torch.Tensor
        Marginal mean with size (n, p).
    offsets : torch.Tensor
        Offset with size (n, p).
    latent_mean : torch.Tensor
        Variational parameter with size (n, p).
    latent_sqrt_variance : torch.Tensor
        Variational parameter with size (n, p).
    covariance : torch.Tensor
        Model parameter with size (p, p).

    Returns:
    -------
    torch.Tensor
        The ELBO (Evidence Lower Bound), of size one.
    """
    n_samples, _ = endog.shape
    latent_var = torch.square(latent_sqrt_variance)
    diag_s = torch.diag(torch.sum(latent_var, dim=0))
    offsets_plus_mean = offsets + latent_mean
    Omega = torch.inverse(covariance)
    m_minus_xb = latent_mean - marginal_mean
    m_moins_xb_outer = torch.mm(m_minus_xb.T, m_minus_xb)
    A = torch.exp(offsets_plus_mean + latent_var / 2)
    elbo = torch.sum(
        endog * offsets_plus_mean - A + 0.5 * torch.log(latent_var)
    ) - 1 / 2 * torch.sum(Omega * m_moins_xb_outer)
    elbo -= 0.5 * torch.trace(Omega @ diag_s)
    elbo -= 0.5 * n_samples * torch.logdet(covariance)
    elbo -= torch.sum(_log_stirling(endog))
    elbo += n_samples * endog.shape[1] / 2
    return elbo / n_samples


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
    latent_variance = torch.square(latent_sqrt_variance)

    log_intensity = offsets + marginal_mean + latent_mean @ components.T

    elbo = torch.sum(endog * log_intensity)
    elbo += torch.sum(
        -torch.exp(log_intensity + 0.5 * latent_variance @ (components * components).T)
    )
    elbo += 0.5 * torch.sum(torch.log(latent_variance))
    elbo -= 0.5 * torch.sum(torch.square(latent_mean) + latent_variance)
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
