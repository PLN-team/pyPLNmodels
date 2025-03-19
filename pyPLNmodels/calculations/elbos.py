import torch

from pyPLNmodels.calculations._closed_forms import (
    _closed_formula_coef,
    _closed_formula_covariance,
    _closed_formula_diag_covariance,
)
from pyPLNmodels.utils._utils import _log_stirling, _trunc_log, _log1pexp, _remove_nan


def profiled_elbo_pln(
    *,
    endog: torch.Tensor,
    exog: torch.Tensor,
    offsets: torch.Tensor,
    latent_mean: torch.Tensor,
    latent_sqrt_variance: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the ELBO (Evidence Lower Bound) for the `Pln` model with profiled
    model parameters (i.e., the model parameters are derived directly from the
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

    Returns
    -------
    torch.Tensor
        The ELBO (Evidence Lower Bound) with size 1.
    """
    n_samples, _ = endog.shape
    latent_var = latent_sqrt_variance**2
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


def elbo_pln_diag(
    *,  # pylint: disable=too-many-arguments
    endog: torch.Tensor,
    marginal_mean: torch.Tensor,
    offsets: torch.Tensor,
    latent_mean: torch.Tensor,
    latent_sqrt_variance: torch.Tensor,
    diag_precision: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the ELBO (Evidence Lower Bound) for the `Pln` model.

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
    diag_precision : torch.Tensor
        Model parameter with size p, each value being positive.

    Returns
    -------
    torch.Tensor
        The ELBO (Evidence Lower Bound), of size one.
    """
    n_samples, _ = endog.shape
    latent_var = latent_sqrt_variance**2
    offsets_plus_mean = offsets + latent_mean
    m_minus_xb = latent_mean - marginal_mean
    exp_term = torch.exp(offsets_plus_mean + latent_var / 2)
    elbo = torch.sum(endog * offsets_plus_mean - exp_term + 0.5 * torch.log(latent_var))
    trace_term = torch.sum(diag_precision * torch.sum(latent_var, dim=0)) + torch.sum(
        torch.sum(m_minus_xb**2, dim=0) * diag_precision
    )
    elbo += -0.5 * trace_term
    elbo += 0.5 * n_samples * torch.sum(torch.log(diag_precision))
    elbo -= torch.sum(_log_stirling(endog))
    elbo += n_samples * endog.shape[1] / 2
    return elbo


def per_sample_elbo_pln_mixture_diag(
    *,
    endog: torch.Tensor,
    marginal_means: torch.Tensor,
    offsets: torch.Tensor,
    latent_means: torch.Tensor,
    latent_sqrt_variances: torch.Tensor,
    diag_precisions: torch.Tensor,
):  # pylint: disable=too-many-arguments
    """
    Compute the ELBO (Evidence LOwer Bound) for the PlnMixture model in a vectorized way.

    endog : torch.Tensor
        Counts with size (n, p).
    marginal_means : torch.Tensor
        Marginal mean for each cluster with size (n_cluster, n, p).
    offsets : torch.Tensor
        Offset with size (n, p).
    latent_means : torch.Tensor
        Variational parameter for each cluster with size (n_cluster, n, p).
    latent_sqrt_variances : torch.Tensor
        Variational parameter for each cluster with size (n_cluster, n, p).
    diag_precisions : torch.Tensor
        Model parameter for each cluster with size (n_cluster, p), each value being positive.

    Returns
    -------
    torch.tensor
        the elbo (evidence lower bound) for each sample, of size n_samples.
    """
    n_samples, dim = endog.shape
    latent_variances = latent_sqrt_variances**2
    offsets_plus_latent_means = offsets.unsqueeze(0) + latent_means
    latent_means_minus_marginal_means = latent_means - marginal_means
    exp_term = torch.exp(offsets_plus_latent_means + latent_variances / 2)
    elbo = (
        endog.unsqueeze(0) * offsets_plus_latent_means
        - exp_term
        + 0.5 * torch.log(latent_variances + 1e-10)
    )
    elbo += -0.5 * diag_precisions.unsqueeze(1) * latent_variances
    elbo += -0.5 * latent_means_minus_marginal_means**2 * diag_precisions.unsqueeze(1)
    elbo += 0.5 * torch.log(diag_precisions).unsqueeze(1)
    elbo += -_log_stirling(endog.unsqueeze(0))
    elbo += torch.ones(latent_variances.shape[0], n_samples, dim).to(endog.device) / 2
    return torch.sum(elbo, dim=-1)


def per_entry_elbo_pln_diag(
    *,  # pylint: disable=too-many-arguments
    endog: torch.Tensor,
    marginal_mean: torch.Tensor,
    offsets: torch.Tensor,
    latent_mean: torch.Tensor,
    latent_sqrt_variance: torch.Tensor,
    diag_precision: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the ELBO (Evidence Lower Bound) for the `Pln` model
    with diagonal covariances, per sample. This is needed for the mixture models.

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
    diag_precision : torch.Tensor
        Model parameter with size p, each value being positive.

    Returns
    -------
    torch.tensor
        the elbo (evidence lower bound) for each sample, of size n_samples.
    """
    n_samples, dim = endog.shape
    latent_variance = latent_sqrt_variance**2
    offsets_plus_latent_mean = offsets + latent_mean
    latent_mean_minus_marginal_mean = latent_mean - marginal_mean
    exp_term = torch.exp(offsets_plus_latent_mean + latent_variance / 2)
    elbo = (
        endog * offsets_plus_latent_mean
        - exp_term
        + 0.5 * torch.log(latent_variance + 1e-10)
    )

    elbo += -0.5 * diag_precision.unsqueeze(0) * latent_variance

    elbo += -0.5 * latent_mean_minus_marginal_mean**2 * diag_precision.unsqueeze(0)
    elbo += 0.5 * torch.log(diag_precision).unsqueeze(0)
    elbo += -_log_stirling(endog)
    elbo += torch.ones(n_samples, dim).to(endog.device) / 2
    return elbo


def weighted_elbo_pln_diag(
    *,  # pylint: disable=too-many-arguments
    endog: torch.Tensor,
    marginal_mean: torch.Tensor,
    offsets: torch.Tensor,
    latent_mean: torch.Tensor,
    latent_sqrt_variance: torch.Tensor,
    diag_precision: torch.Tensor,
    latent_prob: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the ELBO (Evidence Lower Bound) for the `Pln` model with weights.
    The weights are here to handle mixture models.

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
    diag_precision : torch.Tensor
        Model parameter with size p, each value being positive.
    latent_prob: torch.Tensor
        Vector of latent_prob on the samples, of size (n).

    Returns
    -------
    torch.Tensor
        The ELBO (Evidence Lower Bound), of size one.
    """
    per_entry_elbo = per_entry_elbo_pln_diag(
        endog=endog,
        marginal_mean=marginal_mean,
        offsets=offsets,
        latent_mean=latent_mean,
        latent_sqrt_variance=latent_sqrt_variance,
        diag_precision=diag_precision,
    )
    return torch.sum(torch.sum(per_entry_elbo, dim=-1) * latent_prob)


def profiled_elbo_pln_diag(
    *,
    endog: torch.Tensor,
    exog: torch.Tensor,
    offsets: torch.Tensor,
    latent_mean: torch.Tensor,
    latent_sqrt_variance: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the ELBO (Evidence Lower Bound) for the `Pln` model.

    Parameters:
    ----------
    endog : torch.Tensor
        Counts with size (n, p).
    exog : torch.Tensor
        Covariates with size (n_samples, nb_cov).
    offsets : torch.Tensor
        Offset with size (n, p).
    latent_mean : torch.Tensor
        Variational parameter with size (n, p).
    latent_sqrt_variance : torch.Tensor
        Variational parameter with size (n, p).

    Returns
    -------
    torch.Tensor
        The ELBO (Evidence Lower Bound), of size one.
    """
    n_samples, _ = endog.shape
    latent_var = latent_sqrt_variance**2
    offsets_plus_mean = offsets + latent_mean
    coef = _closed_formula_coef(exog, latent_mean)
    marginal_mean = exog @ coef if exog is not None else 0
    diag_covariance = _closed_formula_diag_covariance(
        marginal_mean, latent_mean, latent_sqrt_variance, n_samples
    )
    elbo = -0.5 * n_samples * torch.sum(torch.log(diag_covariance))
    elbo += torch.sum(
        endog * offsets_plus_mean
        - torch.exp(offsets_plus_mean + latent_var / 2)
        + 0.5 * torch.log(latent_var)
    )
    elbo -= torch.sum(_log_stirling(endog))
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
    precision,
    marginal_mean_inflation,
):
    """
    Compute the ELBO (Evidence Lower Bound) for the `ZIPln` model.

    Parameters:
    ----------
    endog : torch.Tensor
        Counts with size (n_samples, dim).
    marginal_mean : torch.Tensor
        The matrix product `exog @ coef`, of size (n_samples, dim).
    offsets : torch.Tensor
        Offset with size (n_samples, dim).
    latent_mean : torch.Tensor
        Variational parameter with size (n_samples, dim).
    latent_sqrt_variance : torch.Tensor
        Variational parameter with size (n_samples, dim). More precisely, it is the unsigned
        square root of the variational variance.
    latent_prob : torch.Tensor
        Variational parameter for the latent probability with size (n_samples, dim).
    precision : torch.Tensor
        The model precision of size (dim, dim).
    marginal_mean_inflation : torch.Tensor
        The matrix product `exog_inflation @ coef`, of size (n_samples, dim).

    Returns
    -------
    torch.Tensor
        The ELBO (Evidence Lower Bound) with size 1, with a gradient.
    """
    n_samples, dim = endog.shape
    latent_var = latent_sqrt_variance**2
    offsets_plus_latent_mean = offsets + latent_mean
    mean_diff = latent_mean - marginal_mean

    poisson_mean = torch.exp(offsets_plus_latent_mean + latent_var / 2)
    poisson_term = torch.sum(
        (1 - latent_prob)
        * (endog * offsets_plus_latent_mean - poisson_mean - _log_stirling(endog))
    )

    mean_diff_outer = torch.mm(mean_diff.T, mean_diff)
    quadratic_term = torch.sum(
        -0.5 * precision * (mean_diff_outer + torch.diag(torch.sum(latent_var, axis=0)))
    )
    inflation_term = torch.sum(
        latent_prob * marginal_mean_inflation - _log1pexp(marginal_mean_inflation)
    )
    entropy_term = torch.sum(0.5 * torch.log(latent_var))
    kl_divergence_term = torch.sum(
        -latent_prob * _trunc_log(latent_prob)
        - (1 - latent_prob) * _trunc_log(1 - latent_prob)
    )
    logdet_term = 0.5 * n_samples * torch.logdet(precision)
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


def elbo_ziplnpca(
    *,
    endog,
    marginal_mean,
    offsets,
    latent_mean,
    latent_sqrt_variance,
    latent_prob,
    components,
    marginal_mean_inflation,
    dirac,
):  # pylint: disable=too-many-positional-arguments
    """
    Compute the ELBO (Evidence Lower Bound) for the `Pln` model
    with PCA parametrization.

    Parameters:
    ----------
    endog : torch.Tensor
        Counts with size (n_samples, dim).
    marginal_mean : torch.Tensor
        The matrix product `exog @ coef`, of size (n_samples, dim).
    offsets : torch.Tensor
        Offset with size (n_samples, dim).
    latent_mean : torch.Tensor
        Variational parameter with size (n_samples, dim).
    latent_sqrt_variance : torch.Tensor
        Variational parameter with size (n_samples, dim). More precisely, it is the unsigned
        square root of the variational variance.
    components : torch.Tensor
        Model parameter with size (dim, rank).
    dirac : torch.Tensor
        Vector with 0s and 1s only, indicating whether `endog` is null or not.
        Size is (n_samples, dim).

    Returns
    -------
    torch.Tensor
        The ELBO (Evidence Lower Bound) with size 1, with a gradient.
    """
    if torch.norm(latent_prob * dirac - latent_prob) > 1e-8:
        raise RuntimeError(
            "Latent probability error. It has non-zeros where it should be zeros."
        )
    complement_prob = 1 - latent_prob
    log_intensity = offsets + marginal_mean + latent_mean @ components.T
    latent_variance = latent_sqrt_variance**2
    elbo = torch.sum(
        complement_prob
        * (
            endog * log_intensity
            - torch.exp(
                log_intensity + 1 / 2 * torch.matmul(latent_variance, (components**2).T)
            )
            - _log_stirling(endog)
        )
    )
    elbo += -0.5 * torch.sum(
        latent_mean**2 + latent_variance - torch.log(latent_variance)
    )
    elbo += torch.sum(
        latent_prob * marginal_mean_inflation - _log1pexp(marginal_mean_inflation)
    )
    elbo += -torch.sum(
        latent_prob * _trunc_log(latent_prob)
        + (complement_prob) * _trunc_log(complement_prob)
    )
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
    Compute the ELBO (Evidence Lower Bound) for the `Pln` model
    with PCA parametrization.

    Parameters:
    ----------
    endog : torch.Tensor
        Counts with size (n_samples, dim).
    marginal_mean : torch.Tensor
        The matrix product `exog @ coef`, of size (n_samples, dim).
    offsets : torch.Tensor
        Offset with size (n_samples, dim).
    latent_mean : torch.Tensor
        Variational parameter with size (n_samples, dim).
    latent_sqrt_variance : torch.Tensor
        Variational parameter with size (n_samples, dim). More precisely, it is the unsigned
        square root of the variational variance.
    components : torch.Tensor
        Model parameter with size (dim, rank).

    Returns
    -------
    torch.Tensor
        The ELBO (Evidence Lower Bound) with size 1.
    """
    n_samples = endog.shape[0]
    rank = components.shape[1]
    latent_variance = latent_sqrt_variance**2
    log_intensity = offsets + marginal_mean + latent_mean @ components.T
    elbo = torch.sum(endog * log_intensity)
    elbo += torch.sum(
        -torch.exp(log_intensity + 0.5 * latent_variance @ (components * components).T)
    )
    elbo += 0.5 * torch.sum(torch.log(latent_variance))
    elbo -= 0.5 * torch.sum(latent_mean**2 + latent_variance)
    elbo -= torch.sum(_log_stirling(endog))
    elbo += 0.5 * n_samples * rank
    return elbo


def profiled_elbo_zipln(
    *,
    endog,
    exog,
    offsets,
    latent_mean,
    latent_sqrt_variance,
    latent_prob,
    marginal_mean_inflation,
    dirac,
):
    """
    Compute the ELBO (Evidence Lower Bound) for the `ZIPln` model in a profiled fashion,
    where closed forms are used to avoid matrix inversion.

    Parameters:
    ----------
    endog : torch.Tensor
        Counts with size (n_samples, dim).
    exog : torch.Tensor
        Covariates with size (n_samples, nb_cov).
    offsets : torch.Tensor
        Offset with size (n_samples, dim).
    latent_mean : torch.Tensor
        Variational parameter with size (n_samples, dim).
    latent_sqrt_variance : torch.Tensor
        Variational parameter with size (n_samples, dim). More precisely, it is the unsigned
        square root of the variational variance.
    latent_prob : torch.Tensor
        Variational parameter for the latent probability with size (n_samples, dim).
    marginal_mean_inflation : torch.Tensor
        The matrix product `exog_inflation @ coef`, of size (n_samples, dim).
    dirac : torch.Tensor
        Vector with 0s and 1s only, indicating whether `endog` is null or not.
        Size is (n_samples, dim).

    Returns
    -------
    torch.Tensor
        The ELBO (Evidence Lower Bound) with size 1, with a gradient.
    """
    if torch.norm(latent_prob * dirac - latent_prob) > 1e-8:
        raise RuntimeError(
            "Latent probability error. It has non-zeros where it should be zeros."
        )
    n_samples, dim = endog.shape
    latent_var = latent_sqrt_variance**2
    offsets_plus_latent_mean = offsets + latent_mean

    poisson_mean = torch.exp(offsets_plus_latent_mean + latent_var / 2)
    poisson_term = torch.sum(
        (1 - latent_prob)
        * (endog * offsets_plus_latent_mean - poisson_mean - _log_stirling(endog))
    )
    quadratic_term = -n_samples * dim / 2
    inflation_term = torch.sum(
        latent_prob * marginal_mean_inflation - _log1pexp(marginal_mean_inflation)
    )

    entropy_term = torch.sum(0.5 * torch.log(latent_var))

    kl_divergence_term = torch.sum(
        -latent_prob * _trunc_log(latent_prob)
        - (1 - latent_prob) * _trunc_log(1 - latent_prob)
    )
    coef = _closed_formula_coef(exog, latent_mean)
    marginal_mean = exog @ coef if exog is not None else 0
    covariance = _closed_formula_covariance(
        marginal_mean, latent_mean, latent_sqrt_variance, n_samples
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


def elbo_pln(
    *,  # pylint: disable=too-many-arguments
    endog: torch.Tensor,
    marginal_mean: torch.Tensor,
    offsets: torch.Tensor,
    latent_mean: torch.Tensor,
    latent_sqrt_variance: torch.Tensor,
    precision: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the ELBO (Evidence Lower Bound) for the `Pln` model.

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
    precision : torch.Tensor
        Model parameter with size (p, p).

    Returns
    -------
    torch.Tensor
        The ELBO (Evidence Lower Bound), of size one.
    """
    n_samples, _ = endog.shape
    latent_var = latent_sqrt_variance**2
    diag_s = torch.diag(torch.sum(latent_var, dim=0))
    offsets_plus_mean = offsets + latent_mean
    m_minus_xb = latent_mean - marginal_mean
    m_minus_xb_outer = torch.mm(m_minus_xb.T, m_minus_xb)
    exp_term = torch.exp(offsets_plus_mean + latent_var / 2)
    elbo = torch.sum(
        endog * offsets_plus_mean
        - exp_term
        + 0.5 * torch.log(latent_var)
        - _log_stirling(endog)
    )
    trace_term = torch.trace(precision @ diag_s)
    elbo += -0.5 * trace_term
    elbo += -0.5 * torch.sum(precision * m_minus_xb_outer)

    elbo += 0.5 * n_samples * torch.logdet(precision)
    elbo += n_samples * endog.shape[1] / 2
    return elbo


def per_sample_elbo_pln(
    *,  # pylint: disable=too-many-arguments
    endog: torch.Tensor,
    marginal_mean: torch.Tensor,
    offsets: torch.Tensor,
    latent_mean: torch.Tensor,
    latent_sqrt_variance: torch.Tensor,
    precision: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the ELBO (Evidence Lower Bound) for the `Pln` model.

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
    precision : torch.Tensor
        Model parameter with size (p, p).

    Returns
    -------
    torch.Tensor
        The ELBO (Evidence Lower Bound), of size one.
    """
    latent_variance = latent_sqrt_variance**2
    offsets_plus_mean = offsets + latent_mean
    m_minus_xb = latent_mean - marginal_mean
    exp_term = torch.exp(offsets_plus_mean + latent_variance / 2)
    elbo = (
        endog * offsets_plus_mean
        - exp_term
        + 0.5 * torch.log(latent_variance)
        - _log_stirling(endog)
    )
    elbo += -0.5 * latent_variance * torch.diag(precision)
    elbo += 0.5 * torch.logdet(precision) / endog.shape[1] + 0.5
    elbo += -0.5 * (m_minus_xb @ precision) * (m_minus_xb)
    return torch.sum(elbo, dim=-1)


def elbo_plnar_diag_autoreg(
    *,
    endog: torch.Tensor,
    marginal_mean: torch.Tensor,
    offsets: torch.Tensor,
    latent_mean: torch.Tensor,
    latent_sqrt_variance: torch.Tensor,
    precision: torch.Tensor,
    ar_coef: torch.Tensor,
):  # pylint: disable=too-many-positional-arguments
    """
    Computes the ELBO for an autoregressive PLN model, i.e. PlnAR model.
    Both the autoregressive coefficient and covariance are diagonal (size p).

    Parameters:
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
    precision : torch.Tensor
        Model parameter (inverse covariance matrix) with size (p).
    ar_coef: torch.Tensor
        Autoregressive model parameter with size (p).
    """
    offsets_plus_mean = _remove_nan(offsets + latent_mean)
    latent_variance = _remove_nan(latent_sqrt_variance**2)
    autoreg_precision = 1 / (1 - ar_coef**2) * precision
    latent_diff = _remove_nan(latent_mean - marginal_mean)
    mask = torch.isfinite(endog)

    ar_elbo = torch.zeros_like(latent_diff)
    ar_elbo[1:] = (
        -0.5
        * (latent_diff[1:] - ar_coef.unsqueeze(0) * latent_diff[:-1]) ** 2
        * autoreg_precision.unsqueeze(0)
        + 0.5 * torch.log(autoreg_precision).unsqueeze(0)
        - 0.5
        * autoreg_precision.unsqueeze(0)
        * (latent_variance[1:] + (ar_coef.unsqueeze(0) ** 2) * latent_variance[:-1])
    )
    ar_elbo[0] = 0.5 * (
        -((latent_diff[0]) ** 2) * precision
        - latent_variance[0] * precision
        + torch.log(precision)
    )
    elbo = (
        endog * offsets_plus_mean
        - torch.exp(offsets_plus_mean + latent_variance / 2)
        + 0.5 * _remove_nan(torch.log(latent_variance))
        - _remove_nan(_log_stirling(endog))
    )
    elbo += ar_elbo
    elbo += 1 / 2
    elbo = elbo * mask
    elbo = torch.sum(torch.nan_to_num(elbo))
    return elbo


def smart_elbo_plnar_full_autoreg(
    *,
    endog: torch.Tensor,
    marginal_mean: torch.Tensor,
    offsets: torch.Tensor,
    latent_mean: torch.Tensor,
    latent_sqrt_variance: torch.Tensor,
    ortho_components: torch.Tensor,
    diag_covariance: torch.Tensor,
    diag_ar_coef: torch.Tensor,
):
    """
    Computes the ELBO for an autoregressive PLN model, i.e. PlnAR model.
    The autoregression is full, as well as the covariance matrix.
    This is the same as the elbo_plnar_full_autoreg function, but
    builds the covariance and ar_coef based on the orthogonal components
    and eigenvalues.

    Parameters:
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
    ortho_components : torch.Tensor
        Orthogonal components of both the covariance matrix and
        ar_coef matrix, of size (p,p).
    diag_covariance : torch.Tensor
        The eigenvalues of the covariance, of size (p).
    diag_covariance : torch.Tensor
        The eigenvalues of the ar_coef matrix, of size (p).

    See also
    --------
    :func:`pyPLNmodels.calculations.elbos.elbo_plnar_full_autoreg`
    :func:`pyPLNmodels.calculations.elbos.elbo_plnar_scalar_autoreg`
    :func:`pyPLNmodels.calculations.elbos.elbo_plnar_diag_autoreg`
    """
    offsets_plus_mean = _remove_nan(offsets + latent_mean)
    latent_variance = _remove_nan(latent_sqrt_variance**2)
    autoreg_precision = (
        ortho_components
        * (1 / (diag_covariance * (1 - diag_ar_coef**2)))
        @ (ortho_components.T)
    )
    precision = ortho_components * (1 / diag_covariance) @ (ortho_components.T)
    ar_coef = ortho_components * diag_ar_coef @ (ortho_components.T)

    logdet_precision = -torch.sum(torch.log(diag_covariance))
    logdet_autoreg = -torch.sum(torch.log(diag_covariance)) + torch.sum(
        torch.log(1 / (1 - diag_ar_coef**2))
    )

    latent_minus_marginal = latent_mean - marginal_mean

    latent_diff = _remove_nan(latent_minus_marginal)
    latent_diff_back = latent_diff[1:]
    ar_term = latent_diff[:-1] @ ar_coef
    latent_diff_back_minus_ar = latent_diff_back - ar_term

    ar_elbo = torch.zeros_like(latent_diff)

    ar_elbo[1:] = 0.5 * (
        -(latent_diff_back_minus_ar @ autoreg_precision) * latent_diff_back_minus_ar
        + 1
        / endog.shape[1]
        * logdet_autoreg.unsqueeze(0)
        .unsqueeze(1)
        .repeat_interleave(endog.shape[0] - 1, dim=0)
        - (latent_variance[:-1] * torch.diag(autoreg_precision))
        - latent_variance[1:] * torch.diag(ar_coef @ autoreg_precision @ ar_coef)
    )
    ar_elbo[0] = (
        -0.5 * ((latent_diff[0]) @ precision) * (latent_diff[0])
        - 0.5 * latent_variance[0] * torch.diag(precision)
        + 0.5 / endog.shape[1] * logdet_precision
    )

    elbo = (
        endog * offsets_plus_mean
        - torch.exp(offsets_plus_mean + latent_variance / 2)
        + 0.5 * torch.log(latent_variance)
        - _remove_nan(_log_stirling(endog))
    )
    elbo += ar_elbo
    elbo += 1 / 2
    elbo = torch.sum(torch.nan_to_num(elbo))
    return elbo


def elbo_plnar_full_autoreg(
    *,
    endog: torch.Tensor,
    marginal_mean: torch.Tensor,
    offsets: torch.Tensor,
    latent_mean: torch.Tensor,
    latent_sqrt_variance: torch.Tensor,
    precision: torch.Tensor,
    ar_coef: torch.Tensor,
):  # pylint: disable=too-many-positional-arguments
    """
    Computes the ELBO for an autoregressive PLN model, i.e. PlnAR model.
    The autoregression is full, as well as the covariance matrix.

    Parameters:
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
    precision : torch.Tensor
        Model parameter (inverse covariance matrix) with size (p, p).
    ar_coef: torch.Tensor
        Autoregressive model matrix parameter with size (p,p).

    See also
    --------
    :func:`pyPLNmodels.calculations.elbos.smart_elbo_plnar_full_autoreg`
    :func:`pyPLNmodels.calculations.elbos.elbo_plnar_scalar_autoreg`
    :func:`pyPLNmodels.calculations.elbos.elbo_plnar_diag_autoreg`
    """
    offsets_plus_mean = _remove_nan(offsets + latent_mean)
    latent_variance = _remove_nan(latent_sqrt_variance**2)
    covariance = torch.inverse(precision)
    autoreg_covariance = covariance - ar_coef @ covariance @ ar_coef
    autoreg_precision = torch.inverse(autoreg_covariance)

    logdet_precision = torch.logdet(precision)
    logdet_autoreg = torch.logdet(autoreg_precision)

    latent_minus_marginal = latent_mean - marginal_mean

    latent_diff = _remove_nan(latent_minus_marginal)
    latent_diff_back = latent_diff[1:]
    ar_term = latent_diff[:-1] @ ar_coef
    latent_diff_back_minus_ar = latent_diff_back - ar_term

    ar_elbo = torch.zeros_like(latent_diff)

    ar_elbo[1:] = 0.5 * (
        -(latent_diff_back_minus_ar @ autoreg_precision) * latent_diff_back_minus_ar
        + 1
        / endog.shape[1]
        * logdet_autoreg.unsqueeze(0)
        .unsqueeze(1)
        .repeat_interleave(endog.shape[0] - 1, dim=0)
        - (latent_variance[:-1] * torch.diag(autoreg_precision))
        - latent_variance[1:] * torch.diag(ar_coef @ autoreg_precision @ ar_coef)
    )
    ar_elbo[0] = (
        -0.5 * ((latent_diff[0]) @ precision) * (latent_diff[0])
        - 0.5 * latent_variance[0] * torch.diag(precision)
        + 0.5 / endog.shape[1] * logdet_precision
    )

    elbo = (
        endog * offsets_plus_mean
        - torch.exp(offsets_plus_mean + latent_variance / 2)
        + 0.5 * torch.log(latent_variance)
        - _remove_nan(_log_stirling(endog))
    )
    elbo += ar_elbo
    elbo += 1 / 2
    elbo = torch.sum(torch.nan_to_num(elbo))
    return elbo


def elbo_plnar_scalar_autoreg(
    *,
    endog: torch.Tensor,
    marginal_mean: torch.Tensor,
    offsets: torch.Tensor,
    latent_mean: torch.Tensor,
    latent_sqrt_variance: torch.Tensor,
    precision: torch.Tensor,
    ar_coef: torch.Tensor,
):  # pylint: disable=too-many-positional-arguments
    """
    Computes the ELBO for an autoregressive PLN model, i.e. PlnAR model.
    The autoregressive coefficient is of size 1 (scalar).
    The covariance matrix is full (p,p).

    Parameters:
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
    precision : torch.Tensor
        Model parameter (inverse covariance matrix) with size (p, p).
    ar_coef: torch.Tensor
        Autoregressive model parameter with size (1).

    See also
    --------
    :func:`pyPLNmodels.calculations.elbos.elbo_plnar_full_autoreg`
    :func:`pyPLNmodels.calculations.elbos.elbo_plnar_diag_autoreg`
    """
    offsets_plus_mean = _remove_nan(offsets + latent_mean)
    latent_variance = _remove_nan(latent_sqrt_variance**2)
    multiplier = 1 / (1 - ar_coef**2)
    autoreg_precision = multiplier * precision

    logdet_precision = torch.logdet(precision)
    logdet_autoreg = endog.shape[1] * torch.log(multiplier) + logdet_precision

    latent_minus_marginal = latent_mean - marginal_mean

    latent_diff = _remove_nan(latent_minus_marginal)
    latent_diff_back = latent_diff[1:]
    ar_term = ar_coef * latent_diff[:-1]
    latent_diff_back_minus_ar = latent_diff_back - ar_term

    ar_elbo = torch.zeros_like(latent_diff)

    ar_elbo[1:] = 0.5 * (
        -(latent_diff_back_minus_ar @ autoreg_precision) * latent_diff_back_minus_ar
        + 1
        / endog.shape[1]
        * logdet_autoreg.unsqueeze(1).repeat_interleave(endog.shape[0] - 1, dim=0)
        - (latent_variance[1:] * torch.diag(autoreg_precision))
        - (ar_coef**2 * latent_variance[:-1] * torch.diag(autoreg_precision))
    )
    ar_elbo[0] = (
        -0.5 * ((latent_diff[0]) @ precision) * (latent_diff[0])
        - 0.5 * latent_variance[0] * torch.diag(precision)
        + 0.5 / endog.shape[1] * logdet_precision
    )
    elbo = (
        endog * offsets_plus_mean
        - torch.exp(offsets_plus_mean + latent_variance / 2)
        + 0.5 * torch.log(latent_variance)
        - _remove_nan(_log_stirling(endog))
    )
    elbo += ar_elbo
    elbo += 1 / 2
    elbo = torch.sum(torch.nan_to_num(elbo))
    return elbo
