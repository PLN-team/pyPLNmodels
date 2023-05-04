import torch  # pylint:disable=[C0114]
from ._utils import log_stirling, trunc_log
from ._closed_forms import closed_formula_covariance, closed_formula_coef


def elbo_pln(counts, covariates, offsets, latent_mean, latent_var, covariance, coef):
    """
    Compute the ELBO (Evidence LOwer Bound) for the PLN model. See the doc for more details
    on the computation.

    Args:
        counts: torch.tensor. Counts with size (n,p)
        0: torch.tensor. Offset, size (n,p)
        covariates: torch.tensor. Covariates, size (n,d)
        latent_mean: torch.tensor. Variational parameter with size (n,p)
        latent_var: torch.tensor. Variational parameter with size (n,p)
        covariance: torch.tensor. Model parameter with size (p,p)
        coef: torch.tensor. Model parameter with size (d,p)
    Returns:
        torch.tensor of size 1 with a gradient.
    """
    n_samples, dim = counts.shape
    s_rond_s = torch.multiply(latent_var, latent_var)
    offsets_plus_m = offsets + latent_mean
    if covariates is None:
        XB = 0
    else:
        XB = covariates @ coef
    m_minus_xb = latent_mean - XB
    d_plus_minus_xb2 = torch.diag(torch.sum(s_rond_s, dim=0)) + torch.mm(
        m_minus_xb.T, m_minus_xb
    )
    elbo = -n_samples / 2 * torch.logdet(covariance)
    elbo += torch.sum(
        torch.multiply(counts, offsets_plus_m)
        - torch.exp(offsets_plus_m + s_rond_s / 2)
        + 1 / 2 * torch.log(s_rond_s)
    )
    elbo -= 1 / 2 * torch.trace(torch.mm(torch.inverse(covariance), d_plus_minus_xb2))
    elbo -= torch.sum(log_stirling(counts))
    elbo += n_samples * dim / 2
    return elbo / n_samples


def profiled_elbo_pln(counts, covariates, offsets, latent_mean, latent_var):
    """
    Compute the ELBO (Evidence LOwer Bound) for the PLN model. We use the fact that covariance and coef are
    completely determined by latent_mean,latent_var, and the covariates. See the doc for more details
    on the computation.

    Args:
        counts: torch.tensor. Counts with size (n,p)
        0: torch.tensor. Offset, size (n,p)
        covariates: torch.tensor. Covariates, size (n,d)
        latent_mean: torch.tensor. Variational parameter with size (n,p)
        latent_var: torch.tensor. Variational parameter with size (n,p)
        covariance: torch.tensor. Model parameter with size (p,p)
        coef: torch.tensor. Model parameter with size (d,p)
    Returns:
        torch.tensor of size 1 with a gradient.
    """
    n_samples, _ = counts.shape
    s_rond_s = torch.multiply(latent_var, latent_var)
    offsets_plus_m = offsets + latent_mean
    closed_coef = closed_formula_coef(covariates, latent_mean)
    closed_covariance = closed_formula_covariance(
        covariates, latent_mean, latent_var, closed_coef, n_samples
    )
    elbo = -n_samples / 2 * torch.logdet(closed_covariance)
    elbo += torch.sum(
        torch.multiply(counts, offsets_plus_m)
        - torch.exp(offsets_plus_m + s_rond_s / 2)
        + 1 / 2 * torch.log(s_rond_s)
    )
    elbo -= torch.sum(log_stirling(counts))
    return elbo / n_samples


def elbo_plnpca(counts, covariates, offsets, latent_mean, latent_var, components, coef):
    """
    Compute the ELBO (Evidence LOwer Bound) for the PLN model with a PCA
    parametrization. See the doc for more details on the computation.

    Args:
        counts: torch.tensor. Counts with size (n,p)
        0: torch.tensor. Offset, size (n,p)
        covariates: torch.tensor. Covariates, size (n,d)
        latent_mean: torch.tensor. Variational parameter with size (n,p)
        latent_var: torch.tensor. Variational parameter with size (n,p)
        components: torch.tensor. Model parameter with size (p,q)
        coef: torch.tensor. Model parameter with size (d,p)
    Returns:
        torch.tensor of size 1 with a gradient.
    """
    n_samples = counts.shape[0]
    rank = components.shape[1]
    log_intensity = (
        offsets + torch.mm(covariates, coef) + torch.mm(latent_mean, components.T)
    )
    s_rond_s = torch.multiply(latent_var, latent_var)
    counts_log_intensity = torch.sum(torch.multiply(counts, log_intensity))
    minus_intensity_plus_s_rond_s_cct = torch.sum(
        -torch.exp(
            log_intensity
            + 1 / 2 * torch.mm(s_rond_s, torch.multiply(components, components).T)
        )
    )
    minuslogs_rond_s = 1 / 2 * torch.sum(torch.log(s_rond_s))
    mm_plus_s_rond_s = torch.sum(
        -1
        / 2
        * (
            torch.multiply(latent_mean, latent_mean)
            + torch.multiply(latent_var, latent_var)
        )
    )
    log_stirlingcounts = torch.sum(log_stirling(counts))
    return (
        counts_log_intensity
        + minus_intensity_plus_s_rond_s_cct
        + minuslogs_rond_s
        + mm_plus_s_rond_s
        - log_stirlingcounts
        + n_samples * rank / 2
    ) / n_samples


## should rename some variables so that is is clearer when we see the formula
def elbo_zi_pln(
    counts,
    covariates,
    offsets,
    latent_mean,
    latent_var,
    pi,
    covariance,
    coef,
    _coef_inflation,
    dirac,
):
    """Compute the ELBO (Evidence LOwer Bound) for the Zero Inflated PLN model.
    See the doc for more details on the computation.

    Args:
        counts: torch.tensor. Counts with size (n,p)
        0: torch.tensor. Offset, size (n,p)
        covariates: torch.tensor. Covariates, size (n,d)
        latent_mean: torch.tensor. Variational parameter with size (n,p)
        latent_var: torch.tensor. Variational parameter with size (n,p)
        pi: torch.tensor. Variational parameter with size (n,p)
        covariance: torch.tensor. Model parameter with size (p,p)
        coef: torch.tensor. Model parameter with size (d,p)
        _coef_inflation: torch.tensor. Model parameter with size (d,p)
    Returns:
        torch.tensor of size 1 with a gradient.
    """
    if torch.norm(pi * dirac - pi) > 0.0001:
        print("Bug")
        return False
    n_samples = counts.shape[0]
    dim = counts.shape[1]
    s_rond_s = torch.multiply(latent_var, latent_var)
    offsets_plus_m = offsets + latent_mean
    m_minus_xb = latent_mean - torch.mm(covariates, coef)
    x_coef_inflation = torch.mm(covariates, _coef_inflation)
    elbo = torch.sum(
        torch.multiply(
            1 - pi,
            torch.multiply(counts, offsets_plus_m)
            - torch.exp(offsets_plus_m + s_rond_s / 2)
            - log_stirling(counts),
        )
        + pi
    )

    elbo -= torch.sum(
        torch.multiply(pi, trunc_log(pi)) + torch.multiply(1 - pi, trunc_log(1 - pi))
    )
    elbo += torch.sum(
        torch.multiply(pi, x_coef_inflation)
        - torch.log(1 + torch.exp(x_coef_inflation))
    )

    elbo -= (
        1
        / 2
        * torch.trace(
            torch.mm(
                torch.inverse(covariance),
                torch.diag(torch.sum(s_rond_s, dim=0))
                + torch.mm(m_minus_xb.T, m_minus_xb),
            )
        )
    )
    elbo += n_samples / 2 * torch.log(torch.det(covariance))
    elbo += n_samples * dim / 2
    elbo += torch.sum(1 / 2 * torch.log(s_rond_s))
    return elbo
