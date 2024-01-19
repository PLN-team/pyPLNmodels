import torch  # pylint:disable=[C0114]
from ._utils import _log_stirling, _trunc_log, _log1pexp
from ._closed_forms import _closed_formula_covariance, _closed_formula_coef

from typing import Optional


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
        Offset with size (n, p).
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
    closed_covariance = _closed_formula_covariance(
        exog, latent_mean, latent_sqrt_var, closed_coef, n_samples
    )
    elbo = -0.5 * n_samples * torch.logdet(closed_covariance)
    elbo += torch.sum(
        endog * offsets_plus_mean
        - torch.exp(offsets_plus_mean + s_squared / 2)
        + 0.5 * torch.log(s_squared)
    )
    elbo -= torch.sum(_log_stirling(endog))
    return elbo / n_samples


def elbo_plnpca(
    endog: torch.Tensor,
    exog: torch.Tensor,
    offsets: torch.Tensor,
    latent_mean: torch.Tensor,
    latent_sqrt_var: torch.Tensor,
    components: torch.Tensor,
    coef: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the ELBO (Evidence Lower Bound) for the Pln model
    with PCA parametrization.

    Parameters:
    ----------
    endog : torch.Tensor
        Counts with size (n, p).
    exog : torch.Tensor
        Covariates with size (n, d).
    offsets : torch.Tensor
        Offset with size (n, p).
    latent_mean : torch.Tensor
        Variational parameter with size (n, p).
    latent_sqrt_var : torch.Tensor
        Variational parameter with size (n, p). More precisely it is the unsigned
        square root of the variational variance.
    components : torch.Tensor
        Model parameter with size (p, q).
    coef : torch.Tensor
        Model parameter with size (d, p).

    Returns:
    -------
    torch.Tensor
        The ELBO (Evidence Lower Bound) with size 1, with a gradient.
    """
    n_samples = endog.shape[0]
    rank = components.shape[1]
    if exog is None:
        XB = 0
    else:
        XB = exog @ coef
    log_intensity = offsets + XB + latent_mean @ components.T
    s_squared = torch.square(latent_sqrt_var)
    endog_log_intensity = torch.sum(endog * log_intensity)
    minus_intensity_plus_s_squared_cct = torch.sum(
        -torch.exp(log_intensity + 0.5 * s_squared @ (components * components).T)
    )
    minus_logs_squared = 0.5 * torch.sum(torch.log(s_squared))
    mm_plus_s_squared = -0.5 * torch.sum(
        torch.square(latent_mean) + torch.square(latent_sqrt_var)
    )
    log_stirling_endog = torch.sum(_log_stirling(endog))
    return (
        endog_log_intensity
        + minus_intensity_plus_s_squared_cct
        + minus_logs_squared
        + mm_plus_s_squared
        - log_stirling_endog
        + 0.5 * n_samples * rank
    ) / n_samples


def log1pexp(x):
    # more stable version of log(1 + exp(x))
    return torch.where(x < 50, torch.log1p(torch.exp(x)), x)


def elbo_pln(
    endog: torch.Tensor,
    exog: Optional[torch.Tensor],
    offsets: torch.Tensor,
    latent_mean: torch.Tensor,
    latent_sqrt_var: torch.Tensor,
    covariance: torch.Tensor,
    coef: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the ELBO (Evidence Lower Bound) for the Pln model.

    Parameters:
    ----------
    endog : torch.Tensor
        Counts with size (n, p).
    offsets : torch.Tensor
        Offset with size (n, p).
    exog : torch.Tensor, optional
        Covariates with size (n, d).
    latent_mean : torch.Tensor
        Variational parameter with size (n, p).
    latent_sqrt_var : torch.Tensor
        Variational parameter with size (n, p).
    covariance : torch.Tensor
        Model parameter with size (p, p).
    coef : torch.Tensor
        Model parameter with size (d, p).

    Returns:
    -------
    torch.Tensor
        The ELBO (Evidence Lower Bound), of size one.
    """
    n_samples, dim = endog.shape
    s_rond_s = torch.square(latent_sqrt_var)
    offsets_plus_m = offsets + latent_mean
    Omega = torch.inverse(covariance)
    if exog is None:
        XB = torch.zeros_like(endog)
    else:
        XB = exog @ coef
    # print('XB:', XB)
    m_minus_xb = latent_mean - XB
    m_moins_xb_outer = torch.mm(m_minus_xb.T, m_minus_xb)
    A = torch.exp(offsets_plus_m + s_rond_s / 2)
    first_a = torch.sum(endog * offsets_plus_m)
    sec_a = -torch.sum(A)
    third_a = -torch.sum(_log_stirling(endog))
    a = first_a + sec_a + third_a
    diag = torch.diag(torch.sum(s_rond_s, dim=0))
    elbo = torch.clone(a)
    b = -0.5 * n_samples * torch.logdet(covariance) + torch.sum(
        -1 / 2 * Omega * m_moins_xb_outer
    )
    elbo += b
    d = n_samples * dim / 2 + torch.sum(+0.5 * torch.log(s_rond_s))
    elbo += d
    f = -0.5 * torch.trace(torch.inverse(covariance) @ diag)
    elbo += f
    return elbo


## pb with trunc_log
## should rename some variables so that is is clearer when we see the formula
def elbo_zi_pln(
    endog,
    exog,
    offsets,
    latent_mean,
    latent_sqrt_var,
    latent_prob,
    components,
    coef,
    coef_inflation,
    dirac,
):
    """Compute the ELBO (Evidence LOwer Bound) for the Zero Inflated PLN model.
    See the doc for more details on the computation.

    Args:
        endog: torch.tensor. Counts with size (n,p)
        0: torch.tensor. Offset, size (n,p)
        exog: torch.tensor. Covariates, size (n,d)
        latent_mean: torch.tensor. Variational parameter with size (n,p)
        latent_var: torch.tensor. Variational parameter with size (n,p)
        pi: torch.tensor. Variational parameter with size (n,p)
        covariance: torch.tensor. Model parameter with size (p,p)
        coef: torch.tensor. Model parameter with size (d,p)
        coef_inflation: torch.tensor. Model parameter with size (d,p)
    Returns:
        torch.tensor of size 1 with a gradient.
    """
    covariance = components @ (components.T)
    if torch.norm(latent_prob * dirac - latent_prob) > 0.00000001:
        raise RuntimeError("Latent probability error.")
    n_samples, dim = endog.shape
    s_rond_s = torch.multiply(latent_sqrt_var, latent_sqrt_var)
    o_plus_m = offsets + latent_mean
    if exog is None:
        XB = torch.zeros_like(endog)
        x_coef_inflation = torch.zeros_like(endog)
    else:
        XB = exog @ coef
        x_coef_inflation = exog @ coef_inflation

    m_minus_xb = latent_mean - XB

    A = torch.exp(o_plus_m + s_rond_s / 2)
    inside_a = torch.multiply(
        1 - latent_prob, torch.multiply(endog, o_plus_m) - A - _log_stirling(endog)
    )
    Omega = torch.inverse(covariance)
    m_moins_xb_outer = torch.mm(m_minus_xb.T, m_minus_xb)
    un_moins_rho = 1 - latent_prob
    un_moins_rho_m_moins_xb = un_moins_rho * m_minus_xb
    un_moins_rho_m_moins_xb_outer = un_moins_rho_m_moins_xb.T @ un_moins_rho_m_moins_xb
    inside_b = -1 / 2 * Omega * un_moins_rho_m_moins_xb_outer
    inside_c = torch.multiply(latent_prob, x_coef_inflation) - _log1pexp(
        x_coef_inflation
    )
    log_diag = torch.log(torch.diag(covariance))
    log_S_term = torch.sum(
        torch.multiply(1 - latent_prob, torch.log(torch.abs(latent_sqrt_var))), axis=0
    )
    y = torch.sum(latent_prob, axis=0)
    covariance_term = 1 / 2 * torch.log(torch.diag(covariance)) * y
    inside_d = covariance_term + log_S_term

    inside_e = -torch.multiply(latent_prob, _trunc_log(latent_prob)) - torch.multiply(
        1 - latent_prob, _trunc_log(1 - latent_prob)
    )
    sum_un_moins_rho_s2 = torch.sum(torch.multiply(1 - latent_prob, s_rond_s), axis=0)
    diag_sig_sum_rho = torch.multiply(
        torch.diag(covariance), torch.sum(latent_prob, axis=0)
    )
    new = torch.sum(latent_prob * un_moins_rho * (m_minus_xb**2), axis=0)
    K = sum_un_moins_rho_s2 + diag_sig_sum_rho + new
    inside_f = -1 / 2 * torch.diag(Omega) * K
    first = torch.sum(inside_a + inside_c + inside_e)
    second = torch.sum(inside_b)
    _, logdet = torch.slogdet(components)
    second -= n_samples * logdet
    third = torch.sum(inside_d + inside_f)
    third += n_samples * dim / 2
    res = first + second + third
    return res
