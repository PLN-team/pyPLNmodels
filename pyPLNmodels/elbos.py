import torch  # pylint:disable=[C0114]
from ._utils import _log_stirling, _trunc_log
from ._closed_forms import _closed_formula_covariance, _closed_formula_coef

from typing import Optional


def elbo_pln(
    endog: torch.Tensor,
    offsets: torch.Tensor,
    exog: Optional[torch.Tensor],
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
    if exog is None:
        XB = torch.zeros_like(endog)
    else:
        XB = exog @ coef
    m_minus_xb = latent_mean - XB
    d_plus_minus_xb2 = (
        torch.diag(torch.sum(s_rond_s, dim=0)) + m_minus_xb.T @ m_minus_xb
    )
    elbo = -0.5 * n_samples * torch.logdet(covariance)
    elbo += torch.sum(
        endog * offsets_plus_m
        - 0.5 * torch.exp(offsets_plus_m + s_rond_s)
        + 0.5 * torch.log(s_rond_s)
    )
    elbo -= 0.5 * torch.trace(torch.inverse(covariance) @ d_plus_minus_xb2)
    elbo -= torch.sum(_log_stirling(endog))
    elbo += 0.5 * n_samples * dim
    return elbo / n_samples


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


## pb with trunc_log
## should rename some variables so that is is clearer when we see the formula
def elbo_zi_pln(
    counts,
    covariates,
    offsets,
    latent_mean,
    latent_var,
    latent_prob,
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
    if torch.norm(latent_prob * dirac - latent_prob) > 0.00000001:
        print("Bug")
        raise RuntimeError("rho error")
    n_samples, dim = counts.shape
    s_rond_s = torch.multiply(latent_var, latent_var)
    o_plus_m = offsets + latent_mean
    m_minus_xb = latent_mean - torch.mm(covariates, coef)
    x_coef_inflation = torch.mm(covariates, _coef_inflation)

    # a = b = c = d = e = f = 0
    A = torch.exp(o_plus_m + s_rond_s / 2)
    inside_a = torch.multiply(
        1 - latent_prob, torch.multiply(counts, o_plus_m) - A - log_stirling(counts)
    )
    a = torch.sum(inside_a)

    Omega = torch.inverse(covariance)
    m_moins_xb_outer = torch.mm(m_minus_xb.T, m_minus_xb)
    un_moins_rho = 1 - latent_prob
    un_moins_rho_outer = torch.mm(un_moins_rho.T, un_moins_rho)
    inside_b = (
        -1
        / 2
        * torch.multiply(Omega, torch.multiply(un_moins_rho_outer, m_moins_xb_outer))
    )
    b = n_samples / 2 * torch.logdet(Omega) + torch.sum(inside_b)

    inside_c = torch.multiply(latent_prob, x_coef_inflation) - torch.log(
        1 + torch.exp(x_coef_inflation)
    )
    c = torch.sum(inside_c)

    log_diag = torch.log(torch.diag(covariance))
    Diag_log_diag = torch.diag_embed(log_diag)
    inside_d = torch.multiply(
        1 - latent_prob, torch.log(torch.abs(latent_var))
    ) + torch.matmul(latent_prob / 2, Diag_log_diag)
    d = n_samples * dim / 2 + torch.sum(inside_d)

    inside_e = torch.multiply(latent_prob, trunc_log(latent_prob)) + torch.multiply(
        1 - latent_prob, trunc_log(1 - latent_prob)
    )
    e = -torch.sum(inside_e)

    sum_un_moins_rho_s2 = torch.sum(torch.multiply(1 - latent_prob, s_rond_s), axis=0)
    diag_sig_sum_rho = torch.multiply(
        torch.diag(covariance), torch.sum(latent_prob, axis=0)
    )
    inside_f = torch.multiply(torch.diag(Omega), sum_un_moins_rho_s2 + diag_sig_sum_rho)
    f = -1 / 2 * torch.sum(inside_f)

    # f = 0
    # elbo = torch.sum(
    #     torch.multiply(
    #         1 - latent_prob,
    #         torch.multiply(counts, o_plus_m)
    #         - torch.exp(o_plus_m + s_rond_s / 2)
    #         - log_stirling(counts),
    #     )
    #     + latent_prob
    # )
    # # print('O:', O)
    # elbo -= torch.sum(
    #     torch.multiply(latent_prob, trunc_log(latent_prob)) + torch.multiply(1 - latent_prob, trunc_log(1 - latent_prob))
    # )
    # elbo -= (
    #     1
    #     / 2
    #     * torch.trace(
    #         torch.mm(
    #             torch.inverse(covariance),
    #             torch.diag(torch.sum(s_rond_s, dim=0))
    #             + torch.mm(m_minus_xb.T, m_minus_xb),
    #         )
    #     )
    # )
    # elbo += n_samples / 2 * torch.log(torch.det(covariance))
    # elbo += n_samples * dim / 2
    # elbo += torch.sum(1 / 2 * torch.log(s_rond_s))
    return a + b + c + d + e + f
