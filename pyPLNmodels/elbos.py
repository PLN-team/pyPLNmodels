import torch
from ._utils import log_stirling, trunc_log
from ._closed_forms import closed_formula_covariance, closed_formula_coef


def ELBOPLN(counts, covariates, offsets, latent_mean, latent_var, covariance, coef):
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
    SrondS = torch.multiply(latent_var, latent_var)
    offsetsplusM = offsets + latent_mean
    m_moins_xb = latent_mean - torch.mm(covariates, coef)
    elbo = -n_samples / 2 * torch.logdet(covariance)
    elbo += torch.sum(
        torch.multiply(counts, offsetsplusM)
        - torch.exp(offsetsplusM + SrondS / 2)
        + 1 / 2 * torch.log(SrondS)
    )
    Dplusm_moins_xb2 = torch.diag(torch.sum(SrondS, dim=0)) + torch.mm(
        m_moins_xb.T, m_moins_xb
    )
    moinspsur2n = (
        1 / 2 * torch.trace(torch.mm(torch.inverse(covariance), Dplusm_moins_xb2))
    )
    elbo -= 1 / 2 * torch.trace(torch.mm(torch.inverse(covariance), Dplusm_moins_xb2))
    elbo -= torch.sum(log_stirling(counts))
    elbo += n_samples * dim / 2
    return elbo


def profiledELBOPLN(counts, covariates, offsets, latent_mean, latent_var):
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
    n_samples, dim = counts.shape
    SrondS = torch.multiply(latent_var, latent_var)
    offsetsplusM = offsets + latent_mean
    closed_coef = closed_formula_coef(covariates, latent_mean)
    closed_covariance = closed_formula_covariance(
        covariates, latent_mean, latent_var, closed_coef, n_samples
    )
    elbo = -n_samples / 2 * torch.logdet(closed_covariance)
    elbo += torch.sum(
        torch.multiply(counts, offsetsplusM)
        - torch.exp(offsetsplusM + SrondS / 2)
        + 1 / 2 * torch.log(SrondS)
    )
    elbo -= torch.sum(log_stirling(counts))
    return elbo


def ELBOPLNPCA(counts, covariates, offsets, latent_mean, latent_var, components, coef):
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
    A = offsets + torch.mm(covariates, coef) + torch.mm(latent_mean, components.T)
    SrondS = torch.multiply(latent_var, latent_var)
    countsA = torch.sum(torch.multiply(counts, A))
    moinsexpAplusSrondSCCT = torch.sum(
        -torch.exp(
            A + 1 / 2 * torch.mm(SrondS, torch.multiply(components, components).T)
        )
    )
    moinslogSrondS = 1 / 2 * torch.sum(torch.log(SrondS))
    MMplusSrondS = torch.sum(
        -1
        / 2
        * (
            torch.multiply(latent_mean, latent_mean)
            + torch.multiply(latent_var, latent_var)
        )
    )
    log_stirlingcounts = torch.sum(log_stirling(counts))
    return (
        countsA
        + moinsexpAplusSrondSCCT
        + moinslogSrondS
        + MMplusSrondS
        - log_stirlingcounts
        + n_samples * rank / 2
    )


## should rename some variables so that is is clearer when we see the formula
def ELBOZIPLN(
    counts,
    covariates,
    offsets,
    latent_mean,
    latent_var,
    pi,
    covariance,
    coef,
    coef_inflation,
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
        coef_inflation: torch.tensor. Model parameter with size (d,p)
    Returns:
        torch.tensor of size 1 with a gradient.
    """
    if torch.norm(pi * dirac - pi) > 0.0001:
        print("Bug")
        return False
    n_samples = counts.shape[0]
    p = counts.shape[1]
    SrondS = torch.multiply(latent_var, latent_var)
    offsetsplusM = offsets + latent_mean
    m_moins_xb = latent_mean - torch.mm(covariates, coef)
    Xcoef_inflation = torch.mm(covariates, coef_inflation)
    elbo = torch.sum(
        torch.multiply(
            1 - pi,
            torch.multiply(counts, offsetsplusM)
            - torch.exp(offsetsplusM + SrondS / 2)
            - log_stirling(counts),
        )
        + pi
    )

    elbo -= torch.sum(
        torch.multiply(pi, trunc_log(pi)) + torch.multiply(1 - pi, trunc_log(1 - pi))
    )
    elbo += torch.sum(
        torch.multiply(pi, Xcoef_inflation) - torch.log(1 + torch.exp(Xcoef_inflation))
    )

    elbo -= (
        1
        / 2
        * torch.trace(
            torch.mm(
                torch.inverse(covariance),
                torch.diag(torch.sum(SrondS, dim=0))
                + torch.mm(m_moins_xb.T, m_moins_xb),
            )
        )
    )
    elbo += n_samples / 2 * torch.log(torch.det(covariance))
    elbo += n_samples * p / 2
    elbo += torch.sum(1 / 2 * torch.log(SrondS))
    return elbo
