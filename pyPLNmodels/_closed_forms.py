import torch  # pylint:disable=[C0114]


def closed_formula_covariance(covariates, latent_mean, latent_var, coef, n_samples):
    """Closed form for covariance for the M step for the noPCA model."""
    m_moins_xb = latent_mean - torch.mm(covariates, coef)
    closed = torch.mm(m_moins_xb.T, m_moins_xb)
    closed += torch.diag(torch.sum(torch.multiply(latent_var, latent_var), dim=0))
    return 1 / (n_samples) * closed


def zi_closed_formula_covariance(
    covariates, latent_mean, latent_var, coef, latent_prob, n_samples
):
    """Closed form for covariance for the M step for the noPCA model."""
    m_moins_xb = latent_mean - torch.mm(covariates, coef)
    un_moins_rho = 1 - latent_prob
    un_moins_rho_outer = torch.mm(un_moins_rho.T, un_moins_rho)
    closed = torch.mm(m_moins_xb.T, m_moins_xb)
    closed = torch.multiply(closed, un_moins_rho_outer)
    closed += torch.diag(
        torch.sum(
            torch.multiply(un_moins_rho, torch.multiply(latent_var, latent_var)), dim=0
        )
    )
    return 1 / (n_samples) * closed


def closed_formula_coef(covariates, latent_mean):
    """Closed form for coef for the M step for the noPCA model."""
    return torch.mm(
        torch.mm(torch.inverse(torch.mm(covariates.T, covariates)), covariates.T),
        latent_mean,
    )


def closed_formula_latent_prob(
    offsets, latent_mean, latent_var, dirac, covariates, _coef_inflation
):
    poiss_param = torch.exp(
        offsets + latent_mean + torch.multiply(latent_var, latent_var) / 2
    )
    return torch.multiply(
        torch.sigmoid(poiss_param + torch.mm(covariates, _coef_inflation)), dirac
    )
