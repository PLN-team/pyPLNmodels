import torch  # pylint:disable=[C0114]


def _closed_formula_covariance(covariates, latent_mean, latent_var, coef, n_samples):
    """Closed form for covariance for the M step for the noPCA model."""
    if covariates is None:
        XB = 0
    else:
        XB = covariates @ coef
    m_moins_xb = latent_mean - XB
    closed = m_moins_xb.T @ m_moins_xb + torch.diag(
        torch.sum(torch.square(latent_var), dim=0)
    )
    return closed / n_samples


def _closed_formula_coef(covariates, latent_mean):
    """Closed form for coef for the M step for the noPCA model."""
    if covariates is None:
        return None
    return torch.inverse(covariates.T @ covariates) @ covariates.T @ latent_mean


def _closed_formula_pi(
    offsets, latent_mean, latent_var, dirac, covariates, _coef_inflation
):
    poiss_param = torch.exp(offsets + latent_mean + 0.5 * torch.square(latent_var))
    return torch._sigmoid(poiss_param + torch.mm(covariates, _coef_inflation)) * dirac
