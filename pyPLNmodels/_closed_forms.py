import torch


def closed_formula_Sigma(covariates, M, S, beta, n):
    """Closed form for Sigma for the M step for the noPCA model."""
    MmoinsXB = M - torch.mm(covariates, beta)
    closed = torch.mm(MmoinsXB.T, MmoinsXB)
    closed += torch.diag(torch.sum(torch.multiply(S, S), dim=0))
    return 1 / (n) * closed


def closed_formula_beta(covariates, M):
    """Closed form for beta for the M step for the noPCA model."""
    return torch.mm(
        torch.mm(torch.inverse(torch.mm(covariates.T, covariates)), covariates.T), M
    )


def closed_formula_pi(O, M, S, dirac, covariates, Theta_zero):
    A = torch.exp(O + M + torch.multiply(S, S) / 2)
    return torch.multiply(torch.sigmoid(A + torch.mm(covariates, Theta_zero)), dirac)
