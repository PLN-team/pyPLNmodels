import torch
from ._utils import log_stirling, trunc_log
from ._closed_forms import closed_formula_Sigma, closed_formula_beta


def ELBOPLN(Y, covariates, O, M, S, Sigma, beta):
    """
    Compute the ELBO (Evidence LOwer Bound) for the PLN model. See the doc for more details
    on the computation.

    Args:
        Y: torch.tensor. Counts with size (n,p)
        0: torch.tensor. Offset, size (n,p)
        covariates: torch.tensor. Covariates, size (n,d)
        M: torch.tensor. Variational parameter with size (n,p)
        S: torch.tensor. Variational parameter with size (n,p)
        Sigma: torch.tensor. Model parameter with size (p,p)
        beta: torch.tensor. Model parameter with size (d,p)
    Returns:
        torch.tensor of size 1 with a gradient.
    """
    n, p = Y.shape
    SrondS = torch.multiply(S, S)
    OplusM = O + M
    MmoinsXB = M - torch.mm(covariates, beta)
    elbo = -n / 2 * torch.logdet(Sigma)
    elbo += torch.sum(
        torch.multiply(Y, OplusM)
        - torch.exp(OplusM + SrondS / 2)
        + 1 / 2 * torch.log(SrondS)
    )
    DplusMmoinsXB2 = torch.diag(torch.sum(SrondS, dim=0)) + torch.mm(
        MmoinsXB.T, MmoinsXB
    )
    moinspsur2n = 1 / 2 * torch.trace(torch.mm(torch.inverse(Sigma), DplusMmoinsXB2))
    elbo -= 1 / 2 * torch.trace(torch.mm(torch.inverse(Sigma), DplusMmoinsXB2))
    elbo -= torch.sum(log_stirling(Y))
    elbo += n * p / 2
    return elbo


def profiledELBOPLN(Y, covariates, O, M, S):
    """
    Compute the ELBO (Evidence LOwer Bound) for the PLN model. We use the fact that Sigma and beta are
    completely determined by M,S, and the covariates. See the doc for more details
    on the computation.

    Args:
        Y: torch.tensor. Counts with size (n,p)
        0: torch.tensor. Offset, size (n,p)
        covariates: torch.tensor. Covariates, size (n,d)
        M: torch.tensor. Variational parameter with size (n,p)
        S: torch.tensor. Variational parameter with size (n,p)
        Sigma: torch.tensor. Model parameter with size (p,p)
        beta: torch.tensor. Model parameter with size (d,p)
    Returns:
        torch.tensor of size 1 with a gradient.
    """
    n, p = Y.shape
    SrondS = torch.multiply(S, S)
    OplusM = O + M
    closed_beta = closed_formula_beta(covariates, M)
    closed_Sigma = closed_formula_Sigma(covariates, M, S, closed_beta, n)
    elbo = -n / 2 * torch.logdet(closed_Sigma)
    elbo += torch.sum(
        torch.multiply(Y, OplusM)
        - torch.exp(OplusM + SrondS / 2)
        + 1 / 2 * torch.log(SrondS)
    )
    elbo -= torch.sum(log_stirling(Y))
    return elbo


def ELBOPLNPCA(Y, covariates, O, M, S, C, beta):
    """
    Compute the ELBO (Evidence LOwer Bound) for the PLN model with a PCA
    parametrization. See the doc for more details on the computation.

    Args:
        Y: torch.tensor. Counts with size (n,p)
        0: torch.tensor. Offset, size (n,p)
        covariates: torch.tensor. Covariates, size (n,d)
        M: torch.tensor. Variational parameter with size (n,p)
        S: torch.tensor. Variational parameter with size (n,p)
        C: torch.tensor. Model parameter with size (p,q)
        beta: torch.tensor. Model parameter with size (d,p)
    Returns:
        torch.tensor of size 1 with a gradient.
    """
    n = Y.shape[0]
    q = C.shape[1]
    A = O + torch.mm(covariates, beta) + torch.mm(M, C.T)
    SrondS = torch.multiply(S, S)
    YA = torch.sum(torch.multiply(Y, A))
    moinsexpAplusSrondSCCT = torch.sum(
        -torch.exp(A + 1 / 2 * torch.mm(SrondS, torch.multiply(C, C).T))
    )
    moinslogSrondS = 1 / 2 * torch.sum(torch.log(SrondS))
    MMplusSrondS = torch.sum(-1 / 2 * (torch.multiply(M, M) + torch.multiply(S, S)))
    log_stirlingY = torch.sum(log_stirling(Y))
    return (
        YA
        + moinsexpAplusSrondSCCT
        + moinslogSrondS
        + MMplusSrondS
        - log_stirlingY
        + n * q / 2
    )


## should rename some variables so that is is clearer when we see the formula
def ELBOZIPLN(Y, covariates, O, M, S, pi, Sigma, beta, B_zero, dirac):
    """Compute the ELBO (Evidence LOwer Bound) for the Zero Inflated PLN model.
    See the doc for more details on the computation.

    Args:
        Y: torch.tensor. Counts with size (n,p)
        0: torch.tensor. Offset, size (n,p)
        covariates: torch.tensor. Covariates, size (n,d)
        M: torch.tensor. Variational parameter with size (n,p)
        S: torch.tensor. Variational parameter with size (n,p)
        pi: torch.tensor. Variational parameter with size (n,p)
        Sigma: torch.tensor. Model parameter with size (p,p)
        beta: torch.tensor. Model parameter with size (d,p)
        B_zero: torch.tensor. Model parameter with size (d,p)
    Returns:
        torch.tensor of size 1 with a gradient.
    """
    if torch.norm(pi * dirac - pi) > 0.0001:
        print("Bug")
        return False
    n = Y.shape[0]
    p = Y.shape[1]
    SrondS = torch.multiply(S, S)
    OplusM = O + M
    MmoinsXB = M - torch.mm(covariates, beta)
    XB_zero = torch.mm(covariates, B_zero)
    elbo = torch.sum(
        torch.multiply(
            1 - pi,
            torch.multiply(Y, OplusM)
            - torch.exp(OplusM + SrondS / 2)
            - log_stirling(Y),
        )
        + pi
    )

    elbo -= torch.sum(
        torch.multiply(pi, trunc_log(pi)) + torch.multiply(1 - pi, trunc_log(1 - pi))
    )
    elbo += torch.sum(torch.multiply(pi, XB_zero) - torch.log(1 + torch.exp(XB_zero)))

    elbo -= (
        1
        / 2
        * torch.trace(
            torch.mm(
                torch.inverse(Sigma),
                torch.diag(torch.sum(SrondS, dim=0)) + torch.mm(MmoinsXB.T, MmoinsXB),
            )
        )
    )
    elbo += n / 2 * torch.log(torch.det(Sigma))
    elbo += n * p / 2
    elbo += torch.sum(1 / 2 * torch.log(SrondS))
    return elbo
