import torch 
from utils import log_stirling


def ELBOnoPCA(Y, O, covariates, M, S, Sigma, beta):
    '''Compute the ELBO (Evidence LOwer Bound. See the doc for more details
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
        torch.tensor of size 1 with a gradient. The ELBO.
    '''
    n, p = Y.shape
    SrondS = torch.multiply(S, S)
    OplusM = O + M
    MmoinsXB = M - torch.mm(covariates, beta)

    elbo = - n / 2 * torch.logdet(Sigma)
    elbo += torch.sum(torch.multiply(Y, OplusM)
                     - torch.exp(OplusM + SrondS / 2)
                     + 1 / 2 * torch.log(SrondS)
                     )
    DplusMmoinsXB2 = torch.diag(
        torch.sum(SrondS, dim=0)) + torch.mm(MmoinsXB.T, MmoinsXB)
    elbo -= 1 / 2 * torch.trace(
        torch.mm(
            torch.inverse(Sigma),
            DplusMmoinsXB2
        )
    )
    elbo -= torch.sum(log_stirling(Y))
    elbo += n * p / 2
    return elbo


def ELBOPCA(Y, O, covariates, M, S, C, beta):
    '''compute the ELBO with a PCA parametrization'''
    n = Y.shape[0]
    q = C.shape[1]
    A = O + torch.mm(covariates, beta) + torch.mm(M, C.T)
    SrondS = torch.multiply(S, S)
    YA = torch.sum(torch.multiply(Y, A))
    moinsexpAplusSrondSCCT = torch.sum(-torch.exp(A + 1 / 2 *
                                                  torch.mm(SrondS, torch.multiply(C, C).T)))
    moinslogSrondS = 1 / 2 * torch.sum(torch.log(SrondS))
    MMplusSrondS = torch.sum(-1 / 2 *
                             (torch.multiply(M, M) + torch.multiply(S, S)))
    log_stirlingY = torch.sum(log_stirling(Y))
    return YA + moinsexpAplusSrondSCCT + moinslogSrondS + MMplusSrondS - log_stirlingY + n * q / 2

