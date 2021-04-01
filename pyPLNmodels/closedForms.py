import torch


def closedSigma(covariates, M,S,beta,n):
    '''Closed form for Sigma for the M step for the noPCA model.'''
    MmoinsXB = M - torch.mm(covariates, beta)
    closed = torch.mm(MmoinsXB.T, MmoinsXB)
    closed += torch.diag(torch.sum(torch.multiply(S, S), dim=0))
    return 1 / (n) * closed

def closedBeta(covariates,M):
    '''Closed form for beta for the M step for the noPCA model.'''
    return torch.mm(
        torch.mm(
            torch.inverse(torch.mm(
                covariates.T,
                covariates)),
            covariates.T),
        M)
    
