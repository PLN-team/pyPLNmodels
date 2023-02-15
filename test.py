import time

import torch
from pyPLNmodels._utils import sample_PLN
from pyPLNmodels import PLNPCA

n = 20
p = 140
q = 10
d = 1
if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"
print("DEVICE :", DEVICE)

beta = torch.randn(d, p, device=DEVICE)
C = torch.randn(p, q, device=DEVICE) / p * 2
O = torch.ones((n, p), device=DEVICE) / 2
covariates = torch.ones((n, d), device=DEVICE)


def mse(tensor):
    return torch.mean(tensor**2)


Y, _, _ = sample_PLN(C, beta, covariates, O)

print("max Y ", torch.max(Y))
pca = PLNPCA(ranks=10)
pca.fit(Y)
print(pca.list_PLNPCA[10].model_parameters)
