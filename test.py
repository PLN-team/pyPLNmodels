import pandas as pd
import numpy as np
from pyPLNmodels._utils import sample_PLN
from pyPLNmodels import PLNPCA
import torch
import time

n = 20
p = 140
q = 10
d = 1
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print("device :", device)

beta = torch.randn(d, p, device=device)
C = torch.randn(p, q, device=device) / p * 2
O = torch.ones((n, p), device=device) / 2
covariates = torch.ones((n, d), device=device)


def MSE(t):
    return torch.mean(t**2)


Y, _, _ = sample_PLN(C, beta, covariates, O)

print("max Y ", torch.max(Y))
pca = PLNPCA(q=10)
t0 = time.time()
pca.fit(Y, covariates, O, verbose=True, do_smart_init=True)
print("time smart init:", time.time() - t0)
print("diff beta smart ", MSE(pca.get_beta() - beta.cpu()))
