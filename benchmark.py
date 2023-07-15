from pyPLNmodels import Pln, PlnPCA
import scanpy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def get_sc_mark_data(max_class=28, max_n=200, dim=100):
    data = scanpy.read_h5ad("2k_cell_per_study_10studies.h5ad")
    Y = data.X.toarray()[:max_n]
    GT_name = data.obs["standard_true_celltype_v5"][:max_n]
    le = LabelEncoder()
    GT = le.fit_transform(GT_name)
    filter = GT < max_class
    GT = GT[filter]
    GT_name = GT_name[filter]
    Y = Y[filter]
    GT = le.fit_transform(GT)
    not_only_zeros = np.sum(Y, axis=0) > 0
    Y = Y[:, not_only_zeros]
    var = np.var(Y, axis=0)
    most_variables = np.argsort(var)[-dim:]
    Y = Y[:, most_variables]
    return Y, GT, list(GT_name.values.__array__())


n = 500
ps = [200, 400]
pln_running_times = []
plnpca_running_times = []
tol = 0.01
rank = 30

for p in ps:
    Y, _, _ = get_sc_mark_data(max_n=n, dim=p)
    pln = Pln(Y)
    plnpca = PlnPCA(Y, rank=rank)
    pln.fit(tol=tol)
    plnpca.fit(tol=tol)
    pln_running_times.append(pln._plotargs.running_times[-1])
    plnpca_running_times.append(plnpca._plotargs.running_times[-1])

print("pln_running_times", pln_running_times)
print("plnpca_running_times", plnpca_running_times)
