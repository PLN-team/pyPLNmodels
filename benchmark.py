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


def append_running_times_pln_plnpca(Y):
    pln = Pln(Y)
    plnpca = PlnPCA(Y, rank=rank)
    pln.fit(tol=sharp_tol)
    plnpca.fit(tol=sharp_tol)

    rough_running_times_pln = pln._plotargs.running_times[
        next(i for i, v in enumerate(pln._plotargs.criterions) if v < rough_tol)
    ]
    rough_running_times_plnpca = plnpca._plotargs.running_times[
        next(i for i, v in enumerate(plnpca._plotargs.criterions) if v < rough_tol)
    ]

    pln_running_times_sharp_conv.append(pln._plotargs.running_times[-1])
    plnpca_running_times_sharp_conv.append(plnpca._plotargs.running_times[-1])
    pln_running_times_rough_conv.append(rough_running_times_pln)
    plnpca_running_times_rough_conv.append(rough_running_times_plnpca)


n = 50
ps = range(50, 1000, 10)
pln_running_times_sharp_conv = []
plnpca_running_times_sharp_conv = []
pln_running_times_rough_conv = []
plnpca_running_times_rough_conv = []
sharp_tol = 0.001
rough_tol = 0.01
if sharp_tol > rough_tol:
    raise ValueError("tols in the wrong order")

rank = 30

for p in ps:
    Y, _, _ = get_sc_mark_data(max_n=n, dim=p)
    append_running_times_pln_plnpca(Y)


plt.plot(ps, pln_running_times_sharp_conv, color="blue", label="Pln sharp")
plt.plot(ps, plnpca_running_times_sharp_conv, color="orange", label="PlnPCA sharp")
plt.plot(
    ps,
    pln_running_times_rough_conv,
    color="blue",
    label="Pln rough",
    linestyle="dotted",
)
plt.plot(
    ps,
    plnpca_running_times_rough_conv,
    color="orange",
    label="PlnPCA rough",
    linestyle="dotted",
)
plt.legend()
plt.show()
