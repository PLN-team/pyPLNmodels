from sandwich import Fisher_Pln
import scanpy
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from pyPLNmodels import Pln
import pandas as pd


def get_sc_mark_data(max_class=28, max_n=200, dim=100, seed=0):
    data = scanpy.read_h5ad("/home/bastien/code/data/2k_cell_per_study_10studies.h5ad")
    # data = scanpy.read_h5ad("/home/bastien/Downloads/2k_cell_per_study_10studies.h5ad")
    Y = data.X.toarray()
    GT_name = data.obs["standard_true_celltype_v5"]
    le = LabelEncoder()
    GT = le.fit_transform(GT_name)
    nb_per_class = np.bincount(GT)
    class_took = nb_per_class.argsort()[::-1][:max_class]
    filter_class = np.isin(GT, class_took)
    GT = GT[filter_class]
    Y = Y[filter_class]
    GT_name = GT_name[filter_class]
    np.random.seed(seed)
    samples_chosen = np.random.choice(Y.shape[0], max_n)
    Y = Y[samples_chosen]
    GT = GT[samples_chosen]
    GT_name = GT_name[samples_chosen]
    GT = le.fit_transform(GT)
    random_indices = np.random.choice(Y.shape[1], dim, replace=False)
    best_var = np.var(Y, axis=0).argsort()[::-1][:dim]
    Y = Y[:, best_var]
    return Y, GT, list(GT_name.values.__array__())


n = 1000
p = 30
nb_max_iter = 800
max_class = 2
Y, GT, GT_name = get_sc_mark_data(max_class=max_class, max_n=n, dim=p, seed=0)
ohe = OneHotEncoder()
GT_onehot = ohe.fit_transform(GT[:, None]).toarray()

pln = Pln(Y, exog=GT_onehot, add_const=False)
pln.fit(nb_max_epoch=nb_max_iter)
A = torch.exp(pln.offsets + pln.latent_mean + 0.5 * pln.latent_sqrt_var**2)
too_low = pln._coef < -1
other = torch.clone(pln._coef)
fisher_pln = Fisher_Pln(
    Y=pln.endog,
    A=A,
    X=pln.exog,
    d=pln.nb_cov,
    p=pln.dim,
    n=pln.n_samples,
    S=pln.latent_sqrt_var,
    Omega=torch.inverse(pln.covariance),
    Sigma=pln.covariance,
)

var_theta = fisher_pln.get_var_theta(pln._coef) * 1.96

df = pd.DataFrame(
    {"coef": pln._coef.detach().cpu().flatten(), "ll": -var_theta, "hh": var_theta}
)
df.to_csv(f"csv_ic/real_ic_n_{n}.csv")
