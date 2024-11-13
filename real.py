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
    genes = data.var["gene"]
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
    genes = genes[best_var].values.__array__()
    Y = Y[:, best_var]
    return Y, GT, list(GT_name.values.__array__()), genes


n = 500
p = 300
nb_max_iter = 800
max_class = 3
Y, GT, GT_name, genes_name = get_sc_mark_data(
    max_class=max_class, max_n=n, dim=p, seed=0
)
ohe = OneHotEncoder()
print("GT", GT_name)
GT_onehot = ohe.fit_transform(GT[:, None]).toarray()
print("onehot", GT_onehot)
groups = np.array([[i] * p for i in range(max_class)])
print("groups", groups)

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
print("coef", pln._coef)
print("flatten", pln._coef.flatten())
groups = np.array([[i] * p for i in range(max_class)])
dimensions = [i for i in range(p)] * max_class
print("dimensions", len(dimensions))
genes_name_repeat = []
for _ in range(max_class):
    genes_name_repeat = genes_name_repeat + genes_name.tolist()
var_theta = fisher_pln.get_var_theta(pln._coef) * 1.96

df = pd.DataFrame(
    {
        "coef": pln._coef.detach().cpu().flatten(),
        "ll": -var_theta,
        "hh": var_theta,
        "groups": groups.flatten(),
        "dimensions": dimensions,
        "genes": genes_name_repeat,
    }
)
df.to_csv(f"csv_ic/real_ic_n_{n}.csv")
