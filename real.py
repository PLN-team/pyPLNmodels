from sandwich import Fisher_Pln
import scanpy
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from pyPLNmodels import Pln


def get_sc_mark_data(max_class=28, max_n=200, dim=100, seed=0):
    # data = scanpy.read_h5ad("/home/bastien/code/data/2k_cell_per_study_10studies.h5ad")
    data = scanpy.read_h5ad("/home/bastien/Downloads/2k_cell_per_study_10studies.h5ad")
    np.random.seed(seed)
    samples_chosen = np.random.choice(data.n_obs, max_n)
    Y = data.X.toarray()[samples_chosen]
    GT_name = data.obs["standard_true_celltype_v5"][samples_chosen]
    le = LabelEncoder()
    GT = le.fit_transform(GT_name)
    filter = GT < max_class
    GT = GT[filter]
    GT_name = GT_name[filter]
    Y = Y[filter]
    GT = le.fit_transform(GT)
    best_var = np.var(Y, axis=0).argsort()[::-1][:dim]
    Y = Y[:, best_var]
    return Y, GT, list(GT_name.values.__array__())


n = 2000
p = 5
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

print("coef", pln._coef)
centered_theta = fisher_pln.get_centered_theta(pln._coef)
print("centered theta", centered_theta)
