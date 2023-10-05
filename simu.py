import scanpy
from pyPLNmodels.models import Pln, ZIPln
from sklearn.preprocessing import LabelEncoder
import numpy as np

from xgboost import XGBClassifier


from sklearn.model_selection import cross_val_score


def get_sc_mark_data(max_class=28, max_n=200, dim=100):
    data = scanpy.read_h5ad("2k_cell_per_study_10studies.h5ad")
    Y = data.X.toarray()[:max_n]
    GT_name = data.obs["standard_true_celltype_v5"][:max_n]
    le = LabelEncoder()
    GT = le.fit_transform(GT_name)
    filter = GT < max_class
    unique, index = np.unique(GT, return_counts=True)
    enough_elem = index > 15
    classes_with_enough_elem = unique[enough_elem]
    filter_bis = np.isin(GT, classes_with_enough_elem)
    mask = filter * filter_bis
    GT = GT[mask]
    GT_name = GT_name[mask]
    Y = Y[mask]
    GT = le.fit_transform(GT)
    not_only_zeros = np.sum(Y, axis=0) > 0
    Y = Y[:, not_only_zeros]
    var = np.var(Y, axis=0)
    print("var ", var.shape)
    # most_variables = np.argsort(var)[11200:(11200+dim)]
    # most_variables = np.argsort(var)[:dim]
    most_variables = np.argsort(var)[-dim:]
    Y = Y[:, most_variables]
    return Y, GT, list(GT_name.values.__array__())


Y, GT, _ = get_sc_mark_data(max_n=19998, dim=12000)
# print('Y:', Y)
print("percentage of zeros", np.sum(Y == 0) / np.prod(Y.shape))
print(Y.shape)
x

zi = ZIPln(Y, use_closed_form_prob=True)
tol = 0.0001
nb_iter = 100
zi.fit(tol=tol, verbose=True, nb_max_iteration=nb_iter)
# print(zi)
# zi.show()

pln = Pln(Y)
pln.fit(tol=tol, verbose=True, nb_max_iteration=nb_iter)
# pln.show()


Z_ZI, W_ZI = zi.transform()
latent_zi = np.concatenate((Z_ZI, W_ZI), axis=-1)
# print('latent', latent_zi.shape)
Z_pln = pln.transform()
xgb = XGBClassifier()
cv_zi = cross_val_score(xgb, Z_ZI, GT, cv=5, scoring="balanced_accuracy")
print("cv_zi:", cv_zi, "  mean ", np.mean(cv_zi))

cv_pln = cross_val_score(xgb, Z_pln, GT, cv=5, scoring="balanced_accuracy")
print("cv pln ", cv_pln, "  mean", np.mean(cv_pln))
