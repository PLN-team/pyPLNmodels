import pandas as pd
import os
import numpy as np
import torch
from pyPLNmodels import ZIPln, load_model, PlnPCA, Pln, load_microcosm
import seaborn as sns
import matplotlib.pyplot as plt
import pkg_resources
from sklearn.preprocessing import OneHotEncoder

# counts_orig = pd.read_csv("data_mahendra/counts.tsv", delimiter="\t")
# counts = counts_orig.drop(columns=["Sample"])
# cov = pd.read_csv("data_mahendra/metadata.tsv", delimiter="\t")[
#     ["site", "time", "lineage"]
# ]
# cov_list = ["site", "lineage", "time", "Sample"]
# endog_stream = pkg_resources.resource_stream(
#     __name__, "pyPLNmodels/data/microcosm/counts.tsv"
# )
# endog = pd.read_csv(endog_stream, delimiter="\t")
# cov_stream = pkg_resources.resource_stream(
#     __name__, "pyPLNmodels/data/microcosm/metadata.tsv"
# )
# cov = pd.read_csv(cov_stream, delimiter="\t")[cov_list]
# endog = endog.drop(columns="Sample")
# sample_names = cov["Sample"]
# cov = cov.drop(columns="Sample")

# data = {}
# for name in cov.columns:
#     encoder = OneHotEncoder(drop="first")
#     hot = torch.from_numpy(encoder.fit_transform(cov).toarray())
#     data[name] = hot
# encoder = OneHotEncoder(drop="first")
# exog = encoder.fit_transform(cov).toarray()

# columns = endog.columns
# dict_init = load_model("zi_inter")
data = load_microcosm(get_interaction=False, n_samples=2000, dim=2000, for_formula=True)
dict_init = None
zi = ZIPln.from_formula("endog ~ 1 + exog", data=data, dict_initialization=dict_init)
zi.fit(tol=0, nb_max_iteration=100, verbose=True)
zi.show()
print("loglike", zi.loglike)
print("cov", zi._covariance)
# pd.DataFrame(zi.proba_inflation).to_csv("zi_inter/proba_inflation.csv")
# pd.DataFrame(zi.exog).to_csv("zi_inter/covariates.csv")
# data["exog"].to_csv("zi_inter/covariates_with_names.csv")

# zi.fit(tol = 0, nb_max_iteration = 4000, verbose = True)
# zi.save("zi_inter")
# zi.show()
# print('exog', zi.exog)
# print(' exog shape', zi.exog.shape)
# print('exog infla', zi.exog_inflation)
# print('exog infla shape', zi.exog_inflation.shape)
# zibrute = ZIPln(
#     endog, exog, exog_inflation=exog_inflation, zero_inflation_formula="column-wise"
# )

# print('elbo', zi.elbo)
# zi.fit(nb_max_iteration=500, verbose=True, do_smart_init = True, tol = 0, lr = 0.00000001)
# zi.save("zi_model")
# print('mean prob', torch.mean(zi.proba_inflation))

# sns.heatmap(zi._covariance.cpu().detach())

# plt.savefig("cov.pdf", format="pdf")
# zi.fit(tol=0, nb_max_iteration=10000, verbose=True)
# zi.save("zi_model")
# print("zi shape", zi.endog.shape)


# def squared_mat_to_csv(mat, name):
#     df = pd.DataFrame(mat.cpu().detach())
#     df.columns = columns
#     df.index = columns
#     df.to_csv(name)


# def rectange_mat_to_csv(mat, name):
#     df = pd.DataFrame(mat.cpu().detach())
#     df.columns = columns
#     df.index = sample_names
#     df.to_csv(name)


# def coef_mat_to_csv(mat, name):
#     df = pd.DataFrame(mat.cpu().detach())
#     df.columns = columns
#     df.to_csv(name)


# squared_mat_to_csv(zi.covariance, "csv_mahendra/covariance.csv")
# rectange_mat_to_csv(zi.latent_mean, "csv_mahendra/latent_mean.csv")
# coef_mat_to_csv(zi.coef_inflation, "csv_mahendra/coef_inflation.csv")
# coef_mat_to_csv(zi.coef, "csv_mahendra/coef.csv")

# rectange_mat_to_csv(zi.endog,"toremove.csv")
# df = pd.read_csv("toremove.csv").drop(columns = "Sample")
# print('df:', df)
# print('diff', np.sum(np.sum((df - endog + 4)**2)))
