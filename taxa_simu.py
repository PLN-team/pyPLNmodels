import pandas as pd
import os
import numpy as np
import torch
from sklearn.preprocessing import OneHotEncoder
from pyPLNmodels import ZIPln, load_model, PlnPCA, Pln
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MiniBatchKMeans, HDBSCAN
from sklearn.mixture import GaussianMixture

counts_orig = pd.read_csv("data_mahendra/counts.tsv", delimiter="\t")
counts = counts_orig.drop(columns=["Sample"])
cov = pd.read_csv("data_mahendra/metadata.tsv", delimiter="\t")[
    ["site", "time", "lineage"]
]
encoder = OneHotEncoder(drop="first")
hot_cov = encoder.fit_transform(cov).toarray()
hot_cov = torch.from_numpy(hot_cov)
# print('rank hot_cov', torch.linalg.matrix_rank(hot_cov))
# print('inverse hot cov', torch.inverse(hot_cov))


dict_init = load_model("ZIPln_nbcov_8_dim_1209")
zi = ZIPln(
    counts,
    exog=hot_cov,
    offsets=None,
    exog_inflation=hot_cov,
    add_const=True,
    add_const_inflation=True,
    zero_inflation_formula="column-wise",
    dict_initialization=dict_init,
)
zi.fit(do_smart_init=False, tol=0.0001)


latent = zi.transform()
skpca = zi.sk_PCA(n_components=2)
latent = skpca.transform(latent)
km = KMeans(n_clusters=10).fit(latent)
labels = km.labels_
# zi.scatter_pca_matrix(n_components = 2, colors = labels)

zi_nocov = ZIPln(
    counts,
    exog=hot_cov,
    offsets=None,
    # exog_inflation=hot_cov,
    add_const=True,
    add_const_inflation=True,
    zero_inflation_formula="global"
    # dict_initialization=dict_init,
)
zi_nocov.fit(do_smart_init=False, tol=0.00001)
# gm = GaussianMixture(n_components = 10).fit(latent)
# labels = gm.predict(latent)
# mkm = MiniBatchKMeans(n_clusters = 10).fit(latent)
# labels = mkm.labels_
# hdb = HDBSCAN(min_cluster_size = 5).fit(latent)
# labels = hdb.labels_
# zi.viz( colors = labels)
# zi.save()
# zi.scatter_pca_matrix(n_components = 5)
pca = PlnPCA(counts, exog=hot_cov, offsets=None, rank=15)
pca.fit(do_smart_init=False, tol=1)
# pca.viz( colors = labels)

pln = Pln(counts, exog=hot_cov, offsets=None)
pln.fit(tol=0.000001)
# pln.viz(colors = labels)

fig, axes = plt.subplots(4)
ax_zi = axes[0]
ax_pln = axes[1]
ax_plnpca = axes[2]
ax_zi_global = axes[3]
zi.plot_expected_vs_true(ax=ax_zi)
pln.plot_expected_vs_true(ax=ax_pln)
pca.plot_expected_vs_true(ax=ax_plnpca)
zi_nocov.plot_expected_vs_true(ax=ax_zi_global)

plt.show()


# zi.show()
# zi.save()
# zi.show()
# sns.heatmap(zi.latent_prob)
# plt.show()
# zi.plot_expected_vs_true()
# zi.plot_pca_correlation_circle(indices_of_variables = [1,2,3], variables_names = ["1","2","3"] )
# zi.pca_projected_latent_variables(n_components = 2)
# print('counts shape', counts.shape)
# print('orig columns length', counts.columns)
# covariance = pd.DataFrame(
#     zi.covariance.numpy(), columns=counts.columns, index=counts.columns
# )
# covariance.to_csv("covariance.csv")
# prob_pd = pd.DataFrame(zi.latent_prob.numpy(), columns=counts.columns)
# prob_pd = prob_pd.set_index(counts_orig["Sample"], "Sample")
# prob_pd.to_csv("latent_probability.csv")
# print('prob pb ', prob_pd)
# prob_pd.to_csv("latent_probability.csv")
# print(cov)
# print(zi.covariance)
