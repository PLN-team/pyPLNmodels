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
from sklearn.decomposition import PCA

counts_orig = pd.read_csv("data_mahendra/counts.tsv", delimiter="\t")
counts = counts_orig.drop(columns=["Sample"])
cov = pd.read_csv("data_mahendra/metadata.tsv", delimiter="\t")[
    ["site", "time", "lineage"]
]
encoder = OneHotEncoder(drop="first")
hot_cov = encoder.fit_transform(cov).toarray()
hot_cov = torch.from_numpy(hot_cov)


fig, axes = plt.subplots(4)
ax_zi = axes[0]
ax_zi_global = axes[1]
ax_pln = axes[2]
ax_plnpca = axes[3]


def viz(model, colors=None, ax=None):
    latent = model.transform()
    pca = PCA(n_components=2)
    latent = pca.fit_transform(latent)
    if ax is None:
        to_show = True
    else:
        to_show = False
    sns.scatterplot(x=latent[:, 0], y=latent[:, 1], hue=colors, ax=ax)
    if to_show is True:
        plt.show()


dict_init = load_model("zicov")
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
latent_zi = zi.latent_mean

latent = zi.transform()
skpca = zi.sk_PCA(n_components=2)
latent = skpca.transform(latent)
km = KMeans(n_clusters=10).fit(latent)
labels = km.labels_


viz(zi, ax=ax_zi, colors=labels)
# zi.fit(do_smart_init=False, tol=0.0001)
# zi.save("zicov")
# print('reconstructed zi', zi._endog_predictions())
mask = counts > 0
print(
    "reconstructionpln ",
    np.mean((zi._endog_predictions().numpy() - counts)[mask] ** 2).mean(),
)

dict_init_pln = load_model("pln")
pln = Pln(counts, exog=hot_cov, offsets=None, dict_initialization=dict_init_pln)
viz(pln, ax=ax_pln, colors=labels)
print(
    "reconstructionpln ",
    np.mean((pln._endog_predictions().numpy() - counts)[mask] ** 2).mean(),
)

# zi.scatter_pca_matrix(n_components = 2, colors = labels)
# zi.viz()

dict_init_nocov = load_model("zinocov")
zi_global = ZIPln(
    counts,
    exog=hot_cov,
    offsets=None,
    # exog_inflation=hot_cov,
    add_const=True,
    add_const_inflation=True,
    zero_inflation_formula="global",
    dict_initialization=dict_init_nocov,
)
viz(zi_global, ax=ax_zi_global, colors=labels)
# zi_global.viz()
# zi_global.fit(do_smart_init=False, tol=0.00001)
# zi_global.save("zinocov")


# gm = GaussianMixture(n_components = 10).fit(latent)
# labels = gm.predict(latent)
# mkm = MiniBatchKMeans(n_clusters = 10).fit(latent)
# labels = mkm.labels_
# hdb = HDBSCAN(min_cluster_size = 5).fit(latent)
# labels = hdb.labels_
# zi.viz( colors = labels)
# zi.save()
# zi.scatter_pca_matrix(n_components = 5)
dict_init_pca = load_model("pca")
pca = PlnPCA(
    counts, exog=hot_cov, offsets=None, rank=15, dict_initialization=dict_init_pca
)
# pca.viz()
viz(pca, ax=ax_plnpca, colors=labels)

# pca.fit(do_smart_init=False, tol=0.00001)
# pca.save("pca")
# pca.viz( colors = labels)
# pln.fit(tol=0.000001)
# pln.save("pln")
# pln.viz(colors = labels)

# zi.plot_expected_vs_true(ax=ax_zi)
# pln.plot_expected_vs_true(ax=ax_pln)
# pca.plot_expected_vs_true(ax=ax_plnpca)
# zi_global.plot_expected_vs_true(ax=ax_zi_global)

plt.show()
