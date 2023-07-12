#!/usr/bin/env python
# coding: utf-8

# get_ipython().system('pip install pyPLNmodels')


# ## pyPLNmodels

# We assume the data comes from a PLN model:  $ \text{counts} \sim  \mathcal P(\exp(\text{Z}))$, where $Z$ are some unknown latent variables.
#
#
# The goal of the package is to retrieve the latent variables $Z$ given the counts. To do so, one can instantiate a Pln or PlnPCA model, fit it and then extract the latent variables.

# ### Import the needed functions

from pyPLNmodels import (
    get_real_count_data,
    get_simulated_count_data,
    load_model,
    Pln,
    PlnPCA,
    PlnPCAcollection,
)
import matplotlib.pyplot as plt


# ### Load the data

counts, labels = get_real_count_data(return_labels=True)  # np.ndarray


# ### PLN model

pln = Pln(counts, add_const=True)
pln.fit()


print(pln)


# #### Once fitted, we can extract multiple variables:

gaussian = pln.latent_variables
print(gaussian.shape)


model_param = pln.model_parameters
print(model_param["coef"].shape)
print(model_param["covariance"].shape)


# ### PlnPCA model

pca = PlnPCA(counts, add_const=True, rank=5)
pca.fit()


print(pca)


print(pca.latent_variables.shape)


print(pca.model_parameters["components"].shape)
print(pca.model_parameters["coef"].shape)


# ### One can save the model in order to load it back after:

pca.save()
dict_init = load_model("PlnPCA_nbcov_1_dim_200_rank_5")
loaded_pca = PlnPCA(counts, add_const=True, dict_initialization=dict_init)
print(loaded_pca)


# ### One can fit multiple PCA and choose the best rank with BIC or AIC criterion

pca_col = PlnPCAcollection(counts, add_const=True, ranks=[5, 15, 25, 40, 50])
pca_col.fit()


pca_col.show()


print(pca_col)


# ### One can extract the best model found (according to AIC or BIC criterion).

# #### AIC best model

print(pca_col.best_model(criterion="AIC"))


# #### BIC best model

print(pca_col.best_model(criterion="BIC"))


# #### Visualization of the individuals (sites) with PCA on the latent variables.

pln.viz(colors=labels)
plt.show()


best_pca = pca_col.best_model()
best_pca.viz(colors=labels)
plt.show()


# ### What would give a PCA on the log normalize data ?

from sklearn.decomposition import PCA
import numpy as np
import seaborn as sns


sk_pca = PCA(n_components=2)
pca_log_counts = sk_pca.fit_transform(np.log(counts + (counts == 0)))
sns.scatterplot(x=pca_log_counts[:, 0], y=pca_log_counts[:, 1], hue=labels)


# ### Visualization of the variables

pln.plot_pca_correlation_graph(["var_1", "var_2"], indices_of_variables=[0, 1])
plt.show()


best_pca.plot_pca_correlation_graph(["var_1", "var_2"], indices_of_variables=[0, 1])
plt.show()


# ### Visualization of each components of the PCA
#

pln.scatter_pca_matrix(color=labels, n_components=5)
plt.show()


best_pca.scatter_pca_matrix(color=labels, n_components=6)
plt.show()
