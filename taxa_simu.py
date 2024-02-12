import pandas as pd
import os
import numpy as np
import torch
from sklearn.preprocessing import OneHotEncoder
from pyPLNmodels import ZIPln, load_model
import seaborn as sns
import matplotlib.pyplot as plt

print(os.getcwd())
counts_orig = pd.read_csv("data_mahendra/counts.tsv", delimiter="\t")
counts = counts_orig.drop(columns=["Sample"])
cov = pd.read_csv("data_mahendra/metadata.tsv", delimiter="\t")[
    ["site", "time", "lineage"]
]
encoder = OneHotEncoder()
hot_cov = encoder.fit_transform(cov).toarray()
dict_init = load_model("ZIPln_nbcov_10_dim_1209")
zi = ZIPln(
    counts,
    exog=hot_cov,
    offsets=None,
    exog_inflation=hot_cov,
    add_const_inflation=True,
    dict_initialization=dict_init,
)
# zi.fit(verbose=True, nb_max_iteration=1000)
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
covariance = pd.DataFrame(
    zi.covariance.numpy(), columns=counts.columns, index=counts.columns
)
covariance.to_csv("covariance.csv")
prob_pd = pd.DataFrame(zi.latent_prob.numpy(), columns=counts.columns)
prob_pd = prob_pd.set_index(counts_orig["Sample"], "Sample")
prob_pd.to_csv("latent_probability.csv")
# print('prob pb ', prob_pd)
# prob_pd.to_csv("latent_probability.csv")
# print(cov)
# print(zi.covariance)
