import pandas as pd
import os
import numpy as np
import torch
from sklearn.preprocessing import OneHotEncoder
from pyPLNmodels import ZIPln
import seaborn as sns
import matplotlib.pyplot as plt

print(os.getcwd())
counts = pd.read_csv("data_mahendra/counts.tsv", delimiter="\t").drop(
    columns=["Sample"]
)
cov = pd.read_csv("data_mahendra/metadata.tsv", delimiter="\t")[
    ["site", "time", "lineage"]
]
encoder = OneHotEncoder()
hot_cov = encoder.fit_transform(cov).toarray()
zi = ZIPln(
    counts, exog=hot_cov, offsets=None, exog_inflation=hot_cov, add_const_inflation=True
)
zi.fit(verbose=True, do_smart_init=False, nb_max_iteration=1000)
zi.save()
zi.show()
sns.heatmap(zi.latent_prob)
plt.show()
