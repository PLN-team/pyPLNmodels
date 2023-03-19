from pyPLNmodels.VEM import PLN, PLNPCA
import torch
import numpy as np
import pandas as pd

if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"


Y = pd.read_csv("./example_data/real_data/oaks_counts.csv")
covariates = None
O = np.log(pd.read_csv("./example_data/real_data/oaks_offsets.csv"))

# pln = PLN()
# pln.fit(Y,covariates,O)

# pln.save_model("param")
# pln.load_model_from_file("param")
# print(pln.beta)


pca = _PLNPCA(ranks=5)
# pca.fit(Y,covariates,O)
# pca.best_model.save_model("param")
pca.load_model_from_file("param")
# print(pca)
