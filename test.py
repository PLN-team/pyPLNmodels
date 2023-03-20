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
# pln_bis = PLN()
# pln_bis.load_model_from_file("param")
# print(pln_bis.beta)


pca = PLNPCA(ranks=[4, 5])
pca.fit(Y, covariates, O, tol=0.1)
print(pca.best_model())
# pca.save_model(5,"param")
# pca_bis = PLNPCA(ranks = [4,5])
# pca_bis.load_model_from_file(5,"param")
