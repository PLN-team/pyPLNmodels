from pyPLNmodels.models import PLNPCA, _PLNPCA, PLN
from pyPLNmodels import get_real_count_data, get_simulated_count_data

import os
import pandas as pd
import numpy as np

os.chdir("./pyPLNmodels/")


counts = get_real_count_data()
covariates = None
offsets = None
# counts, covariates, offsets = get_simulated_count_data(seed = 0)

# pca = PLNPCA(counts, covariates, offsets, ranks=[3, 4])
# pca.fit(tol = 0.1)

# pca.fit()
# print(pca)
data = pd.DataFrame(counts)
print("data :", data)
# pln = PLN("counts~1", data)
# pln.fit()
# print(pln)
# pcamodel = pca.best_model()
# pcamodel.save()
# model = PLNPCA([4])[4]

# model.load()
# # pln.fit(counts, covariates, offsets, tol=0.1)
