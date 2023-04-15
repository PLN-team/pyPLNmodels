from pyPLNmodels.models import PLNPCA, _PLNPCA, PLN
from pyPLNmodels import get_real_count_data, get_simulated_count_data

import os

os.chdir("./pyPLNmodels/")


counts = get_real_count_data()
covariates = None
offsets = None
# counts, covariates, offsets = get_simulated_count_data(seed = 0)

pca = PLNPCA([3, 4])

pca.fit(counts, covariates, offsets, tol=0.1)
pln = PLN()
pln.fit(counts, covariates, offsets, tol=0.1)
