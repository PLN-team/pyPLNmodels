from pyPLNmodels.VEM import PLNPCA, _PLNPCA
from pyPLNmodels import get_real_count_data


Y = get_real_count_data()

pca = _PLNPCA(3)

pca.fit(Y)
