from pyPLNmodels.VEM import PLNPCA, PLN, _PLNPCA, _PLNPCA_noS
from pyPLNmodels._utils import get_simulated_count_data, get_real_count_data
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

print(os.getcwd())
os.chdir("pyPLNmodels/")

Y = get_real_count_data()

pca = _PLNPCA(3)

pca.fit(Y)
