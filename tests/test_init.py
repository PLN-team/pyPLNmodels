# pylint: skip-file
import torch

from pyPLNmodels._initialization import _init_coef
from pyPLNmodels import load_scrna, PlnPCA


def test_verbose_init_coef():
    data = load_scrna()
    endog = torch.from_numpy(data["endog"].values).float()
    exog = torch.from_numpy(data["labels_1hot"].values).float()
    offsets = torch.from_numpy(0 * data["endog"].values).float()
    _init_coef(
        endog=endog, exog=exog, offsets=offsets, verbose=True, itermax=1000, tol=4
    )
    _init_coef(endog=endog, exog=exog, offsets=offsets, verbose=True, itermax=4)


def test_rank_bigger_than_nsamples():
    data = load_scrna()
    data["endog"] = data["endog"].iloc[:10, :]
    pca = PlnPCA(data["endog"], rank=12)
    pca.fit()
