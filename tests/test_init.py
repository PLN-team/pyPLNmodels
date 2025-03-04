# pylint: skip-file
import torch

from pyPLNmodels.calculations._initialization import _init_coef
from pyPLNmodels import load_scrna, PlnPCA

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def test_verbose_init_coef():
    scrna = load_scrna()
    endog = torch.from_numpy(scrna["endog"].values).float().to(DEVICE)
    exog = torch.from_numpy(scrna["labels_1hot"].values).float().to(DEVICE)
    offsets = torch.from_numpy(0 * scrna["endog"].values).float().to(DEVICE)
    _init_coef(
        endog=endog, exog=exog, offsets=offsets, verbose=True, itermax=1000, tol=4
    )
    _init_coef(endog=endog, exog=exog, offsets=offsets, verbose=True, itermax=4)


def test_rank_bigger_than_nsamples():
    scrna = load_scrna()
    scrna["endog"] = scrna["endog"].iloc[:10, :]
    pca = PlnPCA(scrna["endog"], rank=12)
    pca.fit()
