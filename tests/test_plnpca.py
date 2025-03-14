# pylint: skip-file

import pytest
from pyPLNmodels import PlnPCA, ZIPlnPCA, load_scrna, PlnNetwork, PlnMixture


def test_wrong_grid_value():
    data = load_scrna()
    with pytest.raises(AttributeError):
        pca = PlnPCA(data["endog"], rank=-1)

    with pytest.raises(AttributeError):
        pca = ZIPlnPCA(data["endog"], rank=-1)

    with pytest.raises(AttributeError):
        pca = PlnPCA(data["endog"], rank=1.5)

    with pytest.raises(AttributeError):
        pca = ZIPlnPCA(data["endog"], rank=1.5)

    with pytest.raises(AttributeError):
        pca = PlnMixture(data["endog"], n_cluster=1.5)

    with pytest.raises(AttributeError):
        pca = PlnMixture(data["endog"], n_cluster=-1)

    with pytest.raises(AttributeError):
        pca = PlnNetwork(data["endog"], penalty=-1)
