# pylint: skip-file
import pytest

from pyPLNmodels import PlnLDA, load_scrna


def test_properties():
    data = load_scrna()
    lda = PlnLDA(data["endog"], clusters=data["labels"]).fit()
    assert lda._marginal_mean_clusters.shape == (lda.n_samples, lda.dim)
