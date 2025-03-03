# pylint: skip-file
import pytest

from pyPLNmodels import PlnLDA, load_scrna


def test_properties():
    data = load_scrna()
    with pytest.raises(ValueError):
        PlnLDA(data["endog"])
    lda = PlnLDA(data["endog"], clusters=data["labels"]).fit()
    assert lda._marginal_mean_clusters.shape == (lda.n_samples, 3)
