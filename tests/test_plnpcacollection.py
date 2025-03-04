# pylint: skip-file
import pytest

from pyPLNmodels import PlnPCAcollection, load_scrna


def test_ranks():
    rna = load_scrna()
    with pytest.raises(TypeError):
        col = PlnPCAcollection(rna["endog"], ranks=[3, 4.5])
    with pytest.raises(TypeError):
        col = PlnPCAcollection(rna["endog"], ranks=load_scrna)
    col = PlnPCAcollection(rna["endog"], ranks=4)


def test_attributes():
    rna = load_scrna()
    col = PlnPCAcollection(rna["endog"])
    col.fit()
    col.exog
    col.endog
    col.offsets
    col.components
    col.coef
    col.nb_cov
    col.latent_mean
    col.items()
    for pca in col:
        print(pca)
    assert 3 in col
    not_in_col = 4 in col
    assert not_in_col is False
    col.keys()
    col.get(3, None)
    col.get(4, None)
    with pytest.raises(ValueError):
        col.best_model(criterion="None")
