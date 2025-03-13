# pylint: skip-file
import pytest

from .generate_models import get_dict_collections_fitted


def test_best_model():
    dict_collections = get_dict_collections_fitted()
    for collection_name in dict_collections:
        for init_method in ["formula", "explicit"]:
            for collection in dict_collections[collection_name][init_method]:
                print("best model", collection.best_model())


def test_ranks():
    rna = load_scrna()
    with pytest.raises(TypeError):
        col = PlnPCACollection(rna["endog"], ranks=[3, 4.5])
    with pytest.raises(TypeError):
        col = PlnPCACollection(rna["endog"], ranks=load_scrna)
    with pytest.raises(TypeError):
        col = PlnPCACollection(rna["endog"], ranks=[3.5])
    col = PlnPCACollection(rna["endog"], ranks=[4])


def test_attributes():
    rna = load_scrna()
    col = PlnPCACollection(rna["endog"])
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
    assert (4 in col) is False
    col.keys()
    col.get(3, None)
    col.get(4, None)
    with pytest.raises(ValueError):
        col.best_model(criterion="None")
