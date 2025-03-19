# pylint: skip-file
import pytest

from .conftest import dict_collections_fitted

from pyPLNmodels import (
    PlnPCACollection,
    ZIPlnPCACollection,
    PlnNetworkCollection,
    load_scrna,
)


def test_ranks():
    rna = load_scrna()
    with pytest.raises(TypeError):
        col = PlnPCACollection(rna["endog"], ranks=[3, 4.5])
    with pytest.raises(TypeError):
        col = ZIPlnPCACollection(rna["endog"], ranks=[3, 4.5])
    with pytest.raises(TypeError):
        col = PlnPCACollection(rna["endog"], ranks=load_scrna)
    with pytest.raises(TypeError):
        col = ZIPlnPCACollection(rna["endog"], ranks=load_scrna)
    col = PlnPCACollection(rna["endog"], ranks=[4])


def test_attributes():
    for collection_name in dict_collections_fitted:
        for init_method in ["formula", "explicit"]:
            for col in dict_collections_fitted[collection_name][init_method]:
                print(col.exog)
                print(col.endog)
                print(col.offsets)
                print(col.coef)
                print(col.nb_cov)
                if collection_name in ["PlnPCACollection", "ZIPlnPCACollection"]:
                    print(col.components)
                    assert 3 in col
                if collection_name == "PlnNetworkCollection":
                    print(col.components_prec)
                    assert 100 in col
                if collection_name == ["PlnMixtureCollection"]:
                    assert 2 in col
                if collection_name in [
                    "PlnPCACollection",
                    "ZIPlnPCACollection",
                    "PlnNetworkCollection",
                ]:
                    print(col.latent_mean)
                else:
                    print(col.latent_means)
                if collection_name == "ZIPlnPCACollection":
                    print(col.nb_cov_inflation)
                print(col.items())
                for model in col.values():
                    print(model)
                    model.show()
                assert 8 not in col
                print(col.keys())
                print(col.get(3, None))
                print(col.get(4, None))
                with pytest.raises(ValueError):
                    col.best_model(criterion="None")
                print(col.best_model())
                col.show()
                col.best_model(criterion="AIC")
