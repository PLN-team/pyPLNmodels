import os

import pytest
import matplotlib.pyplot as plt
import numpy as np

from tests.conftest import dict_fixtures
from tests.utils import MSE, filter_models


@pytest.mark.parametrize("plnpca", dict_fixtures["loaded_and_fitted_pln"])
@filter_models(["PlnPCAcollection"])
def test_best_model(plnpca):
    best_model = plnpca.best_model()
    print(best_model)


@pytest.mark.parametrize("plnpca", dict_fixtures["loaded_and_fitted_pln"])
@filter_models(["PlnPCAcollection"])
def test_projected_variables(plnpca):
    best_model = plnpca.best_model()
    plv = best_model.projected_latent_variables
    assert plv.shape[0] == best_model.n_samples and plv.shape[1] == best_model.rank


@pytest.mark.parametrize("fitted_pln", dict_fixtures["fitted_pln"])
@filter_models(["PlnPCA"])
def test_number_of_iterations_plnpca(fitted_pln):
    nb_iterations = len(fitted_pln._elbos_list)
    assert 100 < nb_iterations < 5000


@pytest.mark.parametrize("plnpca", dict_fixtures["loaded_and_fitted_pln"])
@filter_models(["PlnPCA"])
def test_latent_var_pca(plnpca):
    assert plnpca.transform(project=False).shape == plnpca.counts.shape
    assert plnpca.transform().shape == (plnpca.n_samples, plnpca.rank)


@pytest.mark.parametrize("plnpca", dict_fixtures["loaded_and_fitted_pln"])
@filter_models(["PlnPCAcollection"])
def test_additional_methods_pca(plnpca):
    plnpca.show()
    plnpca.BIC
    plnpca.AIC
    plnpca.loglikes


@pytest.mark.parametrize("plnpca", dict_fixtures["loaded_and_fitted_pln"])
@filter_models(["PlnPCAcollection"])
def test_wrong_criterion(plnpca):
    with pytest.raises(ValueError):
        plnpca.best_model("AIK")


@pytest.mark.parametrize("collection", dict_fixtures["loaded_and_fitted_pln"])
@filter_models(["PlnPCAcollection"])
def test_item(collection):
    print(collection[collection.ranks[0]])
    with pytest.raises(KeyError):
        collection[collection.ranks[0] + 50]
    assert collection.ranks[0] in collection
    assert collection.ranks[0] in list(collection.keys())
    collection.get(collection.ranks[0], None)
