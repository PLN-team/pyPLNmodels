import os

import pytest
import matplotlib.pyplot as plt
import numpy as np

from tests.conftest import dict_fixtures
from tests.utils import MSE, filter_models


@pytest.mark.parametrize("plnpca", dict_fixtures["loaded_and_fitted_pln"])
@filter_models(["PLNPCA"])
def test_best_model(plnpca):
    best_model = plnpca.best_model()
    print(best_model)


@pytest.mark.parametrize("plnpca", dict_fixtures["loaded_and_fitted_pln"])
@filter_models(["PLNPCA"])
def test_projected_variables(plnpca):
    best_model = plnpca.best_model()
    plv = best_model.projected_latent_variables
    assert plv.shape[0] == best_model.n_samples and plv.shape[1] == best_model.rank


@pytest.mark.parametrize("plnpca", dict_fixtures["loaded_and_fitted_pln"])
@filter_models(["PLNPCA"])
def test_additional_methods_pca(plnpca):
    plnpca.show()
    plnpca.BIC
    plnpca.AIC
    plnpca.loglikes


@pytest.mark.parametrize("plnpca", dict_fixtures["loaded_and_fitted_pln"])
@filter_models(["PLNPCA"])
def test_viz_pca(plnpca):
    models = plnpca.models
    for model in models:
        _, ax = plt.subplots()
        model.viz(ax=ax)
        plt.show()
        model.viz()
        plt.show()
        n_samples = plnpca.n_samples
        colors = np.random.randint(low=0, high=2, size=n_samples)
        model.viz(colors=colors)
        plt.show()


@pytest.mark.parametrize("plnpca", dict_fixtures["loaded_and_fitted_pln"])
@filter_models(["PLNPCA"])
def test_closest(plnpca):
    with pytest.warns(UserWarning):
        plnpca[9]


@pytest.mark.parametrize("plnpca", dict_fixtures["loaded_and_fitted_pln"])
@filter_models(["PLNPCA"])
def test_wrong_criterion(plnpca):
    with pytest.raises(ValueError):
        plnpca.best_model("AIK")
