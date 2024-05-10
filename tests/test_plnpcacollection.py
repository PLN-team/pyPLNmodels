import os

import pytest
import matplotlib.pyplot as plt
import numpy as np

from tests.conftest import dict_fixtures
from tests.utils import MSE, filter_models
from tests.import_data import true_sim_0cov, true_sim_2cov

from pyPLNmodels import PlnPCAcollection


@pytest.mark.parametrize("plnpca", dict_fixtures["loaded_and_fitted_model"])
@filter_models(["PlnPCAcollection"])
def test_best_model(plnpca):
    best_model = plnpca.best_model()
    print(best_model)


@pytest.mark.parametrize("plnpca", dict_fixtures["loaded_and_fitted_model"])
@filter_models(["PlnPCAcollection"])
def test_projected_variables(plnpca):
    best_model = plnpca.best_model()
    plv = best_model.projected_latent_variables
    assert plv.shape[0] == best_model.n_samples and plv.shape[1] == best_model.rank


@pytest.mark.parametrize("plnpca", dict_fixtures["sim_model_instance"])
@filter_models(["PlnPCAcollection"])
def test_right_nbcov(plnpca):
    assert plnpca.nb_cov == 0 or plnpca.nb_cov == 2


@pytest.mark.parametrize("plnpca", dict_fixtures["loaded_and_fitted_model"])
@filter_models(["PlnPCA"])
def test_latent_var_pca(plnpca):
    assert plnpca.transform().shape == plnpca.endog.shape
    assert plnpca.transform(project=True).shape == (plnpca.n_samples, plnpca.rank)


@pytest.mark.parametrize("plnpca", dict_fixtures["loaded_and_fitted_model"])
@filter_models(["PlnPCAcollection"])
def test_additional_methods_pca(plnpca):
    plnpca.show()
    plnpca.BIC
    plnpca.AIC
    plnpca.loglikes


@pytest.mark.parametrize("plnpca", dict_fixtures["loaded_and_fitted_model"])
@filter_models(["PlnPCAcollection"])
def test_wrong_criterion(plnpca):
    with pytest.raises(ValueError):
        plnpca.best_model("AIK")


@pytest.mark.parametrize("collection", dict_fixtures["loaded_and_fitted_model"])
@filter_models(["PlnPCAcollection"])
def test_item(collection):
    print(collection[collection.ranks[0]])
    with pytest.raises(KeyError):
        collection[collection.ranks[0] + 50]
    assert collection.ranks[0] in collection
    assert collection.ranks[0] in list(collection.keys())
    collection.get(collection.ranks[0], None)


@pytest.mark.parametrize("collection", dict_fixtures["sim_model_instance"])
@filter_models(["PlnPCAcollection"])
def test_batch(collection):
    exog = collection.exog
    endog = collection.endog
    offsets = collection.offsets
    ranks = collection.ranks
    collection = PlnPCAcollection(
        endog, exog=exog, offsets=offsets, ranks=ranks, batch_size=20, add_const=False
    )
    collection.fit()
    assert collection.nb_cov == 0 or collection.nb_cov == 2
    if collection.nb_cov == 0:
        true_covariance = true_sim_0cov["Sigma"]
        for model in collection.values():
            assert model.coef is None
        true_coef = None
    elif collection.nb_cov == 2:
        true_covariance = true_sim_2cov["Sigma"]
        true_coef = true_sim_2cov["beta"]
    else:
        raise ValueError(f"Not the right numbers of covariance({collection.nb_cov})")
    for model in collection.values():
        mse_covariance = MSE(model.covariance - true_covariance)
        if true_coef is not None:
            mse_coef = MSE(model.coef - true_coef)
            assert mse_coef < 0.35
        assert mse_covariance < 1.5
    collection.fit()
    assert collection.batch_size == 20
