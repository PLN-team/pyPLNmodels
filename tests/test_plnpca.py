import os

import pytest
from pytest_lazyfixture import lazy_fixture as lf
from pyPLNmodels.models import PLNPCA, _PLNPCA
from pyPLNmodels import get_simulated_count_data, get_real_count_data
from tests.utils import MSE

import matplotlib.pyplot as plt
import numpy as np

(
    counts_sim,
    covariates_sim,
    offsets_sim,
    true_covariance,
    true_coef,
) = get_simulated_count_data(return_true_param=True)

counts_real = get_real_count_data()
RANKS = [2, 8]


@pytest.fixture
def my_instance_plnpca():
    plnpca = PLNPCA(ranks=RANKS)
    return plnpca


@pytest.fixture
def real_fitted_plnpca(my_instance_plnpca):
    my_instance_plnpca.fit(counts_real)
    return my_instance_plnpca


@pytest.fixture
def simulated_fitted_plnpca(my_instance_plnpca):
    my_instance_plnpca.fit(
        counts=counts_sim, covariates=covariates_sim, offsets=offsets_sim
    )
    return my_instance_plnpca


@pytest.fixture
def one_simulated_fitted_plnpca():
    model = PLNPCA(ranks=2)
    model.fit(counts=counts_sim, covariates=covariates_sim, offsets=offsets_sim)
    return model


@pytest.fixture
def real_best_aic(real_fitted_plnpca):
    return real_fitted_plnpca.best_model("AIC")


@pytest.fixture
def real_best_bic(real_fitted_plnpca):
    return real_fitted_plnpca.best_model("BIC")


@pytest.fixture
def simulated_best_aic(simulated_fitted_plnpca):
    return simulated_fitted_plnpca.best_model("AIC")


@pytest.fixture
def simulated_best_bic(simulated_fitted_plnpca):
    return simulated_fitted_plnpca.best_model("BIC")


simulated_best_models = [lf("simulated_best_aic"), lf("simulated_best_bic")]
real_best_models = [lf("real_best_aic"), lf("real_best_bic")]
best_models = simulated_best_models + real_best_models


all_fitted_simulated_plnpca = [
    lf("simulated_fitted_plnpca"),
    lf("one_simulated_fitted_plnpca"),
]
all_fitted_plnpca = [lf("real_fitted_plnpca")] + all_fitted_simulated_plnpca


def test_print_plnpca(simulated_fitted_plnpca):
    print(simulated_fitted_plnpca)


@pytest.mark.parametrize("best_model", best_models)
def test_best_model(best_model):
    print(best_model)


@pytest.mark.parametrize("best_model", best_models)
def test_projected_variables(best_model):
    plv = best_model.projected_latent_variables
    assert plv.shape[0] == best_model.n_samples and plv.shape[1] == best_model.rank


def test_save_load_back_and_refit(simulated_fitted_plnpca):
    simulated_fitted_plnpca.save()
    new = PLNPCA(ranks=RANKS)
    new.load()
    new.fit(counts=counts_sim, covariates=covariates_sim, offsets=offsets_sim)


@pytest.mark.parametrize("plnpca", all_fitted_simulated_plnpca)
def test_find_right_covariance(plnpca):
    passed = True
    for model in plnpca.models:
        mse_covariance = MSE(model.covariance - true_covariance)
        assert mse_covariance < 0.3


@pytest.mark.parametrize("plnpca", all_fitted_simulated_plnpca)
def test_find_right_coef(plnpca):
    for model in plnpca.models:
        mse_coef = MSE(model.coef - true_coef)
        assert mse_coef < 0.3


@pytest.mark.parametrize("all_pca", all_fitted_plnpca)
def test_additional_methods_pca(all_pca):
    all_pca.show()
    all_pca.BIC
    all_pca.AIC
    all_pca.loglikes


@pytest.mark.parametrize("all_pca", all_fitted_plnpca)
def test_viz_pca(all_pca):
    _, ax = plt.subplots()
    all_pca[2].viz(ax=ax)
    plt.show()
    all_pca[2].viz()
    plt.show()
    n_samples = all_pca.n_samples
    colors = np.random.randint(low=0, high=2, size=n_samples)
    all_pca[2].viz(colors=colors)
    plt.show()


@pytest.mark.parametrize(
    "pca", [lf("real_fitted_plnpca"), lf("simulated_fitted_plnpca")]
)
def test_fails_viz_pca(pca):
    with pytest.raises(RuntimeError):
        pca[8].viz()


@pytest.mark.parametrize("all_pca", all_fitted_plnpca)
def test_closest(all_pca):
    with pytest.warns(UserWarning):
        all_pca[9]


@pytest.mark.parametrize("plnpca", all_fitted_plnpca)
def test_wrong_criterion(plnpca):
    with pytest.raises(ValueError):
        plnpca.best_model("AIK")
