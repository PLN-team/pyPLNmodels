import os

import pytest
from pytest_lazyfixture import lazy_fixture as lf
from pyPLNmodels.models import PLNPCA, _PLNPCA
from pyPLNmodels import get_simulated_count_data, get_real_count_data
from tests.utils import MSE

os.chdir("./pyPLNmodels/")
(
    counts_sim,
    covariates_sim,
    offsets_sim,
    true_covariance,
    true_coef,
) = get_simulated_count_data(return_true_param=True)

counts_real = get_real_count_data()
RANKS = [4, 8]


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

fitted_plnpca = [lf("simulated_fitted_plnpca"), lf("real_fitted_plnpca")]


@pytest.mark.parametrize("best_model", best_models)
def test_best_model(best_model):
    print(best_model)


@pytest.mark.parametrize("best_model", best_models)
def test_projected_variables(best_model):
    plv = best_model.projected_latent_variables
    assert plv.shape[0] == best_model.n and plv.shape[0] == plv.rank


def test_find_right_covariance(simulated_fitted_plnpca):
    passed = True
    for model in simulated_fitted_plnpca.models:
        mse_covariance = MSE(model.covariance - true_covariance)
        assert mse_covariance < 0.3


def test_find_right_coef(simulated_fitted_plnpca):
    passed = True
    for model in simulated_fitted_plnpca.models:
        mse_coef = MSE(model.coef - true_coef)
        assert mse_coef < 0.3


def test_additional_methods_pca(plnpca):
    return True
