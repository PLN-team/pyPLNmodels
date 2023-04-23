from pyPLNmodels.models import PLN, PLNPCA
from pyPLNmodels import get_simulated_count_data, get_real_count_data
import pytest
from pytest_lazyfixture import lazy_fixture as lf
import pandas as pd
import numpy as np

(
    counts_sim,
    covariates_sim,
    offsets_sim,
) = get_simulated_count_data()

couts_real = get_real_count_data()
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
        counts=counts_sim, covariates=covariates_sim, offsets=offsets
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


def test_pandas_init(my_instance_plnpca):
    my_instance_plnpca.fit(
        pd.DataFrame(counts_sim.numpy()),
        pd.DataFrame(covariates_sim.numpy()),
        pd.DataFrame(offsets_sim.numpy()),
    )


simulated_best_models = [lf("simulated_best_aic"), lf("simulated_best_bic")]
real_best_models = [lf("real_best_aic"), lf("real_best_bic")]
best_models = simulated_best_models + real_best_models


@pytest.mark.parametrize("best_model", best_models)
def test_best_model(best_models):
    print(best_models)
