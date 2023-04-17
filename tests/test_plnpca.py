import pytest
from pytest_lazyfixture import lazy_fixture as lf

from pyPLNmodels.models import PLN, PLNPCA, _PLNPCA
from tests.utils import MSE
from pyPLNmodels import get_simulated_count_data

RANKS = [2, 4]

(
    counts_sim,
    covariates_sim,
    offsets_sim,
    true_Sigma,
    true_beta,
) = get_simulated_count_data(return_true_param=True)


@pytest.fixture
def my_instance_plnpca():
    plnpca = PLNPCA(ranks=RANKS)
    return plnpca


@pytest.fixture
def simulated_fitted_plnpca():
    plnpca = PLNPCA(RANKS)
    plnpca.fit(counts=counts_sim, covariates=covariates_sim, offsets=offsets_sim)
    return plnpca


@pytest.fixture
def best_aic_model(plnpca):
    return plnpca.best_model("AIC")


@pytest.fixture
def best_bic_model(plnpca):
    return plnpca.best_model("BIC")


@pytest.mark.parametrize("best_model", [lf("best_aic_model"), lf("best_bic_model")])
def test_projected_variables(best_model):
    plv = best_model.projected_latent_variables
    assert plv.shape[0] == best_model.n and plv.shape[0] == plv.rank


def test_find_right_Sigma(simulated_fitted_plnpca):
    passed = True
    for model in simulated_fitted_plnpca.models:
        mse_Sigma = MSE(model.Sigma - true_Sigma)
        if mse_Sigma > 0.3:
            return False
    return True


def test_find_right_beta(simulated_fitted_plnpca):
    passed = True
    for model in simulated_fitted_plnpca.models:
        mse_beta = MSE(model.beta - true_beta)
        if mse_beta > 0.3:
            passed = False
    assert passed


def test_additional_methods_pca(plnpca):
    return True


def test_computable_elbo(simulated_fitted_plnpca):
    new_pca = _PLNPCA(simulated_fitted_plnpca.rank)
    new_pca.counts = simulated_fitted_plnpca.counts
    new_pca.covariates = simulated_fitted_plnpca._covariates
    new_pca.counts = simulated_fitted_plnpca._offsets
    new_pca.latent_mean = simulated_fitted_plnpca._latent_mean
    new_pca.latent_var = simulated_fitted_plnpca._latent_var
    new_pca._components = simulated_fitted_plnpca._components
    new_pca.coef = simulated_fitted_plnpca._coef
    new_pca.compute_elbo()
