import os
import functools

import torch
import pytest

from tests.conftest import dict_fixtures
from tests.utils import MSE

from tests.import_data import true_sim_0cov, true_sim_2cov


def filter_models(models_name):
    def decorator(my_test):
        @functools.wraps(my_test)
        def new_test(**kwargs):
            fixture = next(iter(kwargs.values()))
            if type(fixture).__name__ not in models_name:
                return None
            return my_test(**kwargs)

        return new_test

    return decorator


@pytest.mark.parametrize("any_pln", dict_fixtures["fitted_pln"])
@filter_models(["PLN", "_PLNPCA"])
def test_properties(any_pln):
    assert hasattr(any_pln, "latent_parameters")
    assert hasattr(any_pln, "latent_variables")
    assert hasattr(any_pln, "optim_parameters")
    assert hasattr(any_pln, "model_parameters")


@pytest.mark.parametrize("any_pln", dict_fixtures["loaded_and_fitted_pln"])
def test_print(any_pln):
    print(any_pln)


@pytest.mark.parametrize("any_pln", dict_fixtures["fitted_pln"])
@filter_models(["PLN", "_PLNPCA"])
def test_show_coef_transform_covariance_pcaprojected(any_pln):
    any_pln.show()
    any_pln.plotargs.show_loss(savefig=True)
    any_pln.plotargs.show_stopping_criterion(savefig=True)
    assert hasattr(any_pln, "coef")
    assert callable(any_pln.transform)
    assert hasattr(any_pln, "covariance")
    assert callable(any_pln.pca_projected_latent_variables)
    assert any_pln.pca_projected_latent_variables(n_components=None) is not None
    with pytest.raises(Exception):
        any_pln.pca_projected_latent_variables(n_components=any_pln.dim + 1)


@pytest.mark.parametrize("sim_pln", dict_fixtures["loaded_and_fitted_pln"])
@filter_models(["PLN", "_PLNPCA"])
def test_predict_simulated(sim_pln):
    if sim_pln.nb_cov == 0:
        assert sim_pln.predict() is None
        with pytest.raises(AttributeError):
            sim_pln.predict(1)
    else:
        X = torch.randn((sim_pln.n_samples, sim_pln.nb_cov))
        prediction = sim_pln.predict(X)
        expected = X @ sim_pln.coef
        assert torch.all(torch.eq(expected, prediction))


@pytest.mark.parametrize("any_instance_pln", dict_fixtures["instances"])
def test_verbose(any_instance_pln):
    any_instance_pln.fit(verbose=True, tol=0.1)


@pytest.mark.parametrize(
    "simulated_fitted_any_pln", dict_fixtures["loaded_and_fitted_sim_pln"]
)
@filter_models(["PLN", "_PLNPCA"])
def test_find_right_covariance(simulated_fitted_any_pln):
    if simulated_fitted_any_pln.nb_cov == 0:
        true_covariance = true_sim_0cov["Sigma"]
    elif simulated_fitted_any_pln.nb_cov == 2:
        true_covariance = true_sim_2cov["Sigma"]
    mse_covariance = MSE(simulated_fitted_any_pln.covariance - true_covariance)
    assert mse_covariance < 0.05


@pytest.mark.parametrize(
    "real_fitted_and_loaded_pln", dict_fixtures["loaded_and_fitted_real_pln"]
)
@filter_models(["PLN", "_PLNPCA"])
def test_right_covariance_shape(real_fitted_and_loaded_pln):
    assert real_fitted_and_loaded_pln.covariance.shape == (100, 100)


@pytest.mark.parametrize(
    "simulated_fitted_any_pln", dict_fixtures["loaded_and_fitted_pln"]
)
@filter_models(["PLN", "_PLNPCA"])
def test_find_right_coef(simulated_fitted_any_pln):
    if simulated_fitted_any_pln.nb_cov == 2:
        true_coef = true_sim_2cov["beta"]
        mse_coef = MSE(simulated_fitted_any_pln.coef - true_coef)
        assert mse_coef < 0.1
    elif simulated_fitted_any_pln.nb_cov == 0:
        assert simulated_fitted_any_pln.coef is None


@pytest.mark.parametrize("fitted_pln", dict_fixtures["fitted_pln"])
@filter_models(["PLN"])
def test_number_of_iterations_pln_full(fitted_pln):
    nb_iterations = len(fitted_pln.elbos_list)
    assert 50 < nb_iterations < 300


@pytest.mark.parametrize("fitted_pln", dict_fixtures["fitted_pln"])
@filter_models(["_PLNPCA"])
def test_number_of_iterations_plnpca(fitted_pln):
    nb_iterations = len(fitted_pln.elbos_list)
    assert 100 < nb_iterations < 5000


@pytest.mark.parametrize("pln", dict_fixtures["loaded_and_fitted_pln"])
@filter_models(["PLN"])
def test_fail_count_setter(pln):
    wrong_counts = torch.randint(size=(10, 5), low=0, high=10)
    with pytest.raises(Exception):
        pln.counts = wrong_counts


@pytest.mark.parametrize("pln", dict_fixtures["all_pln"])
@filter_models(["PLN", "PLNPCA"])
def test_setter_with_numpy(pln):
    np_counts = pln.counts.numpy()
    pln.counts = np_counts


@pytest.mark.parametrize("pln", dict_fixtures["all_pln"])
@filter_models(["PLN", "PLNPCA"])
def test_setter_with_pandas(pln):
    pd_counts = pd.DataFrame(any_pln.counts.numpy())
    any_pln.counts = pd_counts


@pytest.mark.parametrize("instance", dict_fixtures["instances"])
def test_random_init(instance):
    instance.fit(do_smart_init=False)


"""
@pytest.mark.parametrize("instance", dict_fixtures["instances"])
def test_print_end_of_fitting_message(instance):
    instance.fit(counts_sim, covariates_sim, offsets_sim, nb_max_iteration=4)


@pytest.mark.parametrize("any_pln", all_fitted_models)
@filter_models(["PLN", "_PLNPCA"])
def test_fail_wrong_covariates_prediction(any_pln):
    X = torch.randn(any_pln.n_samples, any_pln.nb_cov+1)
    with pytest.raises(Exception):
        any_pln.predict(X)


@pytest.mark.parametrize(
    "pln", dict_fixtures["loaded_and_fitted_pln"]
)
@filter_models(["PLN"])
def test_latent_var_pca(any__plnpca):
    assert any__plnpca.transform(project=False).shape == any__plnpca.counts.shape
    assert any__plnpca.transform().shape == (any__plnpca.n_samples, any__plnpca.rank)


@pytest.mark.parametrize(
    "pln", dict_fixtures["loaded_and_fitted_pln"]
)
@filter_models(["PLN"])
def test_latent_var_pln_full(any_pln_full):
    assert any_pln_full.transform().shape == any_pln_full.counts.shape


def test_wrong_rank():
    instance = _PLNPCA(counts_sim.shape[1] + 1)
    with pytest.warns(UserWarning):
        instance.fit(counts=counts_sim, covariates=covariates_sim, offsets=offsets_sim)

"""
