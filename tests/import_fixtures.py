import pytest
from pytest_lazyfixture import lazy_fixture as lf
from pyPLNmodels import load_model
from tests.import_data import (
    data_sim_0cov,
    data_sim_2cov,
    data_real,
)

counts_sim_0cov = data_sim_0cov["counts"]
covariates_sim_0cov = data_sim_0cov["covariates"]
offsets_sim_0cov = data_sim_0cov["offsets"]

counts_sim_2cov = data_sim_2cov["counts"]
covariates_sim_2cov = data_sim_2cov["covariates"]
offsets_sim_2cov = data_sim_2cov["offsets"]

counts_real = data_real["counts"]


def add_fixture_to_dict(my_dict, string_fixture):
    my_dict[string_fixture] = lf(string_fixture)
    return my_dict


def add_list_of_fixture_to_dict(
    my_dict, name_of_list_of_fixtures, list_of_string_fixtures
):
    my_dict[name_of_list_of_fixtures] = []
    for string_fixture in list_of_string_fixtures:
        my_dict[name_of_list_of_fixtures].append(lf(string_fixture))
    return my_dict


def get_dict_fixtures(PLNor_PLNPCA):
    dict_fixtures = {}

    @pytest.fixture
    def simulated_pln_0cov_array():
        pln_full = PLNor_PLNPCA(counts_sim_0cov, covariates_sim_0cov, offsets_sim_0cov)
        return pln_full

    @pytest.fixture
    def simulated_fitted_pln_0cov_array(simulated_pln_0cov_array):
        simulated_pln_0cov_array.fit()
        return simulated_pln_0cov_array

    @pytest.fixture
    def simulated_pln_0cov_formula():
        pln_full = PLNor_PLNPCA("counts ~ 0", data_sim_0cov)
        return pln_full

    @pytest.fixture
    def simulated_fitted_pln_0cov_formula(simulated_pln_0cov_formula):
        simulated_pln_0cov_array.fit()
        return simulated_pln_0cov_formula

    @pytest.fixture
    def simulated_loaded_pln_0cov_formula(simulated_fitted_pln_0cov_formula):
        simulated_fitted_pln_0cov_formula.save()
        if simulated_fitted_pln_0cov_formula.NAME == "PLN":
            init = load_model("PLN_nbcov_0")
        if simulated_fitted_pln_0cov_formula.NAME == "_PLNPCA":
            init = load_model(f"PLNPCA_rank_{simulated_fitted_pln_0cov_formula.rank}")
        new = PLNor_PLNPCA("counts ~0", data_sim_0cov, dict_initialization=init)
        return new

    @pytest.fixture
    def simulated_loaded_pln_0cov_array(simulated_fitted_pln_0cov_array):
        simulated_fitted_pln_0cov_array.save()
        if simulated_fitted_pln_0cov_array.NAME == "PLN":
            init = load_model("PLN_nbcov_0")
        if simulated_fitted_pln_0cov_array.NAME == "_PLNPCA":
            init = load_model(f"PLNPCA_rank_{simulated_fitted_pln_0cov_array.rank}")
        new = PLNor_PLNPCA(
            counts_sim_0cov,
            covariates_sim_0cov,
            offsets_sim_0cov,
            dict_initialization=init,
        )
        return new

    sim_pln_0cov_instance = [
        "simulated_pln_0cov_array",
        "simulated_pln_0cov_formula",
    ]
    dict_fixtures = add_list_of_fixture_to_dict(
        dict_fixtures, "sim_pln_0cov_instance", sim_pln_0cov_instance
    )

    sim_pln_0cov_fitted = [
        "simulated_fitted_pln_0cov_array",
        "simulated_fitted_pln_0cov_formula",
    ]

    dict_fixtures = add_list_of_fixture_to_dict(
        dict_fixtures, "sim_pln_0cov_fitted", sim_pln_0cov_fitted
    )

    sim_pln_0cov_loaded = [
        "simulated_loaded_pln_0cov_array",
        "simulated_loaded_pln_0cov_formula",
    ]

    dict_fixtures = add_list_of_fixture_to_dict(
        dict_fixtures, "sim_pln_0cov_loaded", sim_pln_0cov_loaded
    )

    sim_pln_0cov = sim_pln_0cov_instance + sim_pln_0cov_fitted + sim_pln_0cov_loaded
    dict_fixtures = add_list_of_fixture_to_dict(
        dict_fixtures, "sim_pln_0cov", sim_pln_0cov
    )

    @pytest.fixture
    def simulated_pln_2cov_array():
        pln_full = PLNor_PLNPCA(counts_sim_2cov, covariates_sim_2cov, offsets_sim_2cov)
        return pln_full

    @pytest.fixture
    def simulated_fitted_pln_2cov_array(simulated_pln_2cov_array):
        simulated_pln_2cov_array.fit()
        return simulated_pln_2cov_array

    @pytest.fixture
    def simulated_pln_2cov_formula():
        pln_full = PLNor_PLNPCA("counts ~ 0 + covariates", data_sim_2cov)
        return pln_full

    @pytest.fixture
    def simulated_fitted_pln_2cov_formula(simulated_pln_2cov_formula):
        simulated_pln_2cov_formula.fit()
        return simulated_pln_2cov_formula

    @pytest.fixture
    def simulated_loaded_pln_2cov_formula(simulated_fitted_pln_2cov_formula):
        simulated_fitted_pln_2cov_formula.save()
        if simulated_fitted_pln_2cov_formula.NAME == "PLN":
            init = load_model("PLN_nbcov_2")
        if simulated_fitted_pln_2cov_formula.NAME == "_PLNPCA":
            init = load_model(f"PLNPCA_rank_{simulated_fitted_pln_2cov_formula.rank}")
        new = PLNor_PLNPCA("counts ~2", data_sim_2cov, dict_initialization=init)
        return new

    @pytest.fixture
    def simulated_loaded_pln_2cov_array(simulated_fitted_pln_2cov_array):
        simulated_fitted_pln_2cov_array.save()
        if simulated_fitted_pln_2cov_array.NAME == "PLN":
            init = load_model("PLN_nbcov_2")
        if simulated_fitted_pln_2cov_array.NAME == "_PLNPCA":
            init = load_model(f"PLNPCA_rank_{simulated_fitted_pln_2cov_array.rank}")
        new = PLNor_PLNPCA(
            counts_sim_2cov,
            covariates_sim_2cov,
            offsets_sim_2cov,
            dict_initialization=init,
        )
        return new

    sim_pln_2cov_instance = [
        "simulated_pln_2cov_array",
        "simulated_pln_2cov_formula",
    ]

    dict_fixtures = add_list_of_fixture_to_dict(
        dict_fixtures, "sim_pln_2cov_instance", sim_pln_2cov_instance
    )

    sim_pln_2cov_fitted = [
        "simulated_fitted_pln_2cov_array",
        "simulated_fitted_pln_2cov_formula",
    ]

    dict_fixtures = add_list_of_fixture_to_dict(
        dict_fixtures, "sim_pln_2cov_fitted", sim_pln_2cov_fitted
    )

    sim_pln_2cov_loaded = [
        "simulated_loaded_pln_2cov_array",
        "simulated_loaded_pln_2cov_formula",
    ]

    dict_fixtures = add_list_of_fixture_to_dict(
        dict_fixtures, "sim_pln_2cov_loaded", sim_pln_2cov_loaded
    )

    sim_pln_2cov = sim_pln_2cov_instance + sim_pln_2cov_fitted + sim_pln_2cov_loaded
    dict_fixtures = add_list_of_fixture_to_dict(
        dict_fixtures, "sim_pln_2cov", sim_pln_2cov
    )

    @pytest.fixture
    def real_pln_intercept_array():
        pln_full = PLNor_PLNPCA(counts_real)
        return pln_full

    @pytest.fixture
    def real_fitted_pln_intercept_array(real_pln_intercept_array):
        real_pln_intercept_array.fit()
        return real_pln_intercept_array

    @pytest.fixture
    def real_pln_intercept_formula():
        pln_full = PLNor_PLNPCA("counts ~ 1", data_real)
        return pln_full

    @pytest.fixture
    def real_fitted_pln_intercept_formula(real_pln_intercept_formula):
        real_pln_intercept_formula.fit()
        return real_pln_intercept_formula

    @pytest.fixture
    def real_loaded_pln_intercept_formula(real_fitted_pln_intercept_formula):
        real_fitted_pln_intercept_formula.save()
        if real_fitted_pln_intercept_formula.NAME == "PLN":
            init = load_model("PLN_nbcov_2")
        if real_fitted_pln_intercept_formula.NAME == "_PLNPCA":
            init = load_model(f"PLNPCA_rank_{real_fitted_pln_intercept_formula.rank}")
        new = PLNor_PLNPCA("counts ~2", data_real, dict_initialization=init)
        return new

    @pytest.fixture
    def real_loaded_pln_intercept_array(real_fitted_pln_intercept_array):
        real_fitted_pln_intercept_array.save()
        if real_fitted_pln_intercept_array.NAME == "PLN":
            init = load_model("PLN_nbcov_2")
        if real_fitted_pln_intercept_array.NAME == "_PLNPCA":
            init = load_model(f"PLNPCA_rank_{real_fitted_pln_intercept_array.rank}")
        new = PLNor_PLNPCA(counts_real, dict_initialization=init)
        return new

    real_pln_instance = [
        "real_pln_intercept_array",
        "real_pln_intercept_formula",
    ]
    dict_fixtures = add_list_of_fixture_to_dict(
        dict_fixtures, "real_pln_instance", real_pln_instance
    )

    real_pln_fitted = [
        "real_fitted_pln_intercept_array",
        "real_fitted_pln_intercept_formula",
    ]
    dict_fixtures = add_list_of_fixture_to_dict(
        dict_fixtures, "real_pln_fitted", real_pln_fitted
    )

    real_pln_loaded = [
        "real_loaded_pln_intercept_array",
        "real_loaded_pln_intercept_formula",
    ]
    dict_fixtures = add_list_of_fixture_to_dict(
        dict_fixtures, "real_pln_loaded", real_pln_loaded
    )

    sim_loaded_pln = sim_pln_0cov_loaded + sim_pln_2cov_loaded

    loaded_pln = real_pln_loaded + sim_loaded_pln
    dict_fixtures = add_list_of_fixture_to_dict(dict_fixtures, "loaded_pln", loaded_pln)

    simulated_pln_fitted = sim_pln_0cov_fitted + sim_pln_2cov_fitted
    dict_fixtures = add_list_of_fixture_to_dict(
        dict_fixtures, "simulated_pln_fitted", simulated_pln_fitted
    )

    fitted_pln = real_pln_fitted + simulated_pln_fitted
    dict_fixtures = add_list_of_fixture_to_dict(dict_fixtures, "fitted_pln", fitted_pln)

    loaded_and_fitted_pln = fitted_pln + loaded_pln
    dict_fixtures = add_list_of_fixture_to_dict(
        dict_fixtures, "loaded_and_fitted_pln", loaded_and_fitted_pln
    )

    real_pln = real_pln_instance + real_pln_fitted + real_pln_loaded
    dict_fixtures = add_list_of_fixture_to_dict(dict_fixtures, "real_pln", real_pln)

    sim_pln = sim_pln_2cov + sim_pln_0cov
    dict_fixtures = add_list_of_fixture_to_dict(dict_fixtures, "sim_pln", sim_pln)

    all_pln = real_pln + sim_pln
    dict_fixtures = add_list_of_fixture_to_dict(dict_fixtures, "all_pln", all_pln)

    for string_fixture in all_pln:
        dict_fixtures = add_fixture_to_dict(dict_fixtures, string_fixture)

    return dict_fixtures
