import sys
import glob
from functools import singledispatch

import pytest
from pytest_lazyfixture import lazy_fixture as lf
from pyPLNmodels import load_model, load_plnpca
from tests.import_fixtures import get_dict_fixtures
from pyPLNmodels.models import PLN, _PLNPCA, PLNPCA

sys.path.append("../")


pln_full_fixture = get_dict_fixtures(PLN)
plnpca_fixture = get_dict_fixtures(_PLNPCA)


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


RANK = 8
RANKS = [2, 6]

# dict_fixtures_models = []


@singledispatch
def convenient_plnpca(
    counts,
    covariates=None,
    offsets=None,
    offsets_formula=None,
    dict_initialization=None,
):
    return _PLNPCA(
        counts, covariates, offsets, rank=RANK, dict_initialization=dict_initialization
    )


@convenient_plnpca.register(str)
def _(formula, data, offsets_formula, dict_initialization=None):
    return _PLNPCA(formula, data, rank=RANK, dict_initialization=dict_initialization)


@singledispatch
def convenientplnpca(
    counts,
    covariates=None,
    offsets=None,
    offsets_formula=None,
    dict_initialization=None,
):
    return PLNPCA(
        counts,
        covariates,
        offsets,
        offsets_formula,
        dict_of_dict_initialization=dict_initialization,
        ranks=RANKS,
    )


@convenientplnpca.register(str)
def _(formula, data, offsets_formula, dict_initialization=None):
    return PLNPCA(
        formula,
        data,
        offsets_formula,
        ranks=RANKS,
        dict_of_dict_initialization=dict_initialization,
    )


params = [PLN, convenient_plnpca, convenientplnpca]
dict_fixtures = {}


@pytest.fixture(params=params)
def simulated_pln_0cov_array(request):
    cls = request.param
    pln_full = cls(counts_sim_0cov, covariates_sim_0cov, offsets_sim_0cov)
    return pln_full


@pytest.fixture
def simulated_fitted_pln_0cov_array(simulated_pln_0cov_array):
    simulated_pln_0cov_array.fit()
    return simulated_pln_0cov_array


@pytest.fixture(params=params)
def simulated_pln_0cov_formula(request):
    cls = request.param
    pln_full = cls("counts ~ 0", data_sim_0cov)
    return pln_full


@pytest.fixture
def simulated_fitted_pln_0cov_formula(simulated_pln_0cov_formula):
    simulated_pln_0cov_formula.fit()
    return simulated_pln_0cov_formula


@pytest.fixture
def simulated_loaded_pln_0cov_formula(simulated_fitted_pln_0cov_formula):
    simulated_fitted_pln_0cov_formula.save()
    path = simulated_fitted_pln_0cov_formula.model_path
    name = simulated_fitted_pln_0cov_formula.NAME
    if name == "PLN" or name == "_PLNPCA":
        init = load_model(path)
    if name == "PLNPCA":
        init = load_plnpca(path)
    new = simulated_loaded_pln_0cov_formula.get_class(
        "counts ~0", data_sim_0cov, dict_initialization=init
    )
    return new


@pytest.fixture
def simulated_loaded_pln_0cov_array(simulated_fitted_pln_0cov_array):
    simulated_fitted_pln_0cov_array.save()
    path = simulated_fitted_pln_0cov_array.model_path
    name = simulated_fitted_pln_0cov_array.NAME
    if name == "PLN" or name == "_PLNPCA":
        init = load_model(path)
    if name == "PLNPCA":
        init = load_plnpca(path)
    new = simulated_fitted_pln_0cov_array.get_class(
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
dict_fixtures = add_list_of_fixture_to_dict(dict_fixtures, "sim_pln_0cov", sim_pln_0cov)


@pytest.fixture(params=params)
def simulated_pln_2cov_array(request):
    cls = request.param
    pln_full = cls(counts_sim_2cov, covariates_sim_2cov, offsets_sim_2cov)
    return pln_full


@pytest.fixture
def simulated_fitted_pln_2cov_array(simulated_pln_2cov_array):
    simulated_pln_2cov_array.fit()
    return simulated_pln_2cov_array


@pytest.fixture(params=params)
def simulated_pln_2cov_formula():
    pln_full = cls("counts ~ 0 + covariates", data_sim_2cov)
    return pln_full


@pytest.fixture
def simulated_fitted_pln_2cov_formula(simulated_pln_2cov_formula):
    simulated_pln_2cov_formula.fit()
    return simulated_pln_2cov_formula


@pytest.fixture
def simulated_loaded_pln_2cov_formula(simulated_fitted_pln_2cov_formula):
    simulated_fitted_pln_2cov_formula.save()
    path = simulated_fitted_pln_2cov_formula.model_path
    name = simulated_fitted_pln_2cov_formula.NAME
    if name == "PLN":
        init = load_model(path)
    if name == "PLNPCA":
        init = load_plnpca(path)
    new = simulated_fitted_pln_2cov_formula.get_class(
        "counts ~1", data_sim_2cov, dict_initialization=init
    )
    return new


@pytest.fixture
def simulated_loaded_pln_2cov_array(simulated_fitted_pln_2cov_array):
    simulated_fitted_pln_2cov_array.save()
    path = simulated_fitted_pln_2cov_array.model_path
    name = simulated_fitted_pln_2cov_array.NAME
    if name == "PLN" or name == "_PLNPCA":
        init = load_model(path)
    if name == "PLNPCA":
        init = load_model(path)
    new = simulated_fitted_pln_2cov_array.get_class(
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
dict_fixtures = add_list_of_fixture_to_dict(dict_fixtures, "sim_pln_2cov", sim_pln_2cov)


@pytest.fixture(params=params)
def real_pln_intercept_array(request):
    cls = request.param
    pln_full = cls(counts_real)
    return pln_full


@pytest.fixture
def real_fitted_pln_intercept_array(real_pln_intercept_array):
    real_pln_intercept_array.fit()
    return real_pln_intercept_array


@pytest.fixture(params=params)
def real_pln_intercept_formula(request):
    cls = request.param
    pln_full = cls("counts ~ 1", data_real)
    return pln_full


@pytest.fixture
def real_fitted_pln_intercept_formula(real_pln_intercept_formula):
    real_pln_intercept_formula.fit()
    return real_pln_intercept_formula


@pytest.fixture
def real_loaded_pln_intercept_formula(real_fitted_pln_intercept_formula):
    real_fitted_pln_intercept_formula.save()
    path = real_fitted_pln_intercept_formula.model_path
    name = real_fitted_pln_intercept_formula.NAME
    if name == "PLN" or name == "_PLNPCA":
        init = load_model(path)
    if name == "PLNPCA":
        init = load_plnpca(path)
    new = real_fitted_pln_intercept_formula.get_class(
        "counts~ 1", data_real, dict_initialization=init
    )
    return new


@pytest.fixture
def real_loaded_pln_intercept_array(real_fitted_pln_intercept_array):
    real_fitted_pln_intercept_array.save()
    path = real_fitted_pln_intercept_array.model_path
    name = real_fitted_pln_intercept_array.NAME
    if name == "PLN" or name == "_PLNPCA":
        init = load_model(path)
    if name == "PLNPCA":
        init = load_plnpca(path)
    new = real_fitted_pln_intercept_array.get_class(
        counts_real, dict_initialization=init
    )
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

pytest_plugins = [
    fixture_file.replace("/", ".").replace(".py", "")
    for fixture_file in glob.glob("src/**/tests/fixtures/[!__]*.py", recursive=True)
]
