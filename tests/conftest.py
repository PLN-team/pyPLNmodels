import sys
import glob
from functools import singledispatch
import pytest
import torch
from pytest_lazyfixture import lazy_fixture as lf
import pandas as pd

from pyPLNmodels import load_model, load_plnpcacollection
from pyPLNmodels.models import Pln, PlnPCA, PlnPCAcollection


sys.path.append("../")

pytest_plugins = [
    fixture_file.replace("/", ".").replace(".py", "")
    for fixture_file in glob.glob("src/**/tests/fixtures/[!__]*.py", recursive=True)
]


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
counts_real = pd.DataFrame(counts_real)
counts_real.columns = [f"var_{i}" for i in range(counts_real.shape[1])]


def add_fixture_to_dict(my_dict, string_fixture):
    my_dict[string_fixture] = [lf(string_fixture)]
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
instances = []
# dict_fixtures_models = []


def convenient_PlnPCA(*args, **kwargs):
    dict_init = kwargs.pop("dict_initialization", None)
    if isinstance(args[0], str):
        return PlnPCA.from_formula(
            *args, **kwargs, dict_initialization=dict_init, rank=RANK
        )
    return PlnPCA(*args, **kwargs, dict_initialization=dict_init, rank=RANK)


def convenient_PlnPCAcollection(*args, **kwargs):
    dict_init = kwargs.pop("dict_initialization", None)
    if isinstance(args[0], str):
        return PlnPCAcollection.from_formula(
            *args, **kwargs, dict_of_dict_initialization=dict_init, ranks=RANKS
        )
    return PlnPCAcollection(
        *args, **kwargs, dict_of_dict_initialization=dict_init, ranks=RANKS
    )


def convenientpln(*args, **kwargs):
    if isinstance(args[0], str):
        return Pln.from_formula(*args, **kwargs)
    return Pln(*args, **kwargs)


def generate_new_model(model, *args, **kwargs):
    name_dir = model._directory_name
    print("directory name", name_dir)
    name = model._NAME
    if name in ("Pln", "PlnPCA"):
        path = model._path_to_directory + name_dir
        init = load_model(path)
        if name == "Pln":
            new = convenientpln(*args, **kwargs, dict_initialization=init)
        if name == "PlnPCA":
            new = convenient_PlnPCA(*args, **kwargs, dict_initialization=init)
    if name == "PlnPCAcollection":
        init = load_plnpcacollection(name_dir)
        new = convenient_PlnPCAcollection(*args, **kwargs, dict_initialization=init)
    return new


def cache(func):
    dict_cache = {}

    def new_func(request):
        if request.param.__name__ not in dict_cache:
            dict_cache[request.param.__name__] = func(request)
        return dict_cache[request.param.__name__]

    return new_func


params = [convenientpln, convenient_PlnPCA, convenient_PlnPCAcollection]
dict_fixtures = {}


@pytest.fixture(params=params)
def simulated_pln_0cov_array(request):
    cls = request.param
    pln = cls(counts_sim_0cov, covariates_sim_0cov, offsets_sim_0cov, add_const=False)
    return pln


@pytest.fixture(params=params)
@cache
def simulated_fitted_pln_0cov_array(request):
    cls = request.param
    pln = cls(counts_sim_0cov, covariates_sim_0cov, offsets_sim_0cov, add_const=False)
    pln.fit()
    return pln


@pytest.fixture(params=params)
def simulated_pln_0cov_formula(request):
    cls = request.param
    pln = cls("counts ~ 0", data_sim_0cov)
    return pln


@pytest.fixture(params=params)
@cache
def simulated_fitted_pln_0cov_formula(request):
    cls = request.param
    pln = cls("counts ~ 0", data_sim_0cov)
    pln.fit()
    return pln


@pytest.fixture
def simulated_loaded_pln_0cov_formula(simulated_fitted_pln_0cov_formula):
    simulated_fitted_pln_0cov_formula.save()
    return generate_new_model(
        simulated_fitted_pln_0cov_formula,
        "counts ~ 0",
        data_sim_0cov,
    )


@pytest.fixture
def simulated_loaded_pln_0cov_array(simulated_fitted_pln_0cov_array):
    simulated_fitted_pln_0cov_array.save()
    return generate_new_model(
        simulated_fitted_pln_0cov_array,
        counts_sim_0cov,
        covariates=covariates_sim_0cov,
        offsets=offsets_sim_0cov,
        add_const=False,
    )


sim_pln_0cov_instance = [
    "simulated_pln_0cov_array",
    "simulated_pln_0cov_formula",
]

instances = sim_pln_0cov_instance + instances

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
@cache
def simulated_pln_2cov_array(request):
    cls = request.param
    pln_full = cls(
        counts_sim_2cov,
        covariates=covariates_sim_2cov,
        offsets=offsets_sim_2cov,
        add_const=False,
    )
    return pln_full


@pytest.fixture
def simulated_fitted_pln_2cov_array(simulated_pln_2cov_array):
    simulated_pln_2cov_array.fit()
    return simulated_pln_2cov_array


@pytest.fixture(params=params)
@cache
def simulated_pln_2cov_formula(request):
    cls = request.param
    pln_full = cls("counts ~ 0 + covariates", data_sim_2cov)
    return pln_full


@pytest.fixture
def simulated_fitted_pln_2cov_formula(simulated_pln_2cov_formula):
    simulated_pln_2cov_formula.fit()
    return simulated_pln_2cov_formula


@pytest.fixture
def simulated_loaded_pln_2cov_formula(simulated_fitted_pln_2cov_formula):
    simulated_fitted_pln_2cov_formula.save()
    return generate_new_model(
        simulated_fitted_pln_2cov_formula,
        "counts ~0 + covariates",
        data_sim_2cov,
    )


@pytest.fixture
def simulated_loaded_pln_2cov_array(simulated_fitted_pln_2cov_array):
    simulated_fitted_pln_2cov_array.save()
    return generate_new_model(
        simulated_fitted_pln_2cov_array,
        counts_sim_2cov,
        covariates=covariates_sim_2cov,
        offsets=offsets_sim_2cov,
        add_const=False,
    )


sim_pln_2cov_instance = [
    "simulated_pln_2cov_array",
    "simulated_pln_2cov_formula",
]
instances = sim_pln_2cov_instance + instances

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
@cache
def real_pln_intercept_array(request):
    cls = request.param
    pln_full = cls(counts_real, add_const=True)
    return pln_full


@pytest.fixture
def real_fitted_pln_intercept_array(real_pln_intercept_array):
    real_pln_intercept_array.fit()
    return real_pln_intercept_array


@pytest.fixture(params=params)
@cache
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
    return generate_new_model(
        real_fitted_pln_intercept_formula, "counts ~ 1", data=data_real
    )


@pytest.fixture
def real_loaded_pln_intercept_array(real_fitted_pln_intercept_array):
    real_fitted_pln_intercept_array.save()
    return generate_new_model(
        real_fitted_pln_intercept_array,
        counts_real,
        add_const=True,
    )


real_pln_instance = [
    "real_pln_intercept_array",
    "real_pln_intercept_formula",
]
instances = real_pln_instance + instances

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


loaded_and_fitted_sim_pln = simulated_pln_fitted + sim_loaded_pln
loaded_and_fitted_real_pln = real_pln_fitted + real_pln_loaded
dict_fixtures = add_list_of_fixture_to_dict(
    dict_fixtures, "loaded_and_fitted_real_pln", loaded_and_fitted_real_pln
)
dict_fixtures = add_list_of_fixture_to_dict(
    dict_fixtures, "loaded_and_fitted_sim_pln", loaded_and_fitted_sim_pln
)
loaded_and_fitted_pln = fitted_pln + loaded_pln
dict_fixtures = add_list_of_fixture_to_dict(
    dict_fixtures, "loaded_and_fitted_pln", loaded_and_fitted_pln
)

real_pln = real_pln_instance + real_pln_fitted + real_pln_loaded
dict_fixtures = add_list_of_fixture_to_dict(dict_fixtures, "real_pln", real_pln)

sim_pln = sim_pln_2cov + sim_pln_0cov
dict_fixtures = add_list_of_fixture_to_dict(dict_fixtures, "sim_pln", sim_pln)

all_pln = real_pln + sim_pln + instances
dict_fixtures = add_list_of_fixture_to_dict(dict_fixtures, "instances", instances)
dict_fixtures = add_list_of_fixture_to_dict(dict_fixtures, "all_pln", all_pln)


for string_fixture in all_pln:
    print("string_fixture", string_fixture)
    dict_fixtures = add_fixture_to_dict(dict_fixtures, string_fixture)
