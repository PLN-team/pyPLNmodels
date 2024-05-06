import sys
import pytest
from pytest_lazyfixture import lazy_fixture as lf
import pandas as pd
import torch

from pyPLNmodels import load_model, load_plnpcacollection
from pyPLNmodels.models import Pln, PlnPCA, PlnPCAcollection, ZIPln


sys.path.append("../")


from tests.import_data import (
    data_sim_0cov,
    data_sim_2cov,
    data_real,
    data_zi_g,
    data_zi_c,
    data_zi_r,
)


endog_sim_0cov = data_sim_0cov["endog"]
exog_sim_0cov = data_sim_0cov["exog"]
offsets_sim_0cov = data_sim_0cov["offsets"]

endog_sim_2cov = data_sim_2cov["endog"]
exog_sim_2cov = data_sim_2cov["exog"]
offsets_sim_2cov = data_sim_2cov["offsets"]

endog_real = data_real["endog"]
endog_real = pd.DataFrame(endog_real)
endog_real.columns = [f"var_{i}" for i in range(endog_real.shape[1])]

endog_zi_g = data_zi_g["endog"]
exog_zi_g = data_zi_g["exog"]
exog_inflation_g = data_zi_g["exog_inflation"]
offsets_zi_g = data_zi_g["offsets"]

endog_zi_c = data_zi_c["endog"]
exog_zi_c = data_zi_c["exog"]
exog_inflation_c = data_zi_c["exog_inflation"]
offsets_zi_c = data_zi_c["offsets"]

endog_zi_r = data_zi_r["endog"]
exog_zi_r = data_zi_r["exog"]
exog_inflation_r = data_zi_r["exog_inflation"]
offsets_zi_r = data_zi_r["offsets"]


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
RANKS = [2, 6, 8]
instances = []


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


def convenientzi(*args, **kwargs):
    if isinstance(args[0], str):
        return ZIPln.from_formula(*args, **kwargs)
    return ZIPln(*args, **kwargs)


def generate_new_model(model, *args, **kwargs):
    name_dir = model._directory_name
    name = model._NAME
    if name in ("Pln", "PlnPCA", "ZIPln"):
        path = model._directory_name
        init = load_model(path)
        if name == "Pln":
            new = convenientpln(*args, **kwargs, dict_initialization=init)
        if name == "PlnPCA":
            new = convenient_PlnPCA(*args, **kwargs, dict_initialization=init)
        if name == "ZIPln":
            new = convenientzi(*args, **kwargs, dict_initialization=init)
    if name == "PlnPCAcollection":
        init = load_plnpcacollection(model._directory_name)
        new = convenient_PlnPCAcollection(*args, **kwargs, dict_initialization=init)
    return new


def cache(func):
    dict_cache = {}

    def new_func(request):
        if request.param.__name__ not in dict_cache:
            dict_cache[request.param.__name__] = func(request)
        return dict_cache[request.param.__name__]

    return new_func


params = [convenientpln, convenient_PlnPCA, convenient_PlnPCAcollection, convenientzi]
dict_fixtures = {}


def get_zi_array(formula_inflation, data):
    zi = ZIPln(
        data["endog"],
        exog=data["exog"],
        exog_inflation=data["exog_inflation"],
        offsets=data["offsets"],
        zero_inflation_formula=formula_inflation,
        add_const=False,
        add_const_inflation=False,
    )
    return zi


@pytest.fixture
def simulated_zi_global_array():
    return get_zi_array("global", data_zi_g)


@pytest.fixture
def simulated_zi_column_array():
    return get_zi_array("column-wise", data_zi_c)


@pytest.fixture
def simulated_zi_row_array():
    return get_zi_array("row-wise", data_zi_r)


sim_zi_global_instance_array = ["simulated_zi_global_array"]
sim_zi_column_instance_array = ["simulated_zi_column_array"]
sim_zi_row_instance_array = ["simulated_zi_row_array"]
sim_zi_instances_array = (
    sim_zi_global_instance_array
    + sim_zi_row_instance_array
    + sim_zi_column_instance_array
)

instances = instances + sim_zi_instances_array


@pytest.fixture
def simulated_zi_column_formula():
    zi = ZIPln.from_formula("endog~0+exog | 1", data=data_zi_c)
    return zi


@pytest.fixture
def simulated_zi_global_formula():
    zi = ZIPln.from_formula(
        "endog~0 + exog", data=data_zi_g, zero_inflation_formula="global"
    )
    return zi


@pytest.fixture
def simulated_zi_row_formula():
    zi = ZIPln.from_formula(
        "endog~0 + exog | 0 + exog_inflation ",
        data=data_zi_r,
        zero_inflation_formula="row-wise",
    )
    return zi


sim_zi_global_instance_formula = ["simulated_zi_global_formula"]
sim_zi_column_instance_formula = ["simulated_zi_column_formula"]
sim_zi_row_instance_formula = ["simulated_zi_row_formula"]

sim_zi_instances_formula = (
    sim_zi_global_instance_formula
    + sim_zi_row_instance_formula
    + sim_zi_column_instance_formula
)
instances = instances + sim_zi_instances_formula

dict_fixtures = add_list_of_fixture_to_dict(
    dict_fixtures, "sim_zi_global_instance_array", sim_zi_global_instance_array
)

dict_fixtures = add_list_of_fixture_to_dict(
    dict_fixtures, "sim_zi_row_instance_array", sim_zi_row_instance_array
)
dict_fixtures = add_list_of_fixture_to_dict(
    dict_fixtures, "sim_zi_column_instance_array", sim_zi_column_instance_array
)
dict_fixtures = add_list_of_fixture_to_dict(
    dict_fixtures, "sim_zi_row_instance_formula", sim_zi_row_instance_formula
)

dict_fixtures = add_list_of_fixture_to_dict(
    dict_fixtures, "sim_zi_column_instance_formula", sim_zi_column_instance_formula
)

dict_fixtures = add_list_of_fixture_to_dict(
    dict_fixtures, "sim_zi_column_global_formula", sim_zi_global_instance_formula
)


@pytest.fixture
def simulated_fitted_zi_global_array(simulated_zi_global_array):
    simulated_zi_global_array.fit()
    return simulated_zi_global_array


@pytest.fixture
def simulated_fitted_zi_column_array(simulated_zi_column_array):
    simulated_zi_column_array.fit()
    return simulated_zi_column_array


@pytest.fixture
def simulated_fitted_zi_row_array(simulated_zi_row_array):
    simulated_zi_row_array.fit()
    return simulated_zi_row_array


sim_zi_global_fitted_array = ["simulated_fitted_zi_global_array"]
sim_zi_row_fitted_array = ["simulated_fitted_zi_row_array"]
sim_zi_column_fitted_array = ["simulated_fitted_zi_column_array"]

simulated_zi_fitted_array = (
    sim_zi_global_fitted_array + sim_zi_row_fitted_array + sim_zi_column_fitted_array
)

dict_fixtures = add_list_of_fixture_to_dict(
    dict_fixtures, "simulated_zi_fitted_array", simulated_zi_fitted_array
)

dict_fixtures = add_list_of_fixture_to_dict(
    dict_fixtures, "sim_zi_global_fitted_array", sim_zi_global_fitted_array
)
dict_fixtures = add_list_of_fixture_to_dict(
    dict_fixtures, "sim_zi_row_fitted_array", sim_zi_row_fitted_array
)

dict_fixtures = add_list_of_fixture_to_dict(
    dict_fixtures, "sim_zi_column_fitted_array", sim_zi_column_fitted_array
)


@pytest.fixture
def simulated_fitted_zi_global_formula(simulated_zi_global_formula):
    simulated_zi_global_formula.fit()
    return simulated_zi_global_formula


@pytest.fixture
def simulated_fitted_zi_column_formula(simulated_zi_column_formula):
    simulated_zi_column_formula.fit()
    return simulated_zi_column_formula


@pytest.fixture
def simulated_fitted_zi_row_formula(simulated_zi_row_formula):
    simulated_zi_row_formula.fit()
    return simulated_zi_row_formula


sim_zi_global_fitted_formula = ["simulated_fitted_zi_global_formula"]
sim_zi_row_fitted_formula = ["simulated_fitted_zi_row_formula"]
sim_zi_column_fitted_formula = ["simulated_fitted_zi_column_formula"]

simulated_zi_fitted_formula = (
    sim_zi_global_fitted_formula
    + sim_zi_row_fitted_formula
    + sim_zi_column_fitted_formula
)

dict_fixtures = add_list_of_fixture_to_dict(
    dict_fixtures, "simulated_zi_fitted_formula", simulated_zi_fitted_formula
)

dict_fixtures = add_list_of_fixture_to_dict(
    dict_fixtures, "sim_zi_global_fitted_formula", sim_zi_global_fitted_formula
)
dict_fixtures = add_list_of_fixture_to_dict(
    dict_fixtures, "sim_zi_row_fitted_formula", sim_zi_row_fitted_formula
)

dict_fixtures = add_list_of_fixture_to_dict(
    dict_fixtures, "sim_zi_column_fitted_formula", sim_zi_column_fitted_formula
)


@pytest.fixture
def simulated_loaded_zi_global_array(simulated_fitted_zi_global_array):
    simulated_fitted_zi_global_array.save()
    return generate_new_model(
        simulated_fitted_zi_global_array,
        endog_zi_g,
        exog=exog_zi_g,
        exog_inflation=exog_inflation_g,
        offsets=offsets_zi_g,
        zero_inflation_formula="global",
        add_const=False,
        add_const_inflation=False,
    )


@pytest.fixture
def simulated_loaded_zi_column_array(simulated_fitted_zi_column_array):
    simulated_fitted_zi_column_array.save()
    return generate_new_model(
        simulated_fitted_zi_column_array,
        endog_zi_c,
        exog=exog_zi_c,
        exog_inflation=exog_inflation_c,
        offsets=offsets_zi_c,
        zero_inflation_formula="column-wise",
        add_const=False,
        add_const_inflation=False,
    )


@pytest.fixture
def simulated_loaded_zi_row_array(simulated_fitted_zi_row_array):
    simulated_fitted_zi_row_array.save()
    return generate_new_model(
        simulated_fitted_zi_row_array,
        endog_zi_r,
        exog=exog_zi_r,
        exog_inflation=exog_inflation_r,
        offsets=offsets_zi_r,
        zero_inflation_formula="row-wise",
        add_const=False,
        add_const_inflation=False,
    )


sim_zi_global_loaded_array = ["simulated_loaded_zi_global_array"]
sim_zi_row_loaded_array = ["simulated_loaded_zi_row_array"]
sim_zi_column_loaded_array = ["simulated_loaded_zi_column_array"]

simulated_zi_loaded_array = (
    sim_zi_global_loaded_array + sim_zi_row_loaded_array + sim_zi_column_loaded_array
)

dict_fixtures = add_list_of_fixture_to_dict(
    dict_fixtures, "simulated_zi_loaded_array", simulated_zi_loaded_array
)

dict_fixtures = add_list_of_fixture_to_dict(
    dict_fixtures, "sim_zi_global_loaded_array", sim_zi_global_loaded_array
)
dict_fixtures = add_list_of_fixture_to_dict(
    dict_fixtures, "sim_zi_row_loaded_array", sim_zi_row_loaded_array
)

dict_fixtures = add_list_of_fixture_to_dict(
    dict_fixtures, "sim_zi_column_loaded_array", sim_zi_column_loaded_array
)

dict_fixtures = add_list_of_fixture_to_dict(
    dict_fixtures, "sim_zi_loaded_array", simulated_zi_loaded_array
)


@pytest.fixture
def simulated_loaded_zi_global_formula(simulated_fitted_zi_global_formula):
    simulated_fitted_zi_global_formula.save()
    return generate_new_model(
        simulated_fitted_zi_global_formula,
        "endog ~ 0 + exog",
        data_zi_g,
        zero_inflation_formula="global",
    )


@pytest.fixture
def simulated_loaded_zi_column_formula(simulated_fitted_zi_column_formula):
    simulated_fitted_zi_column_formula.save()
    return generate_new_model(
        simulated_fitted_zi_column_formula,
        "endog ~ 0 + exog | 1",
        data_zi_c,
        zero_inflation_formula="column-wise",
    )


@pytest.fixture
def simulated_loaded_zi_row_formula(simulated_fitted_zi_row_formula):
    simulated_fitted_zi_row_formula.save()
    return generate_new_model(
        simulated_fitted_zi_row_formula,
        "endog ~ 0 + exog | 0 + exog_inflation",
        data_zi_r,
        zero_inflation_formula="row-wise",
    )


sim_zi_global_loaded_formula = ["simulated_loaded_zi_global_formula"]
sim_zi_row_loaded_formula = ["simulated_loaded_zi_row_formula"]
sim_zi_column_loaded_formula = ["simulated_loaded_zi_column_formula"]

simulated_zi_loaded_formula = (
    sim_zi_global_loaded_formula
    + sim_zi_row_loaded_formula
    + sim_zi_column_loaded_formula
)

dict_fixtures = add_list_of_fixture_to_dict(
    dict_fixtures, "simulated_zi_loaded_formula", simulated_zi_loaded_formula
)

dict_fixtures = add_list_of_fixture_to_dict(
    dict_fixtures, "sim_zi_global_loaded_formula", sim_zi_global_loaded_formula
)
dict_fixtures = add_list_of_fixture_to_dict(
    dict_fixtures, "sim_zi_row_loaded_formula", sim_zi_row_loaded_formula
)

dict_fixtures = add_list_of_fixture_to_dict(
    dict_fixtures, "sim_zi_column_loaded_formula", sim_zi_column_loaded_formula
)

dict_fixtures = add_list_of_fixture_to_dict(
    dict_fixtures, "sim_zi_loaded_formula", simulated_zi_loaded_formula
)


@pytest.fixture(params=params)
def simulated_model_0cov_array(request):
    cls = request.param
    model = cls(
        endog_sim_0cov,
        exog=exog_sim_0cov,
        offsets=offsets_sim_0cov,
        add_const=False,
    )
    return model


@pytest.fixture(params=params)
@cache
def simulated_fitted_model_0cov_array(request):
    cls = request.param
    model = cls(
        endog_sim_0cov,
        exog=exog_sim_0cov,
        offsets=offsets_sim_0cov,
        add_const=False,
    )
    model.fit()
    return model


@pytest.fixture(params=params)
def simulated_model_0cov_formula(request):
    cls = request.param
    model = cls("endog ~ 0", data_sim_0cov)
    return model


@pytest.fixture(params=params)
@cache
def simulated_fitted_model_0cov_formula(request):
    cls = request.param
    model = cls("endog ~ 0", data_sim_0cov)
    model.fit()
    return model


@pytest.fixture
def simulated_loaded_model_0cov_formula(simulated_fitted_model_0cov_formula):
    simulated_fitted_model_0cov_formula.save()
    return generate_new_model(
        simulated_fitted_model_0cov_formula,
        "endog ~ 0",
        data_sim_0cov,
    )


@pytest.fixture
def simulated_loaded_model_0cov_array(simulated_fitted_model_0cov_array):
    simulated_fitted_model_0cov_array.save()
    return generate_new_model(
        simulated_fitted_model_0cov_array,
        endog_sim_0cov,
        exog=exog_sim_0cov,
        offsets=offsets_sim_0cov,
        add_const=False,
    )


sim_model_0cov_instance = [
    "simulated_model_0cov_array",
    "simulated_model_0cov_formula",
]


instances = sim_model_0cov_instance + instances

dict_fixtures = add_list_of_fixture_to_dict(
    dict_fixtures, "sim_model_0cov_instance", sim_model_0cov_instance
)

sim_model_0cov_fitted = [
    "simulated_fitted_model_0cov_array",
    "simulated_fitted_model_0cov_formula",
]

dict_fixtures = add_list_of_fixture_to_dict(
    dict_fixtures, "sim_model_0cov_fitted", sim_model_0cov_fitted
)


sim_model_0cov_loaded = [
    "simulated_loaded_model_0cov_array",
    "simulated_loaded_model_0cov_formula",
]

dict_fixtures = add_list_of_fixture_to_dict(
    dict_fixtures, "sim_model_0cov_loaded", sim_model_0cov_loaded
)

sim_model_0cov = sim_model_0cov_instance + sim_model_0cov_fitted + sim_model_0cov_loaded
dict_fixtures = add_list_of_fixture_to_dict(
    dict_fixtures, "sim_model_0cov", sim_model_0cov
)

sim_model_0cov_fitted_and_loaded = sim_model_0cov_fitted + sim_model_0cov_loaded
dict_fixtures = add_list_of_fixture_to_dict(
    dict_fixtures, "sim_model_0cov_fitted_and_loaded", sim_model_0cov_fitted_and_loaded
)


@pytest.fixture(params=params)
@cache
def simulated_model_2cov_array(request):
    cls = request.param
    model = cls(
        endog_sim_2cov,
        exog=exog_sim_2cov,
        offsets=offsets_sim_2cov,
        add_const=False,
    )
    return model


@pytest.fixture
def simulated_fitted_model_2cov_array(simulated_model_2cov_array):
    simulated_model_2cov_array.fit()
    return simulated_model_2cov_array


@pytest.fixture(params=params)
@cache
def simulated_model_2cov_formula(request):
    cls = request.param
    model = cls("endog ~ 0 + exog", data_sim_2cov)
    return model


@pytest.fixture
def simulated_fitted_model_2cov_formula(simulated_model_2cov_formula):
    simulated_model_2cov_formula.fit()
    return simulated_model_2cov_formula


@pytest.fixture
def simulated_loaded_model_2cov_formula(simulated_fitted_model_2cov_formula):
    simulated_fitted_model_2cov_formula.save()
    return generate_new_model(
        simulated_fitted_model_2cov_formula,
        "endog ~0 + exog",
        data_sim_2cov,
    )


@pytest.fixture
def simulated_loaded_model_2cov_array(simulated_fitted_model_2cov_array):
    simulated_fitted_model_2cov_array.save()
    return generate_new_model(
        simulated_fitted_model_2cov_array,
        endog_sim_2cov,
        exog=exog_sim_2cov,
        offsets=offsets_sim_2cov,
        add_const=False,
    )


sim_model_2cov_instance = [
    "simulated_model_2cov_array",
    "simulated_model_2cov_formula",
]
sim_model_instance = (
    sim_model_0cov_instance + sim_model_2cov_instance + sim_zi_instances_array
)

dict_fixtures = add_list_of_fixture_to_dict(
    dict_fixtures, "sim_model_instance", sim_model_instance
)
instances = sim_model_2cov_instance + instances


dict_fixtures = add_list_of_fixture_to_dict(
    dict_fixtures, "sim_model_2cov_instance", sim_model_2cov_instance
)
sim_model_2cov_fitted = [
    "simulated_fitted_model_2cov_array",
    "simulated_fitted_model_2cov_formula",
]

dict_fixtures = add_list_of_fixture_to_dict(
    dict_fixtures, "sim_model_2cov_fitted", sim_model_2cov_fitted
)

sim_model_2cov_loaded = [
    "simulated_loaded_model_2cov_array",
    "simulated_loaded_model_2cov_formula",
]

dict_fixtures = add_list_of_fixture_to_dict(
    dict_fixtures, "sim_model_2cov_loaded", sim_model_2cov_loaded
)

sim_model_2cov = sim_model_2cov_instance + sim_model_2cov_fitted + sim_model_2cov_loaded
dict_fixtures = add_list_of_fixture_to_dict(
    dict_fixtures, "sim_model_2cov", sim_model_2cov
)


@pytest.fixture(params=params)
@cache
def real_model_intercept_array(request):
    cls = request.param
    model = cls(endog_real, add_const=True)
    return model


@pytest.fixture
def real_fitted_model_intercept_array(real_model_intercept_array):
    real_model_intercept_array.fit()
    return real_model_intercept_array


@pytest.fixture(params=params)
@cache
def real_model_intercept_formula(request):
    cls = request.param
    model = cls("endog ~ 1", data_real)
    return model


@pytest.fixture
def real_fitted_model_intercept_formula(real_model_intercept_formula):
    real_model_intercept_formula.fit()
    return real_model_intercept_formula


@pytest.fixture
def real_loaded_model_intercept_formula(real_fitted_model_intercept_formula):
    real_fitted_model_intercept_formula.save()
    return generate_new_model(
        real_fitted_model_intercept_formula, "endog ~ 1", data=data_real
    )


@pytest.fixture
def real_loaded_model_intercept_array(real_fitted_model_intercept_array):
    real_fitted_model_intercept_array.save()
    return generate_new_model(
        real_fitted_model_intercept_array,
        endog_real,
        add_const=True,
    )


real_model_instance = [
    "real_model_intercept_array",
    "real_model_intercept_formula",
]
instances = real_model_instance + instances

dict_fixtures = add_list_of_fixture_to_dict(
    dict_fixtures, "real_model_instance", real_model_instance
)

real_model_fitted = [
    "real_fitted_model_intercept_array",
    "real_fitted_model_intercept_formula",
]
dict_fixtures = add_list_of_fixture_to_dict(
    dict_fixtures, "real_model_fitted", real_model_fitted
)

real_model_loaded = [
    "real_loaded_model_intercept_array",
    "real_loaded_model_intercept_formula",
]
dict_fixtures = add_list_of_fixture_to_dict(
    dict_fixtures, "real_model_loaded", real_model_loaded
)

sim_loaded_pln_model = sim_model_0cov_loaded + sim_model_2cov_loaded
sim_fitted_pln_model = sim_model_0cov_fitted + sim_model_2cov_fitted
sim_loaded_and_fitted_pln = sim_loaded_pln_model + sim_fitted_pln_model
dict_fixtures = add_list_of_fixture_to_dict(
    dict_fixtures, "sim_loaded_and_fitted_pln", sim_loaded_and_fitted_pln
)

sim_loaded_model = (
    sim_model_0cov_loaded
    + sim_model_2cov_loaded
    + simulated_zi_loaded_array
    + simulated_zi_loaded_formula
)

loaded_model = real_model_loaded + sim_loaded_model
dict_fixtures = add_list_of_fixture_to_dict(dict_fixtures, "loaded_model", loaded_model)

simulated_model_fitted = (
    sim_model_0cov_fitted
    + sim_model_2cov_fitted
    + simulated_zi_fitted_array
    + simulated_zi_fitted_formula
)


dict_fixtures = add_list_of_fixture_to_dict(
    dict_fixtures, "simulated_model_fitted", simulated_model_fitted
)
fitted_model = real_model_fitted + simulated_model_fitted
dict_fixtures = add_list_of_fixture_to_dict(dict_fixtures, "fitted_model", fitted_model)

loaded_and_fitted_sim_model = simulated_model_fitted + sim_loaded_model
loaded_and_fitted_real_model = real_model_fitted + real_model_loaded
dict_fixtures = add_list_of_fixture_to_dict(
    dict_fixtures, "loaded_and_fitted_real_model", loaded_and_fitted_real_model
)
dict_fixtures = add_list_of_fixture_to_dict(
    dict_fixtures, "loaded_and_fitted_sim_model", loaded_and_fitted_sim_model
)
loaded_and_fitted_model = fitted_model + loaded_model
dict_fixtures = add_list_of_fixture_to_dict(
    dict_fixtures, "loaded_and_fitted_model", loaded_and_fitted_model
)

real_model = real_model_instance + real_model_fitted + real_model_loaded
dict_fixtures = add_list_of_fixture_to_dict(dict_fixtures, "real_model", real_model)

sim_model = sim_model_2cov + sim_model_0cov
dict_fixtures = add_list_of_fixture_to_dict(dict_fixtures, "sim_model", sim_model)

all_model = real_model + sim_model + instances
dict_fixtures = add_list_of_fixture_to_dict(dict_fixtures, "instances", instances)
dict_fixtures = add_list_of_fixture_to_dict(dict_fixtures, "all_model", all_model)


for string_fixture in all_model:
    print("string_fixture", string_fixture)
    dict_fixtures = add_fixture_to_dict(dict_fixtures, string_fixture)
