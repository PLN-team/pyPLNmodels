import sys
import pytest
from pytest_lazyfixture import lazy_fixture as lf
import pandas as pd

from pyPLNmodels import load_model, load_plnpcacollection
from pyPLNmodels.models import Pln, PlnPCA, PlnPCAcollection, ZIPln
from pyPLNmodels import get_simulated_count_data


sys.path.append("../")


from tests.import_data import (
    data_sim_0cov,
    data_sim_2cov,
    data_real,
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


def add_fixture_to_dict(my_dict, string_fixture):
    my_dict[string_fixture] = [lf(string_fixture)]
    return my_dict


# zi = ZIPln(endog_sim_2cov, exog = exog_sim_2cov)
# zi.fit()
# print(zi)


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
sim_model_instance = sim_model_0cov_instance + sim_model_2cov_instance

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

sim_loaded_model = sim_model_0cov_loaded + sim_model_2cov_loaded

loaded_model = real_model_loaded + sim_loaded_model
dict_fixtures = add_list_of_fixture_to_dict(dict_fixtures, "loaded_model", loaded_model)

simulated_model_fitted = sim_model_0cov_fitted + sim_model_2cov_fitted
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
