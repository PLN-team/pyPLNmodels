from pyPLNmodels import (
    PlnSampler,
    PlnPCASampler,
    ZIPlnSampler,
    PlnDiagSampler,
    PlnNetworkSampler,
    Pln,
    PlnPCA,
    ZIPln,
    PlnDiag,
    PlnNetwork,
)


from tests.utils import _get_formula_from_kw, _generate_combinations
from tests._init_functions import (
    _Pln_init,
    _PlnPCA_init,
    _ZIPln_init,
    _PlnDiag_init,
    _PlnNetwork_init,
)

NB_COVS = [0, 2]
NB_COVS_INFLATION = [1, 2]
ADD_CONSTS = [True, False]
RANKS = [3, 5]

# NB_COVS = [2]
# NB_COVS_INFLATION = [2]
# ADD_CONSTS = [True]
# RANKS = [3]


DICT_SAMPLERS = {
    "Pln": PlnSampler,
    "PlnPCA": PlnPCASampler,
    "ZIPln": ZIPlnSampler,
    "PlnDiag": PlnDiagSampler,
    "PlnNetwork": PlnNetworkSampler,
}
DICT_MODELS = {
    "Pln": Pln,
    "PlnPCA": PlnPCA,
    "ZIPln": ZIPln,
    "PlnDiag": PlnDiag,
    "PlnNetwork": PlnNetwork,
}
DICT_INIT_FUNCTIONS = {
    "Pln": _Pln_init,
    "PlnPCA": _PlnPCA_init,
    "ZIPln": _ZIPln_init,
    "PlnDiag": _PlnDiag_init,
    "PlnNetwork": _PlnNetwork_init,
}
DICT_KWARGS = {
    "Pln": {"nb_cov": NB_COVS, "add_const": ADD_CONSTS},
    "PlnPCA": {"nb_cov": NB_COVS, "add_const": ADD_CONSTS, "rank": RANKS},
    "ZIPln": {
        "add_const": ADD_CONSTS,
        "nb_cov": NB_COVS,
        "nb_cov_inflation": NB_COVS_INFLATION,
    },
    "PlnDiag": {"add_const": ADD_CONSTS, "nb_cov": NB_COVS},
    "PlnNetwork": {"add_const": ADD_CONSTS, "nb_cov": NB_COVS},
}
for key, values in DICT_KWARGS.items():
    DICT_KWARGS[key] = _generate_combinations(values)


def get_dict_models_unfit():
    """
    Generate Pln models instantiate either with explicit datasets
    or formula, with different number of covariates.
    """
    dict_models = {
        model_name: {"formula": [], "explicit": []} for model_name in DICT_SAMPLERS
    }
    for model_name, init_model_function in DICT_INIT_FUNCTIONS.items():
        for kwargs in DICT_KWARGS[model_name]:
            current_sampler = DICT_SAMPLERS[model_name](**kwargs)
            endog = current_sampler.sample()
            is_inflated = "nb_cov_inflation" in kwargs
            formula = _get_formula_from_kw(kwargs, is_inflated)
            data = {
                "endog": endog,
                "exog": current_sampler.exog_no_add,
                "offsets": current_sampler.offsets,
            }
            if is_inflated is True:
                data["exog_inflation"] = current_sampler.exog_inflation

            kwargs_formula, kwargs_explicit = kwargs.copy(), kwargs.copy()
            for kw in [kwargs_formula, kwargs_explicit]:
                kw.pop("nb_cov_inflation", None)
                kw.pop("nb_cov")

            kwargs_formula["data"] = data
            kwargs_formula["formula"] = formula

            kwargs_explicit["endog"] = data["endog"]
            kwargs_explicit["exog"] = data["exog"]
            kwargs_explicit["offsets"] = data["offsets"]

            if is_inflated is True:
                kwargs_explicit["exog_inflation"] = data["exog_inflation"]

            formula_model = init_model_function("formula", **kwargs_formula)
            formula_model.sampler = current_sampler

            explicit_model = init_model_function("explicit", **kwargs_explicit)
            explicit_model.sampler = current_sampler

            dict_models[model_name]["formula"].append(formula_model)
            dict_models[model_name]["explicit"].append(explicit_model)

    return dict_models


def get_dict_models_fitted():
    """
    Generate (fitted) Pln models instantiate either with explicit datasets
    or formula, with different number of covariates."""
    dict_models = get_dict_models_unfit()
    for model_name in dict_models.keys():
        for model in dict_models[model_name]["formula"]:
            model.fit()
        for model in dict_models[model_name]["explicit"]:
            model.fit()
    return dict_models


def get_model(model_name, init_method, kwargs):
    """Return a fitted model."""
    sampler = DICT_SAMPLERS[model_name](**kwargs)
    endog = sampler.sample()
    is_inflated = "nb_cov_inflation" in kwargs

    data = {
        "endog": endog,
        "exog": sampler.exog_no_add,
        "offsets": sampler.offsets,
    }
    if is_inflated is True:
        data["exog_inflation"] = sampler.exog_inflation
    init_model_function = DICT_INIT_FUNCTIONS[model_name]

    if init_method == "explicit":
        kwargs["endog"] = data["endog"]
        kwargs["exog"] = data["exog"]
        kwargs["offsets"] = data["offsets"]
        if is_inflated is True:
            kwargs["exog_inflation"] = data["exog_inflation"]
        model = init_model_function("explicit", **kwargs)
    elif init_method == "formula":
        formula = _get_formula_from_kw(kwargs, is_inflated)
        kwargs["formula"] = formula
        kwargs["data"] = data
        model = init_model_function("formula", **kwargs)
    else:
        raise ValueError("init_method should be either 'formula' or 'explicit'.")
    return model


def get_fitted_model(model_name, init_method, kwargs):
    """Return a fitted model."""
    model = get_model(model_name, init_method, kwargs)
    model.fit()
    return model
