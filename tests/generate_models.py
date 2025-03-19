from pyPLNmodels import (
    PlnSampler,
    PlnPCASampler,
    ZIPlnSampler,
    PlnDiagSampler,
    PlnNetworkSampler,
    PlnMixtureSampler,
    ZIPlnPCASampler,
    PlnARSampler,
    PlnLDASampler,
    Pln,
    PlnPCA,
    ZIPln,
    PlnDiag,
    PlnNetwork,
    PlnMixture,
    ZIPlnPCA,
    PlnAR,
    PlnLDA,
)


from tests.utils import _get_formula_from_kw, _generate_combinations
from tests._init_functions import (
    _Pln_init,
    _PlnPCA_init,
    _ZIPln_init,
    _PlnDiag_init,
    _PlnNetwork_init,
    _ZIPlnPCA_init,
    _PlnMixture_init,
    _PlnAR_init,
    _PlnLDA_init,
    _ZIPlnPCACollection_init,
    _PlnPCACollection_init,
    _PlnMixtureCollection_init,
    _PlnNetworkCollection_init,
)

NB_COVS = [0, 2]
NB_COVS_INFLATION = [1, 2]
ADD_CONSTS = [True, False]
RANKS = [3, 5]
NB_CLUSTERS = [2, 3]
AUTOREG_TYPE = ["diagonal", "spherical", "full"]

# NB_COVS = [2]
# NB_COVS_INFLATION = [2]
# ADD_CONSTS = [True]
# RANKS = [3]
# NB_CLUSTERS = [2]
# AUTOREG_TYPE = ["diagonal", "spherical"]

DICT_COLLECTIONS_NAME = {
    "PlnPCACollection",
    "ZIPlnPCACollection",
    "PlnMixtureCollection",
    "PlnNetworkCollection",
}

DICT_SAMPLERS = {
    "Pln": PlnSampler,
    "PlnPCA": PlnPCASampler,
    "ZIPln": ZIPlnSampler,
    "PlnDiag": PlnDiagSampler,
    "PlnNetwork": PlnNetworkSampler,
    "PlnMixture": PlnMixtureSampler,
    "ZIPlnPCA": ZIPlnPCASampler,
    "PlnAR": PlnARSampler,
    "PlnLDA": PlnLDASampler,
    "ZIPlnPCACollection": ZIPlnPCASampler,
    "PlnPCACollection": PlnPCASampler,
    "PlnMixtureCollection": PlnMixtureSampler,
    "PlnNetworkCollection": PlnNetworkSampler,
}
DICT_MODELS = {
    "PlnLDA": PlnLDA,
    "Pln": Pln,
    "PlnPCA": PlnPCA,
    "ZIPln": ZIPln,
    "PlnDiag": PlnDiag,
    "PlnNetwork": PlnNetwork,
    "PlnMixture": PlnMixture,
    "ZIPlnPCA": ZIPlnPCA,
    "PlnAR": PlnAR,
}
DICT_INIT_FUNCTIONS = {
    "PlnLDA": _PlnLDA_init,
    "Pln": _Pln_init,
    "PlnPCA": _PlnPCA_init,
    "ZIPln": _ZIPln_init,
    "PlnDiag": _PlnDiag_init,
    "PlnNetwork": _PlnNetwork_init,
    "PlnMixture": _PlnMixture_init,
    "ZIPlnPCA": _ZIPlnPCA_init,
    "PlnAR": _PlnAR_init,
    "ZIPlnPCACollection": _ZIPlnPCACollection_init,
    "PlnPCACollection": _PlnPCACollection_init,
    "PlnMixtureCollection": _PlnMixtureCollection_init,
    "PlnNetworkCollection": _PlnNetworkCollection_init,
}
DICT_KWARGS = {
    "PlnLDA": {"nb_cov": NB_COVS, "add_const": [False], "n_cluster": NB_CLUSTERS},
    "Pln": {"nb_cov": NB_COVS, "add_const": ADD_CONSTS},
    "PlnPCA": {"nb_cov": NB_COVS, "add_const": ADD_CONSTS, "rank": RANKS},
    "ZIPln": {
        "add_const": ADD_CONSTS,
        "nb_cov": NB_COVS,
        "nb_cov_inflation": NB_COVS_INFLATION,
    },
    "PlnDiag": {"add_const": ADD_CONSTS, "nb_cov": NB_COVS},
    "PlnNetwork": {"add_const": ADD_CONSTS, "nb_cov": NB_COVS},
    "PlnMixture": {"nb_cov": NB_COVS, "add_const": [False], "n_cluster": NB_CLUSTERS},
    "ZIPlnPCA": {
        "nb_cov": NB_COVS,
        "add_const": ADD_CONSTS,
        "rank": [4],
        "nb_cov_inflation": NB_COVS_INFLATION,
    },
    "PlnAR": {
        "nb_cov": NB_COVS,
        "add_const": ADD_CONSTS,
        "ar_type": AUTOREG_TYPE,
    },
}
for key, values in DICT_KWARGS.items():
    DICT_KWARGS[key] = _generate_combinations(values)


def get_dict_models_unfit():
    """
    Generate Pln models instantiate either with explicit datasets
    or formula, with different number of covariates.
    """
    dict_models = {
        model_name: {"formula": [], "explicit": []} for model_name in DICT_MODELS
    }
    for model_name in DICT_MODELS:
        init_model_function = DICT_INIT_FUNCTIONS[model_name]
        for kwargs in DICT_KWARGS[model_name]:
            current_sampler = DICT_SAMPLERS[model_name](**kwargs)
            endog = current_sampler.sample()
            is_inflated = "nb_cov_inflation" in kwargs
            is_lda = model_name == "PlnLDA"
            formula = _get_formula_from_kw(kwargs, is_inflated, is_lda)
            data = {
                "endog": endog,
                "exog": current_sampler.exog_no_add,
                "offsets": current_sampler.offsets,
            }
            if is_inflated is True:
                data["exog_inflation"] = current_sampler.exog_inflation
            if is_lda is True:
                data["clusters"] = current_sampler.clusters
                data["exog"] = current_sampler.known_exog

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
            if is_lda is True:
                kwargs_explicit["clusters"] = data["clusters"]

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
            # model.fit(maxiter=10)
            model.fit()
        for model in dict_models[model_name]["explicit"]:
            # model.fit(maxiter=10)
            model.fit()
    return dict_models


def get_model(model_name, init_method, kwargs):
    """Return a fitted model."""
    sampler = DICT_SAMPLERS[model_name](**kwargs)
    endog = sampler.sample()
    is_inflated = "nb_cov_inflation" in kwargs
    is_lda = model_name == "PlnLDA"

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
        formula = _get_formula_from_kw(kwargs, is_inflated, is_lda)
        kwargs["formula"] = formula
        kwargs["data"] = data
        model = init_model_function("formula", **kwargs)
    else:
        raise ValueError("init_method should be either 'formula' or 'explicit'.")
    return model


def get_fitted_model(model_name, init_method, kwargs):
    """Return a fitted model."""
    model = get_model(model_name, init_method, kwargs)
    model.fit(maxiter=5)
    return model


def get_dict_collections_fitted():
    """
    Get all the collections models fitted.
    """
    dict_collections = {
        model_name: {"formula": [], "explicit": []}
        for model_name in DICT_COLLECTIONS_NAME
    }
    for collection_name in DICT_COLLECTIONS_NAME:
        for nb_cov in [0, 2]:
            for init_method in ["formula", "explicit"]:
                for add_const in [True, False]:
                    kwargs = {"add_const": add_const, "nb_cov": nb_cov}
                    if collection_name == "ZIPlnPCACollection":
                        kwargs["nb_cov_inflation"] = 1
                    if collection_name != "PlnMixtureCollection" or add_const is False:
                        dict_collections[collection_name][init_method].append(
                            get_fitted_model(collection_name, init_method, kwargs)
                        )
    return dict_collections
