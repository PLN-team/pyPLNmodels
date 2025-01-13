from pyPLNmodels import PlnSampler, PlnPCASampler, Pln, PlnPCA

from tests.utils import _get_formula

NB_COVS = [0, 1, 2]
DICT_SAMPLERS = {"Pln": PlnSampler, "PlnPCA": PlnPCASampler}
DICT_MODELS = {"Pln": Pln, "PlnPCA": PlnPCA}


def get_dict_models_unfit():
    """
    Generate Pln models instantiate either with explicit datasets
    or formula, with different number of covariates.
    """
    dict_models = {nb_cov: {"formula": [], "explicit": []} for nb_cov in NB_COVS}
    for nb_cov in NB_COVS:
        for sampler, model in zip(DICT_SAMPLERS.values(), DICT_MODELS.values()):
            current_sampler = sampler(nb_cov=nb_cov, use_offsets=True)
            endog = current_sampler.sample()
            formula = _get_formula(nb_cov)
            data = {
                "endog": endog,
                "exog": current_sampler.exog,
                "offsets": current_sampler.offsets,
            }
            formula_model = model.from_formula(formula=formula, data=data)
            formula_model.sampler = current_sampler
            dict_models[nb_cov]["formula"].append(formula_model)

            explicit_model = model(
                endog=endog,
                exog=current_sampler.exog,
                add_const=False,
                offsets=current_sampler.offsets,
            )
            explicit_model.sampler = current_sampler
            dict_models[nb_cov]["explicit"].append(explicit_model)
    return dict_models


def get_dict_models_fitted():
    """
    Generate (fitted) Pln models instantiate either with explicit datasets
    or formula, with different number of covariates."""
    dict_models = get_dict_models_unfit()
    for nb_cov in dict_models.keys():
        for model in dict_models[nb_cov]["formula"]:
            model.fit()
        for model in dict_models[nb_cov]["explicit"]:
            model.fit()
    return dict_models


def get_model(model_name, nb_cov, init_method):
    """Return a fitted model."""
    sampler = DICT_SAMPLERS[model_name](nb_cov=nb_cov, use_offsets=True)
    endog = sampler.sample()
    _model = DICT_MODELS[model_name]
    if init_method == "explicit":
        return _model(
            endog=endog, exog=sampler.exog, offsets=sampler.offsets, add_const=False
        )
    data = {
        "endog": endog,
        "exog": sampler.exog,
        "offsets": sampler.offsets,
    }
    return _model.from_formula("endog ~ exog", data=data)


def get_fitted_model(model_name, nb_cov, init_method):
    """Return a fitted model."""
    model = get_model(model_name, nb_cov, init_method)
    model.fit()
    return model
