# pylint: skip-file
import pytest

from pyPLNmodels import PlnSampler, PlnPCASampler, Pln, PlnPCA
from tests.utils import _get_formula, mse


list_nb_covs = [0, 1, 2]
list_samplers = [PlnSampler, PlnPCASampler]
list_models = [Pln, PlnPCA]


def test_mse():
    dict_sampler_model = {
        nb_cov: {"sampler": [], "model": {"formula": [], "explicit": []}}
        for nb_cov in list_nb_covs
    }
    for nb_cov in list_nb_covs:
        for sampler, model in zip(list_samplers, list_models):
            current_sampler = sampler(nb_cov=nb_cov, use_offsets=True)
            endog = current_sampler.sample()
            dict_sampler_model[nb_cov]["sampler"].append(current_sampler)
            dict_sampler_model[nb_cov]["model"]["explicit"].append(
                model(
                    endog=endog,
                    exog=current_sampler.exog,
                    add_const=False,
                    offsets=current_sampler.offsets,
                )
            )
            formula = _get_formula(nb_cov)
            data = {
                "endog": endog,
                "exog": current_sampler.exog,
                "offsets": current_sampler.offsets,
            }
            dict_sampler_model[nb_cov]["model"]["formula"].append(
                model.from_formula(formula=formula, data=data)
            )

            formula_model = dict_sampler_model[nb_cov]["model"]["formula"][-1]
            explicit_model = dict_sampler_model[nb_cov]["model"]["explicit"][-1]
            formula_model.fit()
            for param_key, param in formula_model.dict_model_parameters.items():
                if param is not None:
                    err = mse(param - current_sampler.dict_model_parameters[param_key])
                    assert err < 0.1
