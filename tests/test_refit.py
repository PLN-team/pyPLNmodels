# pylint: skip-file
from .conftest import dict_fitted_models


def test_refit_models():
    for model_name in dict_fitted_models.keys():
        for model_formula, model_explicit in zip(
            dict_fitted_models[model_name]["formula"],
            dict_fitted_models[model_name]["explicit"],
        ):
            model_formula.fit()
            model_explicit.fit(maxiter=30, verbose=True)
