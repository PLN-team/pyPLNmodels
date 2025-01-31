# pylint: skip-file
from tests.generate_models import get_dict_models_fitted

fitted_models = get_dict_models_fitted()


def test_method_properties():
    for model_name in fitted_models.keys():
        for model_formula, model_explicit in zip(
            fitted_models[model_name]["formula"],
            fitted_models[model_name]["explicit"],
        ):
            model_formula.fit()
            model_explicit.fit()
