# pylint: skip-file
from tests.generate_models import get_dict_models_fitted

dict_fitted_models = get_dict_models_fitted()


def test_method_properties():
    for model_name in dict_fitted_models.keys():
        for model_formula, model_explicit in zip(
            dict_fitted_models[model_name]["formula"],
            dict_fitted_models[model_name]["explicit"],
        ):
            model_formula.fit()
            model_explicit.fit()
