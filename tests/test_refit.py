# pylint: skip-file
from tests.generate_models import get_dict_models_fitted

dict_fitted_models = get_dict_models_fitted()


def test_method_properties():
    for nb_cov in dict_fitted_models.keys():
        for model_formula, model_explicit in zip(
            dict_fitted_models[nb_cov]["formula"],
            dict_fitted_models[nb_cov]["explicit"],
        ):
            model_formula.fit()
            model_explicit.fit()
