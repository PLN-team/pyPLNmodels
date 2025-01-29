# pylint: skip-file

from tests.utils import mse

from .conftest import dict_fitted_models


def test_mse():
    i = 0
    for model_name in dict_fitted_models.keys():
        for init_method in ["formula", "explicit"]:
            for model in dict_fitted_models[model_name][init_method]:
                i += 1
                for param_key, param in model.dict_model_parameters.items():
                    if param is not None:
                        err = mse(
                            param - model.sampler.dict_model_true_parameters[param_key]
                        )
                        if err < 0.3:
                            print("param_key", param_key)
                            print("model_name", model_name)
                            print("init_method", init_method)
                            assert err < 0.3
