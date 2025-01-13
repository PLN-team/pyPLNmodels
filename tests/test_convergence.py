# pylint: skip-file

from tests.utils import mse

from tests.generate_models import get_dict_models_fitted


dict_models = get_dict_models_fitted()


def test_mse():
    for nb_cov in dict_models.keys():
        for init_method in ["formula", "explicit"]:
            for model in dict_models[nb_cov][init_method]:
                for param_key, param in model.dict_model_parameters.items():
                    if param is not None:
                        err = mse(
                            param - model.sampler.dict_model_true_parameters[param_key]
                        )
                        assert err < 0.3
