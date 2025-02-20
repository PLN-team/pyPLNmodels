# pylint: skip-file

from tests.utils import median, _get_argmax_mapping

from .conftest import dict_fitted_models


def test_median():
    i = 0
    for model_name in dict_fitted_models.keys():
        for init_method in ["formula", "explicit"]:
            for model in dict_fitted_models[model_name][init_method]:
                if model_name == "PlnMixture":
                    mapping = _get_argmax_mapping(model.cluster_bias)
                i += 1
                for param_key, param in model.dict_model_parameters.items():
                    if param is not None:
                        if param_key in ["covariances", "cluster_bias"]:
                            param = param[mapping]
                        err = median(
                            param - model.sampler.dict_model_true_parameters[param_key]
                        )
                        if err > 0.7:
                            print("param_key", param_key)
                            print("model_name", model_name)
                            print("init_method", init_method)
                            print("exog infla model", model.exog_inflation)
                            print("exog infla sampler", model.sampler.exog_inflation)
                        assert err < 0.7
