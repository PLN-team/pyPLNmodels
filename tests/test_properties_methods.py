# pylint: skip-file
from tests.generate_models import get_dict_models_fitted

dict_models = get_dict_models_fitted()


def test_method_properties():
    for nb_cov in dict_models.keys():
        for model in dict_models[nb_cov]["formula"]:
            for (
                attribute
            ) in model._useful_properties_list:  # pylint: disable=protected-access
                attribute = attribute[1:]
                assert hasattr(model, attribute)
            for (
                method
            ) in model._useful_methods_list:  # pylint: disable=protected-access
                method = method[1:-2]
                assert hasattr(model, method)
        for model in dict_models[nb_cov]["explicit"]:
            for (
                attribute
            ) in model._useful_properties_list:  # pylint: disable=protected-access
                attribute = attribute[1:]
                assert hasattr(model, attribute)
            for (
                method
            ) in model._useful_methods_list:  # pylint: disable=protected-access
                method = method[1:-2]
                assert hasattr(model, method)
