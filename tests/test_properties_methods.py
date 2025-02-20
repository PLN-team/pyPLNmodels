# pylint: skip-file
import pytest
from pyPLNmodels.sampling._utils import _get_exog

from .conftest import dict_fitted_models
from tests.generate_models import get_fitted_model


def test_attributes_method():
    for model_name in dict_fitted_models.keys():
        for model in dict_fitted_models[model_name]["formula"]:
            attributes = model._useful_attributes_list
            for attribute in attributes:  # pylint: disable=protected-access
                attribute = attribute[1:]
                assert hasattr(model, attribute)
                attribute_value = getattr(model, attribute)
            methods = [
                method
                for method in model._useful_methods_list
                + model._additional_methods_list
                if method
                not in [".predict()", ".plot_correlation_circle()", ".biplot()", ".predict_prob_inflation()"]
            ]
            for method in methods:  # pylint: disable=protected-access
                method = method[1:-2]
                assert hasattr(model, method)
                method_to_call = getattr(model, method)
                if callable(method_to_call):
                    result = method_to_call()
        for model in dict_fitted_models[model_name]["explicit"]:
            attributes = (
                model._useful_attributes_list + model._additional_attributes_list
            )
            for attribute in attributes:  # pylint: disable=protected-access
                attribute = attribute[1:]
                assert hasattr(model, attribute)
                attribute_value = getattr(model, attribute)
            methods = [
                method
                for method in model._useful_methods_list
                if method
                not in [".predict()", ".plot_correlation_circle()", ".biplot()"]
            ]
            for method in methods:  # pylint: disable=protected-access
                method = method[1:-2]
                assert hasattr(model, method)
                method_to_call = getattr(model, method)
                if callable(method_to_call):
                    result = method_to_call()


def test_nb_cov_0():
    pln = get_fitted_model("Pln", "formula", {"nb_cov": 0, "add_const": False})
    assert pln.nb_cov == 0


def test_other_attributes():
    for model_name in dict_fitted_models.keys():
        for model in dict_fitted_models[model_name]["formula"]:
            exog = _get_exog(
                n_samples=model.n_samples,
                nb_cov=model.nb_cov,
                will_add_const=False,
                seed=3,
            )
            model.predict(exog)
            model.plot_correlation_circle(
                variables_names=["A", "B"], indices_of_variables=[3, 6]
            )
            model.biplot(variables_names=["A", "B"], indices_of_variables=[3, 6])
            if model.nb_cov == 0:
                exog = _get_exog(
                    n_samples=model.n_samples, nb_cov=1, will_add_const=False, seed=3
                )
                with pytest.raises(AttributeError):
                    pred = model.predict(exog)

            if model.nb_cov == 1:
                exog = _get_exog(
                    n_samples=model.n_samples, nb_cov=2, will_add_const=False, seed=3
                )
                with pytest.raises(RuntimeError):
                    pred = model.predict(exog)
                with pytest.raises(AttributeError):
                    model.predict()
