# pylint: skip-file
import pytest
from pyPLNmodels.sampling._utils import _get_exog

from .conftest import dict_fitted_models
from tests.generate_models import get_fitted_model
import matplotlib.pyplot as plt


def get_methods_model(model):
    methods = []
    list_methods_with_arg = [
        ".predict()",
        ".plot_correlation_circle()",
        ".biplot()",
        ".predict_prob_inflation()",
        ".predict_clusters()",
        ".transform_new()",
        ".viz_transformed()",
        ".viz_dims()",
        ".plot_regression_forest()",
    ]
    for method in model._useful_methods_list + model._additional_methods_list:
        if method not in list_methods_with_arg:
            methods.append(method)
    return methods


def method_test(model, method, model_name):
    if method == ".pca_pairplot()" and model_name == "PlnLDA":
        with pytest.raises(NotImplementedError):
            model.pca_pairplot()
    else:
        method = method[1:-2]
        assert hasattr(model, method)
        method_to_call = getattr(model, method)
        if callable(method_to_call):
            result = method_to_call()


def test_attributes_formula_method():
    for model_name in dict_fitted_models.keys():
        for init_method in ["formula", "explicit"]:
            for model in dict_fitted_models[model_name][init_method]:
                attributes = model._useful_attributes_list
                methods = get_methods_model(model)
                for attribute in attributes:  # pylint: disable=protected-access
                    attribute = attribute[1:]
                    if model_name == "PlnMixture" and attribute == "covariance":
                        attribute = "covariances"
                    if model_name == "PlnMixture" and attribute == "precision":
                        pass
                    else:
                        assert hasattr(model, attribute)
                        attribute_value = getattr(model, attribute)
                for method in methods:  # pylint: disable=protected-access
                    method_test(model, method, model_name)
                print(model)
                plt.clf()
                plt.close("all")


def test_properties_pln():
    pln = get_fitted_model("Pln", "formula", {"nb_cov": 0, "add_const": False})
    assert pln.precision.shape == (pln.dim, pln.dim)
    assert pln.nb_cov == 0
    assert pln.latent_variance.shape == pln.latent_sqrt_variance.shape
    _proj_variables, covariances = (
        pln._pca_projected_latent_variables_with_covariances()
    )
    assert _proj_variables.shape == (pln.n_samples, 2)
    assert covariances.shape == (pln.n_samples, 2, 2)
    assert pln.projected_latent_variables(remove_exog_effect=True).shape == (
        pln.n_samples,
        2,
    )
    assert pln.get_variance_coef() is None
    pln = get_fitted_model("Pln", "formula", {"nb_cov": 60, "add_const": False})
    with pytest.raises(ValueError):
        pln.get_variance_coef()


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
            if model_name != "PlnLDA":
                launch_correlation_circle(model)
            else:
                if model._exog_clusters.shape[1] == 2:
                    with pytest.raises(ValueError):
                        model.plot_correlation_circle(
                            column_names=["A", "B"], column_index=[1, 3]
                        )
                    with pytest.raises(ValueError):
                        model.biplot(column_names=["A", "B"], column_index=[2, 4])
                else:
                    launch_correlation_circle(model)
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


def launch_correlation_circle(model):
    model.plot_correlation_circle(column_names=["A", "B"], column_index=[2, 6])
    model.biplot(column_names=["A", "B"], column_index=[2, 4])


def test_pln_diag():
    pln = get_fitted_model("PlnDiag", "formula", {"nb_cov": 1, "add_const": False})
    assert pln.precision.shape == pln.covariance.shape
    print(pln)
