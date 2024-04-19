import pytest
import matplotlib.pyplot as plt
import numpy as np

from tests.conftest import dict_fixtures
from tests.utils import MSE, filter_models

from tests.import_data import true_sim_0cov, true_sim_2cov, labels_real

single_models = ["Pln", "PlnPCA", "ZIPln"]


@pytest.mark.parametrize("any_model", dict_fixtures["loaded_and_fitted_model"])
def test_print(any_model):
    print(any_model)


@pytest.mark.parametrize("any_model", dict_fixtures["fitted_model"])
@filter_models(single_models)
def test_show_coef_transform_covariance_pcaprojected(any_model):
    any_model.show()
    any_model._criterion_args._show_loss()
    any_model._criterion_args._show_stopping_criterion()
    assert hasattr(any_model, "coef")
    assert callable(any_model.transform)
    assert hasattr(any_model, "covariance")
    assert callable(any_model.sk_PCA)
    assert any_model.sk_PCA(n_components=None) is not None
    with pytest.raises(Exception):
        any_model.sk_PCA(n_components=any_model.dim + 1)


@pytest.mark.parametrize("model", dict_fixtures["fitted_model"])
@filter_models(single_models)
def test_pca_pairplot(model):
    if model._NAME in ["Pln", "ZIPln"]:
        model.pca_pairplot(n_components=8)
    else:
        model.pca_pairplot(n_components=2)
        model.pca_pairplot()
    model.pca_pairplot(n_components=4)


@pytest.mark.parametrize("plnpca", dict_fixtures["loaded_and_fitted_model"])
@filter_models(["PlnPCAcollection"])
def test_viz_pcacol(plnpca):
    for model in plnpca.values():
        _, ax = plt.subplots()
        model.viz(ax=ax)
        plt.show()
        model.viz()
        plt.show()
        n_samples = plnpca.n_samples
        colors = np.random.randint(low=0, high=2, size=n_samples)
        model.viz(colors=colors)
        plt.show()
        model.viz(show_cov=True)
        plt.show()


@pytest.mark.parametrize("model", dict_fixtures["real_fitted_model_intercept_array"])
@filter_models(single_models)
def test_plot_pca_correlation_circle_with_names_only(model):
    model.plot_pca_correlation_circle([f"var_{i}" for i in range(8)])


@pytest.mark.parametrize("model", dict_fixtures["loaded_and_fitted_sim_model"])
@filter_models(single_models)
def test_fail_plot_pca_correlation_circle_without_names(model):
    with pytest.raises(ValueError):
        model.plot_pca_correlation_circle([f"var_{i}" for i in range(8)])
    with pytest.raises(ValueError):
        model.plot_pca_correlation_circle([f"var_{i}" for i in range(6)], [1, 2, 3])
    model.plot_pca_correlation_circle([f"var_{i}" for i in range(3)], [0, 1, 2])


@pytest.mark.parametrize("model", dict_fixtures["loaded_and_fitted_model"])
@filter_models(single_models)
def test_expected_vs_true(model):
    model.plot_expected_vs_true()
    fig, ax = plt.subplots()
    model.plot_expected_vs_true(ax=ax)


@pytest.mark.parametrize("model", dict_fixtures["real_fitted_model_intercept_array"])
@filter_models(["Pln", "PlnPCA"])
def test_expected_vs_true_real(model):
    model.plot_expected_vs_true(colors=labels_real)
    fig, ax = plt.subplots()
    model.plot_expected_vs_true(ax=ax, colors=labels_real)
