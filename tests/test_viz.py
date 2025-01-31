# pylint: skip-file
import pytest
import torch
import numpy as np
import matplotlib.pyplot as plt
from pyPLNmodels._viz import _viz_variables, _plot_ellipse
from pyPLNmodels import Pln, load_scrna, ZIPln, PlnPCASampler, PlnPCA
from .conftest import dict_fitted_models, dict_unfit_models

data = load_scrna()


@pytest.fixture
def pca_projected_variables():
    return torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])


@pytest.fixture
def covariances():
    return torch.tensor(
        [[[0.5, 0.2], [0.2, 0.5]], [[0.3, 0.1], [0.1, 0.3]], [[0.4, 0.2], [0.2, 0.4]]]
    )


@pytest.fixture
def colors():
    return np.array(["red", "green", "blue"])


def test_viz_variables_no_covariances(pca_projected_variables):
    fig, ax = plt.subplots()
    result_ax = _viz_variables(pca_projected_variables, ax=ax)
    assert result_ax is not None
    assert len(result_ax.collections) > 0  # Check if scatter plot is created


def test_viz_variables_with_covariances(pca_projected_variables, covariances):
    fig, ax = plt.subplots()
    result_ax = _viz_variables(pca_projected_variables, ax=ax, covariances=covariances)
    assert result_ax is not None
    assert len(result_ax.collections) > 0  # Check if scatter plot is created
    assert len(result_ax.patches) == len(covariances)  # Check if ellipses are created


def test_viz_variables_with_colors(pca_projected_variables, colors):
    fig, ax = plt.subplots()
    result_ax = _viz_variables(pca_projected_variables, ax=ax, colors=colors)
    assert result_ax is not None
    assert len(result_ax.collections) > 0  # Check if scatter plot is created
    assert len(result_ax.collections[0].get_facecolors()) == len(
        colors
    )  # Check if colors are applied


def test_viz_variables_without_axis(pca_projected_variables):
    result_ax = _viz_variables(pca_projected_variables)
    assert result_ax is not None
    assert len(result_ax.collections) > 0  # Check if scatter plot is created


def test_plot_ellipse():
    fig, ax = plt.subplots()
    mean_x, mean_y = 1.0, 2.0
    cov = np.array([[0.5, 0.2], [0.2, 0.5]])
    _plot_ellipse(mean_x, mean_y, cov=cov, ax=ax)
    assert len(ax.patches) == 1  # Check if ellipse is created


def test_viz_general():
    for model_name in dict_fitted_models:
        for init_method in ["formula", "explicit"]:
            for model in dict_fitted_models[model_name][init_method]:
                with pytest.raises(ValueError):
                    model.plot_correlation_circle(
                        variables_names=None, indices_of_variables=None
                    )
                # _, ax = plt.subplots()
                # model.plot_expected_vs_true(ax=ax)
                # colors = torch.randn(model.n_samples)
                # model.viz(colors=colors)
                # model.viz(colors=colors, show_cov=True)
                # model.viz(show_cov=True)
                # model.viz(show_cov=True, remove_exog_effect=True)
                # model.viz(remove_exog_effect=True)
                # model.viz()

                # model.biplot(variables_names=["A", "B"], indices_of_variables=[3, 4])
                # model.biplot(
                #     variables_names=["A", "B"],
                #     indices_of_variables=[3, 4],
                #     colors=colors,
                # )
                # model.biplot(
                #     variables_names=["A", "B"],
                #     indices_of_variables=[3, 4],
                #     colors=colors,
                #     title="Test",
                # )
                # model.pca_pairplot()
                # model.pca_pairplot(n_components=2)
                # model.pca_pairplot(n_components=2, colors=colors)
                # model.show()
                # with pytest.raises(ValueError):
                #     model.plot_correlation_circle(
                #         variables_names=["A", "B"], indices_of_variables=[1, 2, 3]
                #     )


def test_show_big_matrix():
    sampler = PlnPCASampler(dim=500)
    endog = sampler.sample()
    pca = PlnPCA(endog)
    pca.fit()
    pca.show()


def test_show_no_coef():
    pln = Pln(data["endog"], exog=None, add_const=None)
    pln.fit()
    pln.show()


def test_display_norm_no_ax():
    pln = Pln(data["endog"])
    pln.fit()
    modviz = pln._get_model_viz()
    _, axes_5 = plt.subplots(5, 1)
    modviz.show(axes=axes_5, savefig=True, name_file="Test")
    _, axes_3 = plt.subplots(3, 1)
    with pytest.raises(IndexError):
        modviz.show(axes=axes_3, savefig=False, name_file="Test")


def test_display_norm_no_ax_zi():
    pln = ZIPln(data["endog"])
    pln.fit()
    modviz = pln._get_model_viz()
    _, axes_5 = plt.subplots(6, 1)
    modviz.show(axes=axes_5, savefig=True, name_file="Test")
    _, axes_3 = plt.subplots(3, 1)
    with pytest.raises(IndexError):
        modviz.show(axes=axes_3, savefig=False, name_file="Test")


def test_plot_correlation_circle_pandas():
    pca = PlnPCA(data["endog"])
    pca.fit()
    pca.show()
    pca.plot_correlation_circle(variables_names=["RPL14", "ACTB"])
