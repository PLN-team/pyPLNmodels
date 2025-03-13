# pylint: skip-file
import pytest
import torch
import numpy as np
import matplotlib.pyplot as plt
from pyPLNmodels.utils._viz import _viz_variables, _plot_ellipse
from pyPLNmodels import Pln, load_scrna, ZIPln, PlnPCASampler, PlnPCA
from .conftest import dict_fitted_models, dict_unfit_models


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
                _, ax = plt.subplots()
                model.viz()
                model.show(savefig=True)
                colors = np.random.randint(2, size=model.n_samples)
                model.plot_expected_vs_true(colors=colors, ax=ax)
                plt.clf()
                plt.close("all")
                plt.show()
                if model_name != "PlnLDA":
                    model.viz(show_cov=True, remove_exog_effect=True)
                else:
                    with pytest.raises(ValueError):
                        model.viz(show_cov=True)
                    with pytest.raises(ValueError):
                        model.viz(remove_exog_effect=False)
                model.viz(colors=colors)
                if model_name == "PlnLDA":
                    if model._n_cluster == 2:
                        with pytest.raises(ValueError):
                            model.biplot(column_names=["A", "B"], column_index=[3, 4])
                    else:
                        model.biplot(column_names=["A", "B"], column_index=[3, 4])
                else:
                    model.biplot(
                        column_names=["A", "B"],
                        column_index=[3, 4],
                        colors=colors,
                        title="Test",
                    )
                    model.pca_pairplot()
                    model.pca_pairplot(n_components=2, colors=colors)
                with pytest.raises(ValueError):
                    model.plot_correlation_circle(
                        column_names=["A", "B"], column_index=[1, 2, 3]
                    )
                plt.clf()
                plt.close("all")
                plt.show()


def test_show_big_matrix():
    sampler = PlnPCASampler(dim=500)
    endog = sampler.sample()
    pca = PlnPCA(endog)
    pca.fit()
    pca.show()


def test_show_no_coef():
    rna = load_scrna()
    pln = Pln(rna["endog"], exog=None, add_const=None)
    pln.fit()
    pln.show()


def test_display_norm_no_ax():
    rna = load_scrna()
    pln = Pln(rna["endog"])
    pln.fit()
    modviz = pln._get_model_viz()
    modviz.show(savefig=True, name_file="Test", figsize=(10, 12))


def test_display_norm_no_ax_zi():
    rna = load_scrna()
    pln = ZIPln(rna["endog"])
    pln.fit()
    modviz = pln._get_model_viz()
    modviz.show(savefig=True, name_file="Test", figsize=(10, 12))


def test_plot_correlation_circle_pandas():
    rna = load_scrna()
    pca = PlnPCA(rna["endog"])
    pca.fit()
    pca.show()
    pca.plot_correlation_circle(column_names=["RPL41", "ACTB"])


def test_viz_network():
    for init_method in ["explicit", "formula"]:
        for model in dict_fitted_models["PlnNetwork"][init_method]:
            model.viz_network()
            _, ax = plt.subplots()
            model.viz_network(ax=ax)
