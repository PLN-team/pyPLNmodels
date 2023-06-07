import pytest

from tests.conftest import dict_fixtures
from tests.utils import MSE, filter_models

from tests.import_data import true_sim_0cov, true_sim_2cov, labels_real


@pytest.mark.parametrize("any_pln", dict_fixtures["loaded_and_fitted_pln"])
def test_print(any_pln):
    print(any_pln)


@pytest.mark.parametrize("any_pln", dict_fixtures["fitted_pln"])
@filter_models(["Pln", "PlnPCA"])
def test_show_coef_transform_covariance_pcaprojected(any_pln):
    any_pln.show()
    any_pln._plotargs._show_loss()
    any_pln._plotargs._show_stopping_criterion()
    assert hasattr(any_pln, "coef")
    assert callable(any_pln.transform)
    assert hasattr(any_pln, "covariance")
    assert callable(any_pln.sk_PCA)
    assert any_pln.sk_PCA(n_components=None) is not None
    with pytest.raises(Exception):
        any_pln.sk_PCA(n_components=any_pln.dim + 1)


@pytest.mark.parametrize("pln", dict_fixtures["fitted_pln"])
@filter_models(["Pln"])
def test_scatter_pca_matrix_pln(pln):
    pln.scatter_pca_matrix(n_components=8)


@pytest.mark.parametrize("pln", dict_fixtures["fitted_pln"])
@filter_models(["PlnPCA"])
def test_scatter_pca_matrix_plnpca(pln):
    pln.scatter_pca_matrix(n_components=2)
    pln.scatter_pca_matrix()


@pytest.mark.parametrize("pln", dict_fixtures["loaded_and_fitted_real_pln"])
@filter_models(["Pln", "PlnPCA"])
def test_label_scatter_pca_matrix(pln):
    pln.scatter_pca_matrix(n_components=4, color=labels_real)
