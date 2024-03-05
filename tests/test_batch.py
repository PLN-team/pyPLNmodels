import pytest
import torch

from tests.conftest import dict_fixtures
from tests.utils import MSE, filter_models

from pyPLNmodels import ZIPln, Pln, PlnPCA, PlnPCAcollection

from tests.import_data import true_sim_0cov, true_sim_2cov, endog_real


@pytest.mark.parametrize("model", dict_fixtures["instances"])
def test_batch(model):
    with pytest.raises(ValueError):
        model.batch_size = 20
    endog = model.endog
    exog = model.exog
    offsets = model.offsets
    if model._NAME == "ZIPln":
        instance = ZIPln
    elif model._NAME == "Pln":
        instance = Pln
    elif model._NAME == "PlnPCA":
        instance = PlnPCA
    else:
        instance = PlnPCAcollection
    new_model = instance(
        endog, exog=exog, offsets=offsets, batch_size=20, add_const=False
    )
    new_model.fit()
    if model._NAME in ["PlnPNCA", "Pln"]:
        if model.nb_cov == 2:
            true_coef = true_sim_2cov["beta"]
            mse_coef = MSE(new_model.coef - true_coef)
            assert mse_coef < 0.1
        elif model.nb_cov == 0:
            assert new_model.coef is None
    if new_model._NAME != "PlnPCAcollection":
        assert hasattr(new_model, "latent_parameters")
        assert hasattr(new_model, "latent_variables")
        assert hasattr(new_model, "optim_parameters")
        assert hasattr(new_model, "model_parameters")

        if new_model.nb_cov == 0:
            assert new_model.predict() is None
            with pytest.raises(AttributeError):
                new_model.predict(1)
        else:
            X = torch.randn((new_model.n_samples, new_model.nb_cov))
            prediction = new_model.predict(X)
            expected = X @ new_model.coef
            assert torch.all(torch.eq(expected, prediction))

        new_model.show()
        new_model._criterion_args._show_loss()
        new_model._criterion_args._show_stopping_criterion()
        assert hasattr(new_model, "coef")
        assert callable(new_model.transform)
        assert hasattr(new_model, "covariance")
        assert callable(new_model.sk_PCA)
        assert new_model.sk_PCA(n_components=None) is not None
        with pytest.raises(Exception):
            new_model.sk_PCA(n_components=new_model.dim + 1)
        if new_model._NAME in ["Pln", "ZIPln"]:
            new_model.pca_pairplot(n_components=8)
        else:
            new_model.pca_pairplot(n_components=2)
            new_model.pca_pairplot()
