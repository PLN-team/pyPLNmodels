# pylint: skip-file
import pytest
import torch
import numpy as np
from pyPLNmodels import ZIPln, load_microcosm
import matplotlib.pyplot as plt

from .conftest import dict_fitted_models
from .generate_models import get_model


def test_zi():
    for init_method in ["explicit", "formula"]:
        for model in dict_fitted_models["ZIPln"][init_method]:
            model.pca_pairplot_prob()
            model.pca_pairplot_prob(n_components=3)
            assert model.coef_inflation.shape == (model.nb_cov_inflation, model.dim)
            assert model.latent_prob.shape == (model.n_samples, model.dim)
            assert model.exog_inflation.shape == (
                model.n_samples,
                model.nb_cov_inflation,
            )
            if model.exog is not None:
                assert model.exog.shape == (model.n_samples, model.nb_cov)
            arr_infl = torch.randn(50, model.nb_cov_inflation)

            assert torch.allclose(
                model.predict_prob_inflation(arr_infl),
                torch.sigmoid(arr_infl @ model.coef_inflation),
            )
            with pytest.raises(RuntimeError):
                model.predict_prob_inflation(
                    np.random.randn(10, model.nb_cov_inflation + 1)
                )
            model.show(savefig=True)
            assert torch.allclose(model.latent_prob_variables, model.latent_prob)
