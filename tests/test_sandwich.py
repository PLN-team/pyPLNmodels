# pylint: skip-file
from .conftest import dict_fitted_models


def test_sandwich():
    for init_method in ["explicit", "formula"]:
        for model in dict_fitted_models["Pln"][init_method]:
            if model.nb_cov > 0:
                model.summary()
                coef_shape = model.coef.shape
                assert model.get_coef_p_values().shape == coef_shape
                interval_low, interval_high = model.get_confidence_interval_coef()
                true_coef = model.sampler.coef
                inside_interval = (true_coef > interval_low) & (
                    true_coef < interval_high
                )
                assert 0.99 > inside_interval.float().mean().item() > 0.75
                assert interval_low.shape == coef_shape
                assert interval_high.shape == coef_shape
                assert model.get_variance_coef().shape == coef_shape
            else:
                assert model.summary() is None
                assert model.get_coef_p_values() is None
                assert model.get_confidence_interval_coef() is None
                assert model.get_variance_coef() is None
