import os
import torch

from pyPLNmodels import (
    get_pln_simulated_count_data,
    get_zipln_simulated_count_data,
    load_scrna,
)

if torch.cuda.is_available():
    DEVICE = "cuda:0"
else:
    DEVICE = "cpu"

(
    endog_sim_0cov,
    exog_sim_0cov,
    offsets_sim_0cov,
    true_covariance_0cov,
    true_coef_0cov,
) = get_pln_simulated_count_data(return_true_param=True, nb_cov=0, add_const=False)
(
    endog_sim_2cov,
    exog_sim_2cov,
    offsets_sim_2cov,
    true_covariance_2cov,
    true_coef_2cov,
) = get_pln_simulated_count_data(return_true_param=True, nb_cov=2, add_const=False)

data_sim_0cov = {
    "endog": endog_sim_0cov,
    "exog": exog_sim_0cov,
    "offsets": offsets_sim_0cov,
}
true_sim_0cov = {"Sigma": true_covariance_0cov, "beta": true_coef_0cov}
true_sim_2cov = {"Sigma": true_covariance_2cov, "beta": true_coef_2cov}


data_sim_2cov = {
    "endog": endog_sim_2cov,
    "exog": exog_sim_2cov,
    "offsets": offsets_sim_2cov,
}
endog_real, labels_real = load_scrna(
    return_labels=True, n_samples=100, dim=50, for_formula=False
)

data_real = {"endog": endog_real}


(
    endog_zi_g,
    exog_zi_g,
    exog_infla_zi_g,
    offsets_zi_g,
    cov_zi_g,
    coef_zi_g,
    coef_infla_g,
) = get_zipln_simulated_count_data(
    add_const=True,
    nb_cov=0,
    add_const_inflation=False,
    zero_inflation_formula="global",
    return_true_param=True,
)
(
    endog_zi_c,
    exog_zi_c,
    exog_zi_infla_c,
    offsets_zi_c,
    cov_zi_c,
    coef_zi_c,
    coef_infla_c,
) = get_zipln_simulated_count_data(
    add_const=False,
    nb_cov=2,
    add_const_inflation=True,
    zero_inflation_formula="column-wise",
    return_true_param=True,
)
(
    endog_zi_r,
    exog_zi_r,
    exog_infla_r,
    offsets_zi_r,
    cov_zi_r,
    coef_zi_r,
    coef_infla_r,
) = get_zipln_simulated_count_data(
    add_const=True,
    nb_cov=1,
    add_const_inflation=True,
    zero_inflation_formula="row-wise",
    return_true_param=True,
    nb_cov_inflation=2,
)


data_zi_g = {
    "endog": endog_zi_g,
    "exog": exog_zi_g,
    "exog_inflation": exog_infla_zi_g,
    "offsets": offsets_zi_g,
}

true_zi_g = {
    "Sigma": cov_zi_g,
    "coef": coef_zi_g,
    "coef_inflation": coef_infla_g,
}

data_zi_c = {
    "endog": endog_zi_c,
    "exog": exog_zi_c,
    "exog_inflation": exog_zi_infla_c,
    "offsets": offsets_zi_c,
}
true_zi_c = {
    "Sigma": cov_zi_c,
    "coef": coef_zi_c,
    "coef_inflation": coef_infla_c,
}

data_zi_r = {
    "endog": endog_zi_r,
    "exog": exog_zi_r,
    "exog_inflation": exog_infla_r,
    "offsets": offsets_zi_r,
}
true_zi_r = {
    "Sigma": cov_zi_r,
    "coef": coef_zi_r,
    "coef_inflation": coef_infla_r,
}
