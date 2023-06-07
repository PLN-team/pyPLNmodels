import os

from pyPLNmodels import (
    get_simulated_count_data,
    get_real_count_data,
)


(
    counts_sim_0cov,
    covariates_sim_0cov,
    offsets_sim_0cov,
    true_covariance_0cov,
    true_coef_0cov,
) = get_simulated_count_data(return_true_param=True, nb_cov=0)
(
    counts_sim_2cov,
    covariates_sim_2cov,
    offsets_sim_2cov,
    true_covariance_2cov,
    true_coef_2cov,
) = get_simulated_count_data(return_true_param=True, nb_cov=2)

data_sim_0cov = {
    "counts": counts_sim_0cov,
    "covariates": covariates_sim_0cov,
    "offsets": offsets_sim_0cov,
}
true_sim_0cov = {"Sigma": true_covariance_0cov, "beta": true_coef_0cov}
true_sim_2cov = {"Sigma": true_covariance_2cov, "beta": true_coef_2cov}


data_sim_2cov = {
    "counts": counts_sim_2cov,
    "covariates": covariates_sim_2cov,
    "offsets": offsets_sim_2cov,
}
counts_real, labels_real = get_real_count_data(
    return_labels=True, n_samples=100, dim=50
)
data_real = {"counts": counts_real}
