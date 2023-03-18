import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
import scanpy
import numpy as np
import os


def get_simulated_data():
    Y = pd.read_csv("../example_data/test_data/Y_test.csv")
    covariates = pd.read_csv("../example_data/test_data/cov_test.csv")
    O = pd.read_csv("../example_data/test_data/O_test.csv")
    true_Sigma = torch.from_numpy(
        pd.read_csv(
            "../example_data/test_data/true_parameters/true_Sigma_test.csv"
        ).values
    )
    true_beta = torch.from_numpy(
        pd.read_csv(
            "../example_data/test_data/true_parameters/true_beta_test.csv"
        ).values
    )
    return Y, covariates, O, true_Sigma, true_beta


def get_real_data(take_oaks=True, max_class=5, max_n=200, max_dim=100):
    if take_oaks is True:
        Y = pd.read_csv("../example_data/real_data/oaks_counts.csv")
        n, p = Y.shape
        covariates = None
        O = pd.read_csv("../example_data/real_data/oaks_offsets.csv")
        return Y, covariates, O
    else:
        data = scanpy.read_h5ad(
            "../example_data/real_data/2k_cell_per_study_10studies.h5ad"
        )
        Y = data.X.toarray()[:max_n]
        GT = data.obs["standard_true_celltype_v5"][:max_n]
        le = LabelEncoder()
        GT = le.fit_transform(GT)
        filter = GT < max_class
        GT = GT[filter]
        Y = Y[filter]
        not_only_zeros = np.sum(Y, axis=0) > 0
        Y = Y[:, not_only_zeros]
        var = np.var(Y, axis=0)
        most_variables = np.argsort(var)[-max_dim:]
        Y = Y[:, most_variables]
        return Y, GT


def MSE(t):
    return torch.mean(t**2)
