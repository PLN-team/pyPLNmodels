import torch
from pyPLNmodels import sample_pln, Pln, get_simulation_parameters
import math
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import product
import pandas as pd
import numpy as np

nb_max_iter = 800
nb_seed = 20
ns = [250, 500, 1000, 2000]
dims = [10, 20, 40]
nb_covs = [1, 2, 3, 4]
mean_gaussian = 2


def normalizing(Theta, true_Theta, hess, n):
    ## Studentization of the error
    vec_Theta = Theta.flatten()
    vec_true_Theta = true_Theta.flatten()
    C_hess = torch.linalg.cholesky(hess)
    n01 = math.sqrt(n) * torch.matmul(C_hess, vec_Theta - vec_true_Theta)
    return n01


def compute_rmse(t):
    return torch.sqrt(torch.mean(t**2))


class Fisher_Pln:
    def __init__(self, Y, A, X, d, p, n, S, Omega, Sigma):
        self.Y = Y
        self.A = A
        self.X = X
        self.d = d
        self.p = p
        self.n = n
        self.S = S
        self.Omega = Omega
        self.Sigma = Sigma

    def getDnTheta(self):
        YmA = self.Y - self.A
        YmA_outer = torch.matmul(YmA.unsqueeze(2), YmA.unsqueeze(1))
        xxt_outer = torch.matmul(self.X.unsqueeze(2), self.X.unsqueeze(1))
        res = torch.zeros(self.d * self.p, self.d * self.p)
        for i in range(self.n):
            res += torch.kron(YmA_outer[i], xxt_outer[i])
        return res / self.n

    def getMat_iCnTheta(self, i):
        a_i = self.A[i].clone().detach()
        s_i = self.S[i].clone().detach()
        d_omega = torch.diag(self.Omega)
        diag_mat_i = torch.diag(1 / a_i + s_i**4 / (1 + s_i**2 * (a_i + d_omega)))
        Sigma = self.Sigma.clone().detach()
        return torch.inverse(Sigma + diag_mat_i)

    def getCnTheta(self):
        xxt_outer = torch.clone(torch.matmul(self.X.unsqueeze(2), self.X.unsqueeze(1)))
        C_n = torch.zeros(self.d * self.p, self.d * self.p)
        for i in tqdm(range(self.n)):
            mat_i = self.getMat_iCnTheta(i)
            big_mat = torch.zeros(mat_i.shape)
            big_mat[:] = mat_i[:]
            xxt_i = xxt_outer[i].clone().detach()
            C_n += torch.kron(big_mat, xxt_i)
        return -C_n / self.n

    def getInvSandwich(self):
        Dn = self.getDnTheta()
        Cn = self.getCnTheta()
        return torch.mm(torch.mm(Cn, torch.inverse(Dn)), Cn)

    def getInvFisher(self):
        vecA = self.A.flatten()
        # X = torch.zeros((self.n,self.d)).detach().clone()
        IXt = torch.kron(torch.eye(self.p), self.X).T
        IX = torch.kron(torch.eye(self.p), self.X)
        out = torch.multiply(IXt, vecA.unsqueeze(0)) @ (IX)
        return out / (self.n)

    def get_centered_theta(self, beta):
        inv_sandwich = self.getInvSandwich()
        print("inv sandwich", inv_sandwich)
        vec_Theta = beta.flatten()
        chol = torch.linalg.cholesky(inv_sandwich)
        n_theta_1 = math.sqrt(self.n) * torch.matmul(chol, vec_Theta)
        return n_theta_1


def get_cover_from_gaussian(asymptoticGaussianVariational):
    q1MoinsAlphaSurDeux = 1.96  ## quantile
    inside = torch.abs(asymptoticGaussianVariational) < q1MoinsAlphaSurDeux
    couverture = torch.sum(inside) / len(inside)
    return couverture


def get_each_cover(ns, nb_cov, dim):
    dict_covers_sandwich = {ns[i]: [] for i in range(len(ns))}
    dict_covers_fisher = {ns[i]: [] for i in range(len(ns))}
    rmses = {
        "coef": {ns[i]: [] for i in range(len(ns))},
        "sigma": {ns[i]: [] for i in range(len(ns))},
    }
    for seed_param in range(nb_seed):
        sim_param = get_simulation_parameters(
            n_samples=ns[-1],
            dim=dim,
            seed=seed_param,
            nb_cov=nb_cov - 1,
            add_const=True,
            mean_gaussian=mean_gaussian,
        )
        sim_param._exog = torch.from_numpy(
            np.random.multinomial(1, [1 / nb_cov] * nb_cov, size=sim_param.n_samples)
        ).double()
        # sim_param._set_gaussian_mean(2)
        _endog = sample_pln(sim_param, seed=seed_param)
        _exog = sim_param.exog
        _offsets = sim_param.offsets
        true_covariance = sim_param.covariance
        true_coef = sim_param.coef
        XB = _exog @ true_coef
        print(" mean XB", torch.mean(XB))
        print("min XB", torch.min(XB))
        for i, n in enumerate(ns):
            print("n:", n)
            endog = _endog[:n]
            exog = _exog[:n]
            pln = Pln(endog, exog=exog, add_const=False)
            pln.fit(nb_max_epoch=nb_max_iter, tol=1e-8)
            rmse_sigma = compute_rmse(pln.covariance - true_covariance)
            rmse_coef = compute_rmse(pln.coef - true_coef)
            rmses["sigma"][n].append(rmse_sigma)
            rmses["coef"][n].append(rmse_coef)

            A = torch.exp(pln.offsets + pln.latent_mean + 0.5 * pln.latent_sqrt_var**2)

            test = Fisher_Pln(
                pln.endog,
                A,
                pln.exog,
                pln.nb_cov,
                pln.dim,
                pln.n_samples,
                pln.latent_sqrt_var,
                torch.inverse(pln.covariance),
                pln.covariance,
            )

            var_sandwich = test.getInvSandwich()
            var_variational = test.getInvFisher()

            N01_sandwich = normalizing(pln.coef, true_coef, var_sandwich, pln.n_samples)
            N01_fisher = normalizing(
                pln.coef, true_coef, var_variational, pln.n_samples
            )
            cover_fisher = get_cover_from_gaussian(N01_fisher)
            # sns.histplot(N01_fisher)
            # plt.title(f"Coverage {cover_fisher}")
            # plt.show()
            dict_covers_fisher[n].append(cover_fisher)
            cover_sandwich = get_cover_from_gaussian(N01_sandwich)
            dict_covers_sandwich[n].append(cover_sandwich)
    return dict_covers_sandwich, dict_covers_fisher, rmses


if __name__ == "__main__":
    method = ["Sandwich", "Variational Fisher"]
    seed_list = np.arange(nb_seed)
    params = ["coef", "sigma"]

    df_cov = pd.DataFrame(
        list(product(ns, dims, nb_covs, seed_list, method)),
        columns=["n", "p", "d", "seed", "method"],
    )
    df_cov["Coverage"] = -1

    df_rmse = pd.DataFrame(
        list(product(ns, dims, nb_covs, seed_list, params)),
        columns=["n", "p", "d", "seed", "param"],
    )
    df_rmse["RMSE"] = -1

    for dim in tqdm(dims):
        print("dim:", dim)
        for nb_cov in nb_covs:
            covers_sandwich, covers_fisher, _rmses = get_each_cover(ns, nb_cov, dim)
            rmses_coef = _rmses["coef"]
            rmses_sigma = _rmses["sigma"]
            for i in range(len(ns)):
                n = ns[i]
                sandwich = covers_sandwich[n]
                fisher = covers_fisher[n]
                for seed in seed_list:
                    df_cov["Coverage"][
                        (df_cov["n"] == n)
                        & (df_cov["p"] == dim)
                        & (df_cov["d"] == nb_cov)
                        & (df_cov["seed"] == seed)
                        & (df_cov["method"] == "Sandwich")
                    ] = sandwich[seed].item()
                    df_cov["Coverage"][
                        (df_cov["n"] == n)
                        & (df_cov["p"] == dim)
                        & (df_cov["d"] == nb_cov)
                        & (df_cov["seed"] == seed)
                        & (df_cov["method"] == "Variational Fisher")
                    ] = fisher[seed].item()

                    df_rmse["RMSE"][
                        (df_cov["n"] == n)
                        & (df_rmse["p"] == dim)
                        & (df_rmse["d"] == nb_cov)
                        & (df_rmse["seed"] == seed)
                        & (df_rmse["param"] == "sigma")
                    ] = rmses_sigma[n][seed].item()
                    df_rmse["RMSE"][
                        (df_rmse["n"] == n)
                        & (df_rmse["p"] == dim)
                        & (df_rmse["d"] == nb_cov)
                        & (df_rmse["seed"] == seed)
                        & (df_rmse["param"] == "coef")
                    ] = rmses_coef[n][seed].item()

    df_cov.to_csv("coverage.csv")
    df_rmse.to_csv("rmse.csv")
    print("df_cov", df_cov)
    print("df rmse ", df_rmse)
