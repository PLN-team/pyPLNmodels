import time
from abc import ABC, abstractmethod

import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from ._closed_forms import closed_formula_beta, closed_formula_Sigma, closed_formula_pi
from .elbos import ELBOPLN, ELBOPLNPCA, ELBOZIPLN, profiledELBOPLN
from ._utils import (
    PLNPlotArgs,
    init_Sigma,
    init_C,
    init_beta,
    get_O_from_sum_of_Y,
    check_dimensions_are_equal,
    init_M,
    init_S,
    NotFitError,
)

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print("device :", device)
# shoudl add a good init for M. for plnpca we should not put the maximum of the log posterior, for plnpca it may be ok.


class _PLN(ABC):
    """
    Virtual class for all the PLN models.

    This class must be derivatived. The methods `get_Sigma`, `compute_ELBO`,
    `random_init_var_parameters` and `list_of_parameters_needing_gradient` must
    be defined.
    """

    def __init__(self):
        """
        Simple initialization method.
        """
        self.window = 3
        self.fitted = False
        self.plotargs = PLNPlotArgs(self.window)

    def format_datas(self, Y, covariates, O, O_formula):
        self.Y = self.format_data(Y)
        if covariates is None:
            self.covariates = torch.full((self.Y.shape[0], 1), 1).float()
        else:
            self.covariates = self.format_data(covariates)
        if O is None:
            if O_formula == "sum":
                self.O = torch.log(get_O_from_sum_of_Y(self.Y)).float()
            else:
                self.O = torch.zeros(self.Y.shape)
        else:
            self.O = self.format_data(O)

    def smart_init_beta(self):
        self._beta = init_beta(self.Y, self.covariates, self.O)

    def random_init_beta(self):
        self._beta = torch.randn((self.d, self.p), device=device)

    @abstractmethod
    def random_init_model_parameters(self):
        pass

    @abstractmethod
    def smart_init_model_parameters(self):
        pass

    @abstractmethod
    def random_init_var_parameters(self):
        pass

    def format_data(self, data):
        if isinstance(data, pd.DataFrame):
            return torch.from_numpy(data.values).float().to(device)
        if isinstance(data, np.ndarray):
            return torch.from_numpy(data).float().to(device)
        if isinstance(data, torch.Tensor):
            return data
        else:
            raise AttributeError(
                "Please insert either a numpy array, pandas.DataFrame or torch.tensor"
            )

    def smart_init_var_parameters(self):
        pass

    def init_parameters(self, Y, covariates, O, do_smart_init):
        self.n, self.p = self.Y.shape
        self.d = self.covariates.shape[1]
        print("Initialization ...")
        if do_smart_init:
            self.smart_init_model_parameters()
            self.smart_init_var_parameters()
        else:
            self.random_init_model_parameters()
            self.random_init_var_parameters()
        print("Initialization finished")
        self.putParametersToDevice()

    def putParametersToDevice(self):
        for parameter in self.list_of_parameters_needing_gradient:
            parameter.requires_grad_(True)

    @property
    def list_of_parameters_needing_gradient(self):
        """
        A list containing all the parameters that needs to be upgraded via a gradient step.
        """
        pass

    def check_parameters_shape(self):
        nY, pY = self.Y.shape
        nO, pO = self.O.shape
        nCov, _ = self.covariates.shape
        check_dimensions_are_equal("Y", "O", nY, nO, 0)
        check_dimensions_are_equal("Y", "covariates", nY, nCov, 0)
        check_dimensions_are_equal("Y", "O", pY, pO, 1)

    def fit(
        self,
        Y,
        covariates=None,
        O=None,
        nb_max_iteration=15000,
        lr=0.01,
        class_optimizer=torch.optim.Rprop,
        tol=1e-3,
        do_smart_init=True,
        verbose=False,
        O_formula="sum",
    ):
        """
        Main function of the class. Fit a PLN to the data.
        Parameters
        ----------
        Y : torch.tensor or ndarray or DataFrame.
            2-d count data.
        covariates : torch.tensor or ndarray or DataFrame or None, default = None
            If not `None`, the first dimension should equal the first dimension of `Y`.
        O : torch.tensor or ndarray or DataFrame or None, default = None
            Model offset. If not `None`, size should be the same as `Y`.
        """
        self.t0 = time.time()
        if self.fitted is False:
            self.format_datas(Y, covariates, O, O_formula)
            self.check_parameters_shape()
            self.init_parameters(Y, covariates, O, do_smart_init)
            print("ELBO after init:", self.compute_ELBO())
        else:
            self.t0 -= self.plotargs.running_times[-1]
        self.optim = class_optimizer(self.list_of_parameters_needing_gradient, lr=lr)
        nb_iteration_done = 0
        stop_condition = False
        while nb_iteration_done < nb_max_iteration and stop_condition == False:
            nb_iteration_done += 1
            loss = self.trainstep()
            criterion = self.compute_criterion_and_update_plotargs(loss, tol)
            if abs(criterion) < tol:
                stop_condition = True
            if verbose:
                self.print_stats()
        self.print_end_of_fitting_message(stop_condition, tol)
        self.fitted = True

    def trainstep(self):
        """
        simple docstrings with black errors
        """
        self.optim.zero_grad()
        loss = -self.compute_ELBO()
        loss.backward()
        self.optim.step()
        self.update_closed_forms()
        return loss

    def print_end_of_fitting_message(self, stop_condition, tol):
        if stop_condition:
            print(
                f"Tolerance {tol} reached in {self.plotargs.iteration_number} iterations"
            )
        else:
            print(
                "Maximum number of iterations reached : ",
                self.plotargs.iteration_number,
                "last criterion = ",
                np.round(self.plotargs.criterions[-1], 8),
            )

    def print_stats(self):
        print("-------UPDATE-------")
        print("Iteration number: ", self.plotargs.iteration_number)
        print("Criterion: ", np.round(self.plotargs.criterions[-1], 8))
        print("ELBO:", np.round(self.plotargs.ELBOs_list[-1], 6))

    def compute_criterion_and_update_plotargs(self, loss, tol):
        self.plotargs.ELBOs_list.append(-loss.item() / self.n)
        self.plotargs.running_times.append(time.time() - self.t0)
        if self.plotargs.iteration_number > self.window:
            criterion = abs(
                self.plotargs.ELBOs_list[-1]
                - self.plotargs.ELBOs_list[-1 - self.window]
            )
            self.plotargs.criterions.append(criterion)
            return criterion
        return tol

    def update_closed_forms(self):
        pass

    @abstractmethod
    def compute_ELBO(self):
        """
        Compute the Evidence Lower BOund (ELBO) that will be maximized by pytorch.
        """
        pass

    def display_Sigma(self, ax=None, savefig=False, name_file=""):
        """
        Display a heatmap of Sigma to visualize correlations.

        If Sigma is too big (size is > 400), will only display the first block
        of size (400,400).

        Parameters
        ----------
        ax : matplotlib Axes, optional
            Axes in which to draw the plot, otherwise use the currently-active Axes.
        savefig: bool, optional
            If True the figure will be saved. Default is False.
        name_file : str, optional
            The name of the file the graphic will be saved to if saved.
            Default is an empty string.
        """
        sigma = self._Sigma()
        if self.p > 400:
            sigma = sigma[:400, :400]
        sns.heatmap(sigma, ax=ax)
        if savefig:
            plt.savefig(name_file + self.NAME)
        plt.close()  # to avoid displaying a blanck screen

    def __str__(self):
        string = "A multivariate Poisson Lognormal with " + self.DESCRIPTION + "\n"
        string += "Best likelihood:" + str(np.max(-self.plotargs.ELBOs_list[-1])) + "\n"
        return string

    def show(self):
        print("Best likelihood:", np.max(-self.plotargs.ELBOs_list[-1]))
        fig, axes = plt.subplots(1, 3, figsize=(23, 5))
        self.plotargs.show_loss(ax=axes[0])
        self.plotargs.show_stopping_criterion(ax=axes[1])
        self.display_Sigma(ax=axes[2])
        plt.show()
        return ""

    @property
    def ELBOs_list(self):
        return self.plotargs.ELBOs_list

    @property
    def loglike(self):
        if self.fitted is False:
            raise NotFitError()
        return self.ELBOs_list[-1]

    @property
    def BIC(self):
        return -2 * self.loglike + self.number_of_parameters * np.log(self.n)

    @property
    def number_of_parameters(self):
        return self.p * (self.d + (self.p + 1) / 2)

    @property
    def var_parameters(self):
        return {"S": self._S, "M": self._M}

    @property
    def model_parameters(self):
        return {"Beta": self._beta, "Sigma": self._Sigma}


class PLN(_PLN):
    NAME = "PLN"
    DESCRIPTION = "full covariance model."

    def random_init_var_parameters(self):
        self._S = 1 / 2 * torch.ones((self.n, self.p)).to(device)
        self._M = torch.ones((self.n, self.p)).to(device)

    @property
    def list_of_parameters_needing_gradient(self):
        return [self._M, self._S]

    def compute_ELBO(self):
        """
        Compute the Evidence Lower BOund (ELBO) that will be maximized by pytorch. Here we use the profiled ELBO
        for the full covariance matrix.
        """
        return profiledELBOPLN(self.Y, self.covariates, self.O, self._M, self._S)

    def smart_init_model_parameters(self):
        pass

    def random_init_model_parameters(self):
        pass

    @property
    def beta(self):
        return closed_formula_beta(self.covariates, self._M).detach().cpu()

    @property
    def Sigma(self):
        return (
            closed_formula_Sigma(
                self.covariates, self._M, self._S, self.get_beta(), self.n
            )
            .detach()
            .cpu()
        )

    @property
    def Sigma(self):
        return self._Sigma

    @property
    def beta(self):
        return self._beta

    @property
    def M(self):
        return self._M

    @property
    def S(self):
        return self._S


class PLNPCA:
    def __init__(self, ranks):
        if isinstance(ranks, list):
            self.ranks = ranks
            self.list_PLNPCA = {}
            for rank in ranks:
                if isinstance(rank, int):
                    self.list_PLNPCA[rank] = _PLNPCA(rank)
                else:
                    TypeError("Please instantiate with either a list of integers.")
        elif isinstance(ranks, int):
            self.list_PLNPCA = {ranks: _PLNPCA(ranks)}
        else:
            raise TypeError(
                "Please instantiate with either a list of integer or an integer"
            )

    def fit(
        self,
        Y,
        covariates=None,
        O=None,
        nb_max_iteration=15000,
        lr=0.01,
        class_optimizer=torch.optim.Rprop,
        tol=1e-3,
        do_smart_init=True,
        verbose=False,
        O_formula="sum",
    ):
        for pca in self.list_PLNPCA.values():
            pca.fit(
                Y,
                covariates,
                O,
                nb_max_iteration,
                lr,
                class_optimizer,
                tol,
                do_smart_init,
                verbose,
                O_formula,
            )

    def get_model():
        pass


class _PLNPCA(_PLN):
    NAME = "PLNPCA"
    DESCRIPTION = " with Principal Component Analysis."

    def __init__(self, q):
        super().__init__()
        self._q = _q

    @property
    def model_parameters(self):
        betaAndSigma = super().model_parameters
        betaAndSigma["C"] = self._C

    def smart_init_model_parameters(self):
        t0 = time.time()
        super().smart_init_beta()
        t0 = time.time()
        self._C = init_C(self.Y, self.covariates, self.O, self._beta, self.q)

    def random_init_model_parameters(self):
        super().random_init_beta()
        self._C = torch.randn((self.d, self.q)).to(device)

    def smart_init_var_parameters(self):
        self._M = init_M(self.Y, self.covariates, self.O, self._beta, self._C)
        self._S = init_S(self.Y, self.covariates, self.O, self._beta, self._C, self._M)

    def random_init_var_parameters(self):
        self._S = 1 / 2 * torch.ones((self.n, self.q)).to(device)
        self._M = torch.ones((self.n, self.q)).to(device)

    def smart_init_var_parameters(self):
        self._M = (
            init_M(self.Y, self.covariates, self.O, self._beta, self._C)
            .to(device)
            .detach()
        )
        self._S = (
            init_S(self._M, self.covariates, self.O, self._beta, self._C, self._M)
            .to(device)
            .detach()
        )
        self._M.requires_grad_(True)
        self._S.requires_grad_(True)

    @property
    def list_of_parameters_needing_gradient(self):
        return [self._C, self._beta, self._M, self._S]

    def compute_ELBO(self):
        return ELBOPLNPCA(
            self.Y, self.covariates, self.O, self._M, self._S, self._C, self._beta
        )

    def get_Sigma(self):
        return (self._C @ (self._C.T)).detach().cpu()

    def get_beta(self):
        return self._beta.detach().cpu()

    @property
    def number_of_parameters(self):
        return self.p * (self.d + self.q) - self.q * (self.q - 1) / 2


class ZIPLN(PLN):
    NAME = "ZIPLN"
    DESCRIPTION = "with full covariance model and zero-inflation."

    def random_init_model_parameters(self):
        super().random_init_model_parameters()
        self.Theta_zero = torch.randn(self.d, self.p)
        self._Sigma = torch.diag(torch.ones(self.p)).to(device)

    # should change the good initialization, especially for Theta_zero
    def smart_init_model_parameters(self):
        super().smart_init_model_parameters()
        self._Sigma = init_Sigma(self.Y, self.covariates, self.O, self._beta)
        self._Theta_zero = torch.randn(self.d, self.p)

    def random_init_var_parameters(self):
        self.dirac = self.Y == 0
        self._M = torch.randn(self.n, self.p)
        self._S = torch.randn(self.n, self.p)
        self.pi = torch.empty(self.n, self.p).uniform_(0, 1).to(device) * self.dirac

    def compute_ELBO(self):
        return ELBOZIPLN(
            self.Y,
            self.covariates,
            self.O,
            self._M,
            self._S,
            self.pi,
            self._Sigma,
            self._beta,
            self.Theta_zero,
            self.dirac,
        )

    def get_Sigma(self):
        return self._Sigma.detach().cpu()

    @property
    def list_of_parameters_needing_gradient(self):
        return [self._M, self._S, self._Theta_zero]

    def update_closed_forms(self):
        self._beta = closed_formula_beta(self.covariates, self._M)
        self._Sigma = closed_formula_Sigma(
            self.covariates, self._M, self._S, self._beta, self._n
        )
        self.pi = closed_formula_pi(
            self.O, self._M, self._S, self.dirac, self._covariates, self._Theta_zero
        )

    @property
    def number_of_parameters(self):
        return self._p * (2 * self.d + (self._p + 1) / 2)
