import time
from abc import ABC, abstractmethod
import pickle
import warnings
import os

import pandas as pd
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


from ._closed_forms import (
    closed_formula_coef,
    closed_formula_covariance,
    closed_formula_pi,
)
from .elbos import elbo_plnpca, elbo_zi_pln, profiled_elbo_pln
from ._utils import (
    PLNPlotArgs,
    init_sigma,
    init_components,
    init_coef,
    check_two_dimensions_are_equal,
    init_latent_mean,
    format_data,
    format_model_param,
    check_data_shape,
    nice_string_of_dict,
    plot_ellipse,
    closest,
    prepare_covariates,
    to_tensor,
    check_dimensions_are_equal,
)

if torch.cuda.is_available():
    DEVICE = "cuda"
    print("Using a GPU")
else:
    DEVICE = "cpu"
# shoudl add a good init for M. for pln we should not put
# the maximum of the log posterior, for plnpca it may be ok.

NB_CHARACTERS_FOR_NICE_PLOT = 70


class _PLN(ABC):
    """
    Virtual class for all the PLN models.

    This class must be derivatived. The methods `get_covariance`, `compute_elbo`,
    `random_init_latent_parameters` and `list_of_parameters_needing_gradient` must
    be defined.
    """

    WINDOW = 15
    n_samples: int
    dim: int
    nb_cov: int
    _counts: torch.Tensor
    _covariates: torch.Tensor
    _offsets: torch.Tensor
    _coef: torch.Tensor
    beginnning_time: float
    _latent_var: torch.Tensor
    _latent_mean: torch.Tensor

    def __init__(self):
        """
        Simple initialization method.
        """
        self._fitted = False
        self.plotargs = PLNPlotArgs(self.WINDOW)

    def format_model_param(self, counts, covariates, offsets, offsets_formula):
        self._counts, self._covariates, self._offsets = format_model_param(
            counts, covariates, offsets, offsets_formula
        )

    @property
    def nb_iteration_done(self):
        return len(self.plotargs.elbos_list)

    @property
    def n_samples(self):
        return self._counts.shape[0]

    @property
    def dim(self):
        return self._counts.shape[1]

    @property
    def nb_cov(self):
        return self.covariates.shape[1]

    def smart_init_coef(self):
        self._coef = init_coef(self._counts, self._covariates)

    def random_init_coef(self):
        self._coef = torch.randn((self.nb_cov, self.dim), device=DEVICE)

    @abstractmethod
    def random_init_model_parameters(self):
        pass

    @abstractmethod
    def smart_init_model_parameters(self):
        pass

    @abstractmethod
    def random_init_latent_parameters(self):
        pass

    def smart_init_latent_parameters(self):
        pass

    def init_parameters(self, do_smart_init):
        print("Initialization ...")
        if do_smart_init:
            self.smart_init_model_parameters()
            self.smart_init_latent_parameters()
        else:
            self.random_init_model_parameters()
            self.random_init_latent_parameters()
        print("Initialization finished")
        self.put_parameters_to_device()

    def put_parameters_to_device(self):
        for parameter in self.list_of_parameters_needing_gradient:
            parameter.requires_grad_(True)

    @property
    def list_of_parameters_needing_gradient(self):
        """
        A list containing all the parameters that needs to be upgraded via a gradient step.
        """

    def fit(
        self,
        counts,
        covariates=None,
        offsets=None,
        nb_max_iteration=50000,
        lr=0.01,
        class_optimizer=torch.optim.Rprop,
        tol=1e-6,
        do_smart_init=True,
        verbose=False,
        offsets_formula="logsum",
        keep_going=False,
    ):
        """
        Main function of the class. Fit a PLN to the data.
        Parameters
        ----------
        counts : torch.tensor or ndarray or DataFrame.
            2-d count data.
        covariates : torch.tensor or ndarray or DataFrame or
            None, default = None
            If not `None`, the first dimension should equal the first
            dimension of `counts`.
        offsets : torch.tensor or ndarray or DataFrame or None, default = None
            Model offset. If not `None`, size should be the same as `counts`.
        """
        self.print_beginning_message()
        self.beginnning_time = time.time()

        if keep_going is False:
            self.format_model_param(counts, covariates, offsets, offsets_formula)
            check_data_shape(self._counts, self._covariates, self._offsets)
            self.init_parameters(do_smart_init)
        if self._fitted is True and keep_going is True:
            self.beginnning_time -= self.plotargs.running_times[-1]
        self.optim = class_optimizer(self.list_of_parameters_needing_gradient, lr=lr)
        stop_condition = False
        while self.nb_iteration_done < nb_max_iteration and stop_condition == False:
            loss = self.trainstep()
            criterion = self.compute_criterion_and_update_plotargs(loss, tol)
            if abs(criterion) < tol:
                stop_condition = True
            if verbose and self.nb_iteration_done % 50 == 0:
                self.print_stats()
        self.print_end_of_fitting_message(stop_condition, tol)
        self._fitted = True

    def trainstep(self):
        """
        simple docstrings with black errors
        """
        self.optim.zero_grad()
        loss = -self.compute_elbo()
        loss.backward()
        self.optim.step()
        self.update_closed_forms()
        return loss

    def pca_projected_latent_variables(self, n_components=None):
        if n_components is None:
            n_components = self.get_max_components()
        if n_components > self.dim:
            raise RuntimeError(
                f"You ask more components ({n_components}) than variables ({self.dim})"
            )
        pca = PCA(n_components=n_components)
        return pca.fit_transform(self.latent_variables.detach().cpu())

    @property
    @abstractmethod
    def latent_variables(self):
        pass

    def print_end_of_fitting_message(self, stop_condition, tol):
        if stop_condition is True:
            print(
                f"Tolerance {tol} reached"
                f"n {self.plotargs.iteration_number} iterations"
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
        print("ELBO:", np.round(self.plotargs.elbos_list[-1], 6))

    def compute_criterion_and_update_plotargs(self, loss, tol):
        self.plotargs.elbos_list.append(-loss.item())
        self.plotargs.running_times.append(time.time() - self.beginnning_time)
        if self.plotargs.iteration_number > self.WINDOW:
            criterion = abs(
                self.plotargs.elbos_list[-1]
                - self.plotargs.elbos_list[-1 - self.WINDOW]
            )
            self.plotargs.criterions.append(criterion)
            return criterion
        return tol

    def update_closed_forms(self):
        pass

    @abstractmethod
    def compute_elbo(self):
        """
        Compute the Evidence Lower BOund (ELBO) that will be maximized
        by pytorch.
        """

    def display_covariance(self, ax=None, savefig=False, name_file=""):
        """
        Display a heatmap of covariance to visualize correlations.

        If covariance is too big (size is > 400), will only display the
        first block of size (400,400).

        Parameters
        ----------
        ax : matplotlib Axes, optional
            Axes in which to draw the plot, otherwise use the
            currently-active Axes.
        savefig: bool, optional
            If True the figure will be saved. Default is False.
        name_file : str, optional
            The name of the file the graphic will be saved to if saved.
            Default is an empty string.
        """
        if self.dim > 400:
            warnings.warn("Only displaying the first 400 variables.")
            sigma = sigma[:400, :400]
            sns.heatmap(self.covariance[:400, :400], ax=ax)
        else:
            sns.heatmap(self.covariance, ax=ax)
        if savefig:
            plt.savefig(name_file + self.NAME)
        plt.show()  # to avoid displaying a blanck screen

    def __str__(self):
        delimiter = "=" * NB_CHARACTERS_FOR_NICE_PLOT
        string = f"A multivariate Poisson Lognormal with {self.description} \n"
        string += f"{delimiter}\n"
        string += nice_string_of_dict(self.dict_for_printing)
        string += f"{delimiter}\n"
        string += "* Useful properties\n"
        string += f"    {self.useful_properties_string}\n"
        string += "* Useful methods\n"
        string += f"    {self.useful_methods_string}\n"
        string += f"* Additional properties for {self.NAME}\n"
        string += f"    {self.additional_properties_string}\n"
        string += f"* Additionial methods for {self.NAME}\n"
        string += f"    {self.additional_methods_string}"
        return string

    @property
    def additional_methods_string(self):
        pass

    @property
    def additional_properties_string(self):
        pass

    def show(self, axes=None):
        print("Likelihood:", -self.loglike)
        if self._fitted is False:
            nb_axes = 1
        else:
            nb_axes = 3
        if axes is None:
            _, axes = plt.subplots(1, nb_axes, figsize=(23, 5))
        if self._fitted is True:
            self.plotargs.show_loss(ax=axes[2])
            self.plotargs.show_stopping_criterion(ax=axes[1])
            self.display_covariance(ax=axes[0])
        else:
            self.display_covariance(ax=axes)
        plt.show()

    @property
    def elbos_list(self):
        return self.plotargs.elbos_list

    @property
    def loglike(self):
        if self._fitted is False:
            t0 = time.time()
            self.plotargs.elbos_list.append(self.compute_elbo().item())
            self.plotargs.running_times.append(time.time() - t0)
        return self.n_samples * self.elbos_list[-1]

    @property
    def BIC(self):
        return -self.loglike + self.number_of_parameters / 2 * np.log(self.n_samples)

    @property
    def AIC(self):
        return -self.loglike + self.number_of_parameters

    @property
    def latent_parameters(self):
        return {"latent_var": self.latent_var, "latent_mean": self.latent_mean}

    @property
    def model_parameters(self):
        return {"coef": self.coef, "covariance": self.covariance}

    @property
    def dict_data(self):
        return {
            "counts": self.counts,
            "covariates": self.covariates,
            "offsets": self.offsets,
        }

    @property
    def model_in_a_dict(self):
        return self.dict_data | self.model_parameters | self.latent_parameters

    @property
    def coef(self):
        return self.attribute_or_none("_coef")

    @property
    def latent_mean(self):
        return self.attribute_or_none("_latent_mean")

    @property
    def latent_var(self):
        return self.attribute_or_none("_latent_var")

    @latent_var.setter
    def latent_var(self, latent_var):
        self._latent_var = latent_var

    @latent_mean.setter
    def latent_mean(self, latent_mean):
        self._latent_mean = latent_mean

    def attribute_or_none(self, attribute_name):
        if hasattr(self, attribute_name):
            attr = getattr(self, attribute_name)
            if isinstance(attr, torch.Tensor):
                return attr.detach().cpu()
            return attr
        return None

    def save(self, path_of_directory="./"):
        path = f"{path_of_directory}/{self.model_path}/"
        os.makedirs(path, exist_ok=True)
        for key, value in self.model_in_a_dict.items():
            filename = f"{path}/{key}.csv"
            if isinstance(value, torch.Tensor):
                pd.DataFrame(np.array(value.cpu().detach())).to_csv(
                    filename, header=None, index=None
                )
            else:
                pd.DataFrame(np.array([value])).to_csv(
                    filename, header=None, index=None
                )
        self._fitted = True

    def load(self, path_of_directory="./"):
        path = f"{path_of_directory}/{self.model_path}/"
        for key, value in self.model_in_a_dict.items():
            value = torch.from_numpy(
                pd.read_csv(path + key + ".csv", header=None).values
            )
            setattr(self, key, value)
        self.put_parameters_to_device()

    @property
    def counts(self):
        return self.attribute_or_none("_counts")

    @property
    def offsets(self):
        return self.attribute_or_none("_offsets")

    @property
    def covariates(self):
        return self.attribute_or_none("_covariates")

    @counts.setter
    def counts(self, counts):
        counts = to_tensor(counts)
        if hasattr(self, "_counts"):
            check_dimensions_are_equal(self._counts, counts)
        self._counts = counts

    @offsets.setter
    def offsets(self, offsets):
        self._offsets = offsets

    @covariates.setter
    def covariates(self, covariates):
        self._covariates = covariates

    @coef.setter
    def coef(self, coef):
        self._coef = coef

    @property
    def dict_for_printing(self):
        return {
            "Loglike": np.round(self.loglike, 2),
            "Dimension": self.dim,
            "Nb param": int(self.number_of_parameters),
            "BIC": int(self.BIC),
            "AIC": int(self.AIC),
        }

    @property
    def optim_parameters(self):
        return {"Number of iterations done": self.nb_iteration_done}

    @property
    def useful_properties_string(self):
        return ".latent_variables, .model_parameters, .latent_parameters, \
.optim_parameters"

    @property
    def useful_methods_string(self):
        return ".show(), .coef() .transform(), .sigma(), .predict(), \
.pca_projected_latent_variables()"

    def sigma(self):
        return self.covariance

    def predict(self, covariates=None):
        if isinstance(covariates, torch.Tensor):
            if covariates.shape[-1] != self.nb_cov - 1:
                error_string = f"X has wrong shape ({covariates.shape}).Should"
                error_string += f" be ({self.n_samples, self.nb_cov-1})."
                raise RuntimeError(error_string)
        covariates_with_ones = prepare_covariates(covariates, self.n_samples)
        return covariates_with_ones @ self.coef


# need to do a good init for M and S
class PLN(_PLN):
    NAME = "PLN"
    coef: torch.Tensor

    @property
    def description(self):
        return "full covariance model."

    @property
    def coef(self):
        if hasattr(self, "_latent_mean") and hasattr(self, "_covariates"):
            return self._coef
        return None

    @coef.setter
    def coef(self, coef):
        pass

    def smart_init_latent_parameters(self):
        self.random_init_latent_parameters()

    def random_init_latent_parameters(self):
        self._latent_var = 1 / 2 * torch.ones((self.n_samples, self.dim)).to(DEVICE)
        self._latent_mean = torch.ones((self.n_samples, self.dim)).to(DEVICE)

    @property
    def model_path(self):
        return self.NAME

    @property
    def list_of_parameters_needing_gradient(self):
        return [self._latent_mean, self._latent_var]

    def get_max_components(self):
        return self.dim

    def compute_elbo(self):
        """
        Compute the Evidence Lower BOund (ELBO) that will be
        maximized by pytorch. Here we use the profiled ELBO
        for the full covariance matrix.
        """
        return profiled_elbo_pln(
            self._counts,
            self._covariates,
            self._offsets,
            self._latent_mean,
            self._latent_var,
        )

    def smart_init_model_parameters(self):
        # no model parameters since we are doing a profiled ELBO
        pass

    def random_init_model_parameters(self):
        # no model parameters since we are doing a profiled ELBO
        pass

    @property
    def _coef(self):
        return closed_formula_coef(self._covariates, self._latent_mean)

    @property
    def _covariance(self):
        return closed_formula_covariance(
            self._covariates,
            self._latent_mean,
            self._latent_var,
            self._coef,
            self.n_samples,
        )

    def print_beginning_message(self):
        print(f"Fitting a PLN model with {self.description}")

    @property
    def latent_variables(self):
        return self.latent_mean

    @property
    def number_of_parameters(self):
        return self.dim * (self.dim + self.nb_cov)

    def transform(self):
        return self.latent_variables

    @property
    def covariance(self):
        if all(
            hasattr(self, attr)
            for attr in [
                "_covariates",
                "_latent_mean",
                "_latent_var",
                "_coef",
                "n_samples",
            ]
        ):
            return self._covariance.detach()
        return None

    @covariance.setter
    def covariance(self, covariance):
        pass


class PLNPCA:
    def __init__(self, ranks):
        if isinstance(ranks, (list, np.ndarray)):
            self.ranks = ranks
            self.dict_models = {}
            for rank in ranks:
                if isinstance(rank, (int, np.int64)):
                    self.dict_models[rank] = _PLNPCA(rank)
                else:
                    raise TypeError(
                        "Please instantiate with either a list\
                              of integers or an integer."
                    )
        elif isinstance(ranks, int):
            self.ranks = [ranks]
            self.dict_models = {ranks: _PLNPCA(ranks)}
        else:
            raise TypeError(
                "Please instantiate with either a list of \
                        integers or an integer."
            )

    @property
    def models(self):
        return list(self.dict_models.values())

    def print_beginning_message(self):
        return f"Adjusting {len(self.ranks)} PLN models for PCA analysis \n"

    @property
    def dim(self):
        return self[self.ranks[0]].dim

    ## should do something for this weird init. pb: if doing the init of self._counts etc
    ## only in PLNPCA, then we don't do it for each _PLNPCA but then PLN is not doing it.
    def fit(
        self,
        counts,
        covariates=None,
        offsets=None,
        nb_max_iteration=100000,
        lr=0.01,
        class_optimizer=torch.optim.Rprop,
        tol=1e-6,
        do_smart_init=True,
        verbose=False,
        offsets_formula="logsum",
        keep_going=False,
    ):
        self.print_beginning_message()
        counts, _, offsets = format_model_param(
            counts, covariates, offsets, offsets_formula
        )
        for pca in self.dict_models.values():
            pca.fit(
                counts,
                covariates,
                offsets,
                nb_max_iteration,
                lr,
                class_optimizer,
                tol,
                do_smart_init,
                verbose,
                None,
                keep_going,
            )
        self.print_ending_message()

    def print_ending_message(self):
        delimiter = "=" * NB_CHARACTERS_FOR_NICE_PLOT
        print(f"{delimiter}\n")
        print("DONE!")
        print(f"    Best model(lower BIC): {self.criterion_dict('BIC')}\n ")
        print(f"    Best model(lower AIC): {self.criterion_dict('AIC')}\n ")
        print(f"{delimiter}\n")

    def criterion_dict(self, criterion="AIC"):
        return self.best_model(criterion).rank

    def __getitem__(self, rank):
        if (rank in self.ranks) is False:
            asked_rank = rank
            rank = closest(self.ranks, asked_rank)
            warning_string = " \n No such a model in the collection."
            warning_string += "Returning model with closest value.\n"
            warning_string += f"Requested: {asked_rank}, returned: {rank}"
            warnings.warn(message=warning_string)
        return self.dict_models[rank]

    @property
    def BIC(self):
        return {model.rank: int(model.BIC) for model in self.models}

    @property
    def AIC(self):
        return {model.rank: int(model.AIC) for model in self.models}

    @property
    def loglikes(self):
        return {model.rank: model.loglike for model in self.models}

    def show(self):
        bic = self.BIC
        aic = self.AIC
        loglikes = self.loglikes
        bic_color = "blue"
        aic_color = "red"
        loglikes_color = "orange"
        plt.scatter(bic.keys(), bic.values(), label="BIC criterion", c=bic_color)
        plt.plot(bic.keys(), bic.values(), c=bic_color)
        plt.axvline(self.best_BIC_model_rank, c=bic_color, linestyle="dotted")
        plt.scatter(aic.keys(), aic.values(), label="AIC criterion", c=aic_color)
        plt.axvline(self.best_AIC_model_rank, c=aic_color, linestyle="dotted")
        plt.plot(aic.keys(), aic.values(), c=aic_color)
        plt.xticks(list(aic.keys()))
        plt.scatter(
            loglikes.keys(),
            -np.array(list(loglikes.values())),
            label="Negative log likelihood",
            c=loglikes_color,
        )
        plt.plot(loglikes.keys(), -np.array(list(loglikes.values())), c=loglikes_color)
        plt.legend()
        plt.show()

    @property
    def best_BIC_model_rank(self):
        return self.ranks[np.argmin(list(self.BIC.values()))]

    @property
    def best_AIC_model_rank(self):
        return self.ranks[np.argmin(list(self.AIC.values()))]

    def best_model(self, criterion="AIC"):
        if criterion == "BIC":
            return self[self.best_BIC_model_rank]
        if criterion == "AIC":
            return self[self.best_AIC_model_rank]
        raise ValueError(f"Unknown criterion {criterion}")

    def save(self, path_of_directory="./"):
        for model in self.models:
            model.save(path_of_directory)

    def load(self, path_of_directory="./"):
        for model in self.models:
            model.load(path_of_directory)

    @property
    def n_samples(self):
        return self.models[0].n_samples

    @property
    def _p(self):
        return self[self.ranks[0]].p

    def __str__(self):
        nb_models = len(self.models)
        delimiter = "\n" + "-" * NB_CHARACTERS_FOR_NICE_PLOT + "\n"
        to_print = delimiter
        to_print += f"Collection of {nb_models} PLNPCA models with \
                    {self.dim} variables."
        to_print += delimiter
        to_print += f" - Ranks considered:{self.ranks}\n"
        dict_bic = {"rank": "criterion"} | self.BIC
        to_print += f" - BIC metric:\n{nice_string_of_dict(dict_bic)}\n"

        dict_to_print = self.best_model(criterion="BIC")._rank
        to_print += f"   Best model(lower BIC): {dict_to_print}\n \n"
        dict_aic = {"rank": "criterion"} | self.AIC
        to_print += f" - AIC metric:\n{nice_string_of_dict(dict_aic)}\n"
        to_print += f"   Best model(lower AIC): \
                {self.best_model(criterion = 'AIC')._rank}\n"
        to_print += delimiter
        to_print += f"* Useful properties\n"
        to_print += f"    {self.useful_properties_string}\n"
        to_print += "* Useful methods \n"
        to_print += f"    {self.useful_methods_string}"
        to_print += delimiter
        return to_print

    @property
    def useful_methods_string(self):
        return ".show(), .best_model()"

    @property
    def useful_properties_string(self):
        return ".BIC, .AIC, .loglikes"


class _PLNPCA(_PLN):
    NAME = "PLNPCA"
    _components: torch.Tensor

    def __init__(self, rank):
        super().__init__()
        self._rank = rank

    def init_parameters(self, do_smart_init):
        if self.dim < self._rank:
            warning_string = f"\nThe requested rank of approximation {self._rank} \
                is greater than the number of variables {self.dim}. \
                Setting rank to {self.dim}"
            warnings.warn(warning_string)
            self._rank = self.dim
        super().init_parameters(do_smart_init)

    @property
    def model_path(self):
        return f"{self.NAME}_{self._rank}_rank"

    @property
    def rank(self):
        return self._rank

    def get_max_components(self):
        return self._rank

    def print_beginning_message(self):
        print("-" * NB_CHARACTERS_FOR_NICE_PLOT)
        print(f"Fitting a PLNPCA model with {self._rank} components")

    @property
    def model_parameters(self):
        return {"coef": self.coef, "components": self.components}

    def smart_init_model_parameters(self):
        super().smart_init_coef()
        self._components = init_components(
            self._counts, self._covariates, self._coef, self._rank
        )

    def random_init_model_parameters(self):
        super().random_init_coef()
        self._components = torch.randn((self.dim, self._rank)).to(DEVICE)

    def random_init_latent_parameters(self):
        self._latent_var = 1 / 2 * torch.ones((self.n_samples, self._rank)).to(DEVICE)
        self._latent_mean = torch.ones((self.n_samples, self._rank)).to(DEVICE)

    def smart_init_latent_parameters(self):
        self._latent_mean = (
            init_latent_mean(
                self._counts,
                self._covariates,
                self._offsets,
                self._coef,
                self._components,
            )
            .to(DEVICE)
            .detach()
        )
        self._latent_var = 1 / 2 * torch.ones((self.n_samples, self._rank)).to(DEVICE)
        self._latent_mean.requires_grad_(True)
        self._latent_var.requires_grad_(True)

    @property
    def list_of_parameters_needing_gradient(self):
        return [self._components, self._coef, self._latent_mean, self._latent_var]

    def compute_elbo(self):
        return elbo_plnpca(
            self._counts,
            self._covariates,
            self._offsets,
            self._latent_mean,
            self._latent_var,
            self._components,
            self._coef,
        )

    @property
    def number_of_parameters(self):
        return self.dim * (self.nb_cov + self._rank) - self._rank * (self._rank - 1) / 2

    @property
    def additional_properties_string(self):
        return ".projected_latent_variables"

    @property
    def additional_methods_string(self):
        string = "    only for rank=2: .viz()"
        return string

    @property
    def covariance(self):
        if hasattr(self, "_components"):
            cov_latent = self._latent_mean.T @ self._latent_mean
            cov_latent += torch.diag(
                torch.sum(self._latent_var * self._latent_var), dim=0)
            )
            cov_latent /= self.n_samples
            return (self._components @ cov_latent @ self._components.T).detach()
        return None

    @property
    def description(self):
        return f" {self.rank} principal component."

    @property
    def latent_variables(self):
        return torch.matmul(self._latent_mean, self._components.T)

    @property
    def projected_latent_variables(self):
        ortho_components = torch.linalg.qr(self._components, "reduced")[0]
        return torch.mm(self.latent_variables, ortho_components).detach().cpu()

    @property
    def components(self):
        return self.attribute_or_none("_components")

    @components.setter
    def components(self, components):
        self._components = components

    def viz(self, ax=None, colors=None):
        if self._rank != 2:
            raise RuntimeError("Can't perform visualization for rank != 2.")
        if ax is None:
            ax = plt.gca()
        proj_variables = self.projected_latent_variables
        x = proj_variables[:, 0].cpu().numpy()
        y = proj_variables[:, 1].cpu().numpy()
        sns.scatterplot(x=x, y=y, hue=colors, ax=ax)
        covariances = torch.diag_embed(self._latent_var**2).detach().cpu()
        for i in range(covariances.shape[0]):
            plot_ellipse(x[i], y[i], cov=covariances[i], ax=ax)
        return ax

    def transform(self, project=True):
        if project is True:
            return self.projected_latent_variables
        return self.latent_variables


class ZIPLN(PLN):
    NAME = "ZIPLN"

    _pi: torch.Tensor
    _coef_inflation: torch.Tensor
    _dirac: torch.Tensor

    @property
    def description(self):
        return "with full covariance model and zero-inflation."

    def random_init_model_parameters(self):
        super().random_init_model_parameters()
        self._coef_inflation = torch.randn(self.nb_cov, self.dim)
        self._covariance = torch.diag(torch.ones(self.dim)).to(DEVICE)

    # should change the good initialization, especially for _coef_inflation
    def smart_init_model_parameters(self):
        super().smart_init_model_parameters()
        self._covariance = init_sigma(
            self._counts, self._covariates, self._offsets, self._coef
        )
        self._coef_inflation = torch.randn(self.nb_cov, self.dim)

    def random_init_latent_parameters(self):
        self._dirac = self._counts == 0
        self._latent_mean = torch.randn(self.n_samples, self.dim)
        self._latent_var = torch.randn(self.n_samples, self.dim)
        self._pi = (
            torch.empty(self.n_samples, self.dim).uniform_(0, 1).to(DEVICE)
            * self._dirac
        )

    def compute_elbo(self):
        return elbo_zi_pln(
            self._counts,
            self._covariates,
            self._offsets,
            self._latent_mean,
            self._latent_var,
            self._pi,
            self._covariance,
            self._coef,
            self._coef_inflation,
            self._dirac,
        )

    @property
    def list_of_parameters_needing_gradient(self):
        return [self._latent_mean, self._latent_var, self._coef_inflation]

    def update_closed_forms(self):
        self._coef = closed_formula_coef(self._covariates, self._latent_mean)
        self._covariance = closed_formula_covariance(
            self._covariates,
            self._latent_mean,
            self._latent_var,
            self._coef,
            self.n_samples,
        )
        self._pi = closed_formula_pi(
            self._offsets,
            self._latent_mean,
            self._latent_var,
            self._dirac,
            self._covariates,
            self._coef_inflation,
        )

    @property
    def number_of_parameters(self):
        return self.dim * (2 * self.nb_cov + (self.dim + 1) / 2)
