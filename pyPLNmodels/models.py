import time
from abc import ABC, abstractmethod
import warnings
import os
from typing import Optional, Dict, List, Type, Any, Iterable

import pandas as pd
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import plotly.express as px
from mlxtend.plotting import plot_pca_correlation_graph

from ._closed_forms import (
    _closed_formula_coef,
    _closed_formula_covariance,
    _closed_formula_pi,
)
from .elbos import elbo_plnpca, elbo_zi_pln, profiled_elbo_pln
from ._utils import (
    _PlotArgs,
    _format_data,
    _nice_string_of_dict,
    _plot_ellipse,
    _check_data_shape,
    _extract_data_from_formula,
    _get_dict_initialization,
    _array2tensor,
    _handle_data,
)

from ._initialization import (
    _init_covariance,
    _init_components,
    _init_coef,
    _init_latent_mean,
)

if torch.cuda.is_available():
    DEVICE = "cuda"
    print("Using a GPU.")
else:
    DEVICE = "cpu"
# shoudl add a good init for M. for pln we should not put
# the maximum of the log posterior, for plnpca it may be ok.

NB_CHARACTERS_FOR_NICE_PLOT = 70


class model(ABC):
    _WINDOW = 15
    n_samples: int
    dim: int
    nb_cov: int
    _counts: torch.Tensor
    _covariates: torch.Tensor
    _offsets: torch.Tensor
    _coef: torch.Tensor
    _beginning_time: float
    _latent_sqrt_var: torch.Tensor
    _latent_mean: torch.Tensor

    def __init__(
        self,
        counts: torch.Tensor,
        covariates: Optional[torch.Tensor] = None,
        offsets: Optional[torch.Tensor] = None,
        offsets_formula: str = "logsum",
        dict_initialization: Optional[dict] = None,
        take_log_offsets: bool = False,
        add_const: bool = True,
    ):
        """
        Initializes the model class.

        Parameters
        ----------
        counts : torch.Tensor
            The count data.
        covariates : torch.Tensor, optional
            The covariate data. Defaults to None.
        offsets : torch.Tensor, optional
            The offsets data. Defaults to None.
        offsets_formula : str, optional
            The formula for offsets. Defaults to "logsum".
        dict_initialization : dict, optional
            The initialization dictionary. Defaults to None.
        take_log_offsets : bool, optional
            Whether to take the log of offsets. Defaults to False.
        add_const: bool, optional
            Whether to add a column of one in the covariates. Defaults to True.
        """
        (
            self._counts,
            self._covariates,
            self._offsets,
            self.column_counts,
        ) = _handle_data(
            counts, covariates, offsets, offsets_formula, take_log_offsets, add_const
        )
        self._fitted = False
        self._plotargs = _PlotArgs(self._WINDOW)
        if dict_initialization is not None:
            self._set_init_parameters(dict_initialization)

    @classmethod
    def from_formula(
        cls,
        formula: str,
        data: dict,
        offsets_formula: str = "logsum",
        dict_initialization: Optional[dict] = None,
        take_log_offsets: bool = False,
    ):
        """
        Create a model instance from a formula and data.
        See also :func:`~pyPLNmodels.PlnPCAcollection.__init__`

        Parameters
        ----------
        formula : str
            The formula.
        data : dict
            The data dictionary.
        offsets_formula : str, optional
            The formula for offsets. Defaults to "logsum".
        dict_initialization : dict, optional
            The initialization dictionary. Defaults to None.
        take_log_offsets : bool, optional
            Whether to take the log of offsets. Defaults to False.

        Returns
        -------
        model
            The initialized model instance.
        """
        counts, covariates, offsets = _extract_data_from_formula(formula, data)
        return cls(
            counts,
            covariates,
            offsets,
            offsets_formula,
            dict_initialization,
            take_log_offsets,
            add_const=False,
        )

    def _set_init_parameters(self, dict_initialization: dict):
        """
        Set initial parameters based on a dictionary.

        Parameters
        ----------
        dict_initialization : dict
            The initialization dictionary.
        """
        if "coef" not in dict_initialization.keys():
            print("No coef is initialized.")
            self.coef = None
        for key, array in dict_initialization.items():
            array = _format_data(array)
            setattr(self, key, array)
        self._fitted = True

    @property
    def fitted(self) -> bool:
        """
        Whether the model is fitted.

        Returns
        -------
        bool
            True if the model is fitted, False otherwise.
        """
        return self._fitted

    def viz(self, ax=None, colors=None, show_cov: bool = False):
        """
        Visualize the latent variables with a classic PCA.

        Parameters
        ----------
        ax : Optional[Any], optional
            The matplotlib axis to use. If None, the current axis is used, by default None.
        colors : Optional[Any], optional
            The colors to use for plotting, by default None.
        show_cov: bool, optional
            If True, will display ellipses with right covariances. Default is False.
        Raises
        ------
        RuntimeError
            If the rank is less than 2.

        Returns
        -------
        Any
            The matplotlib axis.
        """
        if ax is None:
            ax = plt.gca()
        if self._get_max_components() < 2:
            raise RuntimeError("Can't perform visualization for dim < 2.")
        pca = self.sk_PCA(n_components=2)
        proj_variables = pca.transform(self.latent_variables)
        x = proj_variables[:, 0]
        y = proj_variables[:, 1]
        sns.scatterplot(x=x, y=y, hue=colors, ax=ax)
        if show_cov is True:
            sk_components = torch.from_numpy(pca.components_)
            covariances = self._get_pca_low_dim_covariances(sk_components).detach()
            for i in range(covariances.shape[0]):
                _plot_ellipse(x[i], y[i], cov=covariances[i], ax=ax)
        return ax

    @property
    def nb_iteration_done(self) -> int:
        """
        The number of iterations done.

        Returns
        -------
        int
            The number of iterations done.
        """
        return len(self._plotargs._elbos_list)

    @property
    def n_samples(self) -> int:
        """
        The number of samples.

        Returns
        -------
        int
            The number of samples.
        """
        return self._counts.shape[0]

    @property
    def dim(self) -> int:
        """
        The dimension.

        Returns
        -------
        int
            The dimension.
        """
        return self._counts.shape[1]

    @property
    def nb_cov(self) -> int:
        """
        The number of covariates.

        Returns
        -------
        int
            The number of covariates.
        """
        if self.covariates is None:
            return 0
        return self.covariates.shape[1]

    def _smart_init_coef(self):
        """
        Initialize coefficients smartly.
        """
        self._coef = _init_coef(self._counts, self._covariates, self._offsets)

    def _random_init_coef(self):
        """
        Randomly initialize coefficients.
        """
        if self.nb_cov == 0:
            self._coef = None
        self._coef = torch.randn((self.nb_cov, self.dim), device=DEVICE)

    @abstractmethod
    def _random_init_model_parameters(self):
        """
        Abstract method to randomly initialize model parameters.
        """
        pass

    @abstractmethod
    def _random_init_latent_parameters(self):
        """
        Abstract method to randomly initialize latent parameters.
        """
        pass

    def _smart_init_latent_parameters(self):
        """
        Initialize latent parameters smartly.
        """
        pass

    def _init_parameters(self, do_smart_init: bool):
        """
        Initialize model parameters.

        Parameters
        ----------
        do_smart_init : bool
            Whether to perform smart initialization.
        """
        print("Initialization ...")
        if do_smart_init:
            self._smart_init_model_parameters()
            self._smart_init_latent_parameters()
        else:
            self._random_init_model_parameters()
            self._random_init_latent_parameters()
        print("Initialization finished")

    def _put_parameters_to_device(self):
        """
        Move parameters to the device.
        """
        for parameter in self._list_of_parameters_needing_gradient:
            parameter.requires_grad_(True)

    @property
    def _list_of_parameters_needing_gradient(self):
        """
        A list containing all the parameters that need to be upgraded via a gradient step.

        Returns
        -------
        List[torch.Tensor]
            List of parameters needing gradient.
        """
        ...

    def fit(
        self,
        nb_max_iteration: int = 50000,
        lr: float = 0.01,
        class_optimizer: torch.optim.Optimizer = torch.optim.Rprop,
        tol: float = 1e-3,
        do_smart_init: bool = True,
        verbose: bool = False,
    ):
        """
        Fit the model.

        Parameters
        ----------
        nb_max_iteration : int, optional
            The maximum number of iterations. Defaults to 50000.
        lr : float, optional
            The learning rate. Defaults to 0.01.
        class_optimizer : torch.optim.Optimizer, optional
            The optimizer class. Defaults to torch.optim.Rprop.
        tol : float, optional
            The tolerance for convergence. Defaults to 1e-3.
        do_smart_init : bool, optional
            Whether to perform smart initialization. Defaults to True.
        verbose : bool, optional
            Whether to print training progress. Defaults to False.
        .. code-block:: python
        Examples
        --------
            >>> from pyPLNmodels import Pln, get_real_count_data
            >>> counts = get_real_count_data()
            >>> pln = Pln(counts,add_const = True)
            >>> pln.fit()
            >>> print(pln)
        """
        self._pring_beginning_message()
        self._beginning_time = time.time()

        if self._fitted is False:
            self._init_parameters(do_smart_init)
        elif len(self._plotargs.running_times) > 0:
            self._beginning_time -= self._plotargs.running_times[-1]
        self._put_parameters_to_device()
        self.optim = class_optimizer(self._list_of_parameters_needing_gradient, lr=lr)
        stop_condition = False
        while self.nb_iteration_done < nb_max_iteration and not stop_condition:
            loss = self._trainstep()
            criterion = self._compute_criterion_and_update_plotargs(loss, tol)
            if abs(criterion) < tol:
                stop_condition = True
            if verbose and self.nb_iteration_done % 50 == 0:
                self._print_stats()
        self._print_end_of_fitting_message(stop_condition, tol)
        self._fitted = True

    def _trainstep(self):
        """
        Perform a single training step.

        Returns
        -------
        torch.Tensor
            The loss value.
        """
        self.optim.zero_grad()
        loss = -self.compute_elbo()
        loss.backward()
        self.optim.step()
        self._update_closed_forms()
        return loss

    def pca_projected_latent_variables(self, n_components: Optional[int] = None):
        """
        Perform PCA on the latent variables and project them onto a lower-dimensional space.

        Parameters
        ----------
        n_components : int, optional
            The number of components to keep. If None, all components are kept. Defaults to None.

        Returns
        -------
        numpy.ndarray
            The projected latent variables.
        Raises
        ------
        ValueError
           If the number of components asked is greater than the number of dimensions.
        Examples
        --------
            >>> from pyPLNmodels import Pln, get_real_count_data
            >>> counts = get_real_count_data()
            >>> data = {"counts": counts}
            >>> pln = Pln.from_formula("counts ~ 1", data = data)
            >>> pln.fit()
            >>> pca_proj = pln.pca_projected_latent_variables()
            >>> print(pca_proj.shape)
        """
        pca = self.sk_PCA(n_components=n_components)
        return pca.transform(self.latent_variables.cpu())

    def sk_PCA(self, n_components=None):
        """
        Perform PCA on the latent variables.

        Parameters
        ----------
        n_components : int, optional
            The number of components to keep. If None, all components are kept. Defaults to None.

        Returns
        -------
        sklearn.decomposition.PCA
            sklearn.decomposition.PCA object with all the features from sklearn.
        Raises
        ------
        ValueError
           If the number of components asked is greater than the number of dimensions.
        """
        if n_components is None:
            n_components = self._get_max_components()
        if n_components > self.dim:
            raise ValueError(
                f"You ask more components ({n_components}) than variables ({self.dim})"
            )
        pca = PCA(n_components=n_components)
        pca.fit(self.latent_variables.cpu())
        return pca

    @property
    def latent_var(self) -> torch.Tensor:
        """
        Property representing the latent variance.

        Returns
        -------
        torch.Tensor
            The latent variance tensor.
        """
        return (self.latent_sqrt_var**2).detach()

    def scatter_pca_matrix(self, n_components=None, color=None):
        """
        Generates a scatter matrix plot based on Principal Component Analysis (PCA).

        Parameters
        ----------
            n_components (int, optional): The number of components to consider for plotting.
                If not specified, the maximum number of components will be used.
                Defaults to None.

            color (str, optional): The name of the variable used for color coding the scatter plot.
                If not specified, the scatter plot will not be color-coded.
                Defaults to None.
        Raises
        ------
            ValueError: If the number of components requested is greater than the number of variables in the dataset.
        """

        if n_components is None:
            n_components = self._get_max_components()

        if n_components > self.dim:
            raise ValueError(
                f"You ask more components ({n_components}) than variables ({self.dim})"
            )
        pca = self.sk_PCA(n_components=n_components)
        proj_variables = pca.transform(self.latent_variables)
        components = torch.from_numpy(pca.components_)

        labels = {
            str(i): f"PC{i+1}: {np.round(pca.explained_variance_ratio_*100, 1)[i]}%"
            for i in range(n_components)
        }
        proj_variables
        fig = px.scatter_matrix(
            proj_variables,
            dimensions=range(n_components),
            color=color,
            labels=labels,
        )
        fig.update_traces(diagonal_visible=False)
        fig.show()

    def plot_pca_correlation_graph(self, variables_names, indices_of_variables=None):
        """
        Visualizes variables using PCA and plots a correlation graph.

        Parameters
        ----------
            variables_names : List[str]
                A list of variable names to visualize.
            indices_of_variables : Optional[List[int]], optional
                A list of indices corresponding to the variables.
                If None, indices are determined based on `column_counts`, by default None

        Raises
        ------
            ValueError
                If `indices_of_variables` is None and `column_counts` is not set.
            ValueError
                If the length of `indices_of_variables` is different from the length of `variables_names`.

        Returns
        -------
            None
        """
        if indices_of_variables is None:
            if self.column_counts is None:
                raise ValueError(
                    "No names have been given to the column of "
                    "counts. Please set the column_counts to the"
                    "needed names or instantiate a new model with"
                    "a pd.DataFrame with appropriate column names"
                )
            indices_of_variables = []
            for variables_name in variables_names:
                index = self.column_counts.get_loc(variables_name)
                indices_of_variables.append(index)
        else:
            if len(indices_of_variables) != len(variables_names):
                raise ValueError(
                    f"Number of variables {len(indices_of_variables)} should be the same as the number of variables_names {len(variables_names)}"
                )

        n_components = 2
        pca = self.sk_PCA(n_components=n_components)
        variables = self.latent_variables
        proj_variables = pca.transform(variables)
        ## the package is not correctly printing the variance ratio
        figure, correlation_matrix = plot_pca_correlation_graph(
            variables[:, indices_of_variables],
            variables_names=variables_names,
            X_pca=proj_variables,
            explained_variance=pca.explained_variance_ratio_,
            dimensions=(1, 2),
            figure_axis_size=10,
        )
        plt.show()

    @property
    @abstractmethod
    def latent_variables(self):
        """
        Abstract property representing the latent variables.
        """
        pass

    def _print_end_of_fitting_message(self, stop_condition: bool, tol: float):
        """
        Print the end-of-fitting message.

        Parameters
        ----------
        stop_condition : bool
            Whether the stop condition was met.
        tol : float
            The tolerance for convergence.
        """
        if stop_condition is True:
            print(
                f"Tolerance {tol} reached "
                f"in {self._plotargs.iteration_number} iterations"
            )
        else:
            print(
                "Maximum number of iterations reached : ",
                self._plotargs.iteration_number,
                "last criterion = ",
                np.round(self._plotargs.criterions[-1], 8),
            )

    def _print_stats(self):
        """
        Print the training statistics.
        """
        print("-------UPDATE-------")
        print("Iteration number: ", self._plotargs.iteration_number)
        print("Criterion: ", np.round(self._plotargs.criterions[-1], 8))
        print("ELBO:", np.round(self._plotargs._elbos_list[-1], 6))

    def _compute_criterion_and_update_plotargs(self, loss, tol):
        """
        Compute the convergence criterion and update the plot arguments.

        Parameters
        ----------
        loss : torch.Tensor
            The loss value.
        tol : float
            The tolerance for convergence.

        Returns
        -------
        float
            The computed criterion.
        """
        self._plotargs._elbos_list.append(-loss.item())
        self._plotargs.running_times.append(time.time() - self._beginning_time)
        if self._plotargs.iteration_number > self._WINDOW:
            criterion = abs(
                self._plotargs._elbos_list[-1]
                - self._plotargs._elbos_list[-1 - self._WINDOW]
            )
            self._plotargs.criterions.append(criterion)
            return criterion
        return tol

    def _update_closed_forms(self):
        """
        Update closed-form expressions.
        """
        pass

    @abstractmethod
    def compute_elbo(self):
        """
        Compute the Evidence Lower BOund (ELBO) that will be maximized
        by pytorch.
        """
        pass

    def display_covariance(self, ax=None, savefig=False, name_file=""):
        """
        Display the covariance matrix.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            The axes to plot on. If None, a new figure will be created. Defaults to None.
        savefig : bool, optional
            Whether to save the figure. Defaults to False.
        name_file : str, optional
            The name of the file to save. Defaults to "".
        """
        if self.dim > 400:
            warnings.warn("Only displaying the first 400 variables.")
            sigma = sigma[:400, :400]
            sns.heatmap(self.covariance[:400, :400].cpu(), ax=ax)
        else:
            sns.heatmap(self.covariance.cpu(), ax=ax)
        ax.set_title("Covariance Matrix")
        plt.legend()
        if savefig:
            plt.savefig(name_file + self._NAME)
        plt.show()  # to avoid displaying a blank screen

    def __repr__(self):
        """
        Generate the string representation of the object.

        Returns
        -------
        str
            The string representation of the object.
        """
        if self._fitted is False:
            raise RuntimeError("Please fit the model before printing it.")
        delimiter = "=" * NB_CHARACTERS_FOR_NICE_PLOT
        string = f"A multivariate Poisson Lognormal with {self._description} \n"
        string += f"{delimiter}\n"
        string += _nice_string_of_dict(self._dict_for_printing)
        string += f"{delimiter}\n"
        string += "* Useful properties\n"
        string += f"    {self._useful_properties_string}\n"
        string += "* Useful methods\n"
        string += f"    {self._useful_methods_strings}\n"
        string += f"* Additional properties for {self._NAME}\n"
        string += f"    {self._additional_properties_string}\n"
        string += f"* Additional methods for {self._NAME}\n"
        string += f"    {self._additional_methods_string}"
        return string

    @property
    def _additional_methods_string(self):
        """
        Abstract property representing the additional methods string.
        """

        pass

    @property
    def _additional_properties_string(self):
        """
        Abstract property representing the additional properties string.
        """
        pass

    def show(self, axes=None):
        """
        Show plots.

        Parameters
        ----------
        axes : numpy.ndarray, optional
            The axes to plot on. If None, a new figure will be created. Defaults to None.
        """
        print("Likelihood:", -self.loglike)
        if self._fitted is False:
            nb_axes = 1
        else:
            nb_axes = 3
        if axes is None:
            _, axes = plt.subplots(1, nb_axes, figsize=(23, 5))
        if self._fitted is True:
            self._plotargs._show_loss(ax=axes[2])
            self._plotargs._show_stopping_criterion(ax=axes[1])
            self.display_covariance(ax=axes[0])
        else:
            self.display_covariance(ax=axes)
        plt.show()

    @property
    def _elbos_list(self):
        """
        Property representing the list of ELBO values.
        """
        return self._plotargs._elbos_list

    @property
    def loglike(self):
        """
        Property representing the log-likelihood.

        Returns
        -------
        float
            The log-likelihood.
        """
        if len(self._elbos_list) == 0:
            t0 = time.time()
            self._plotargs._elbos_list.append(self.compute_elbo().item())
            self._plotargs.running_times.append(time.time() - t0)
        return self.n_samples * self._elbos_list[-1]

    @property
    def BIC(self):
        """
        Property representing the Bayesian Information Criterion (BIC).

        Returns
        -------
        float
            The BIC value.
        """
        return -self.loglike + self.number_of_parameters / 2 * np.log(self.n_samples)

    @property
    def AIC(self):
        """
        Property representing the Akaike Information Criterion (AIC).

        Returns
        -------
        float
            The AIC value.
        """
        return -self.loglike + self.number_of_parameters

    @property
    def latent_parameters(self):
        """
        Property representing the latent parameters.

        Returns
        -------
        dict
            The dictionary of latent parameters.
        """
        return {
            "latent_sqrt_var": self.latent_sqrt_var,
            "latent_mean": self.latent_mean,
        }

    @property
    def model_parameters(self):
        """
        Property representing the model parameters.

        Returns
        -------
        dict
            The dictionary of model parameters.
        """
        return {"coef": self.coef, "covariance": self.covariance}

    @property
    def dict_data(self):
        """
        Property representing the data dictionary.

        Returns
        -------
        dict
            The dictionary of data.
        """
        return {
            "counts": self.counts,
            "covariates": self.covariates,
            "offsets": self.offsets,
        }

    @property
    def _model_in_a_dict(self):
        """
        Property representing the model in a dictionary.

        Returns
        -------
        dict
            The dictionary representing the model.
        """
        return {**self.dict_data, **self._dict_parameters}

    @property
    def _dict_parameters(self):
        """
        Property representing the dictionary of parameters.

        Returns
        -------
        dict
            The dictionary of parameters.
        """
        return {**self.model_parameters, **self.latent_parameters}

    @property
    def coef(self):
        """
        Property representing the coefficients.

        Returns
        -------
        torch.Tensor or None
            The coefficients or None.
        """
        return self._cpu_attribute_or_none("_coef")

    @property
    def latent_mean(self):
        """
        Property representing the latent mean.

        Returns
        -------
        torch.Tensor or None
            The latent mean or None.
        """
        return self._cpu_attribute_or_none("_latent_mean")

    @property
    def latent_sqrt_var(self):
        """
        Property representing the latent variance.

        Returns
        -------
        torch.Tensor or None
            The latent variance or None.
        """
        return self._cpu_attribute_or_none("_latent_sqrt_var")

    @latent_mean.setter
    @_array2tensor
    def latent_mean(self, latent_mean):
        """
        Setter for the latent mean property.

        Parameters
        ----------
        latent_mean : torch.Tensor
            The latent mean.

        Raises
        ------
        ValueError
            If the shape of the latent mean is incorrect.
        """
        if latent_mean.shape != (self.n_samples, self.dim):
            raise ValueError(
                f"Wrong shape. Expected {self.n_samples, self.dim}, got {latent_mean.shape}"
            )
        self._latent_mean = latent_mean

    @latent_sqrt_var.setter
    @_array2tensor
    def latent_sqrt_var(self, latent_sqrt_var):
        """
        Setter for the latent variance property.

        Parameters
        ----------
        latent_sqrt_var : torch.Tensor
            The latent variance.

        Raises
        ------
        ValueError
            If the shape of the latent variance is incorrect.
        """
        if latent_sqrt_var.shape != (self.n_samples, self.dim):
            raise ValueError(
                f"Wrong shape. Expected {self.n_samples, self.dim}, got {latent_sqrt_var.shape}"
            )
        self._latent_sqrt_var = latent_sqrt_var

    def _cpu_attribute_or_none(self, attribute_name):
        """
        Get the CPU attribute or return None.

        Parameters
        ----------
        attribute_name : str
            The attribute name.

        Returns
        -------
        torch.Tensor or None
            The attribute value or None.
        """
        if hasattr(self, attribute_name):
            attr = getattr(self, attribute_name)
            if isinstance(attr, torch.Tensor):
                return attr.detach().cpu()
            return attr
        return None

    def save(self, path: str = None):
        """
        Save the model parameters to disk.

        Parameters
        ----------
        path : str, optional
            The path of the directory to save the parameters, by default "./".
        """
        if path is None:
            path = f"./{self._directory_name}"
        os.makedirs(path, exist_ok=True)
        for key, value in self._dict_parameters.items():
            filename = f"{path}/{key}.csv"
            if isinstance(value, torch.Tensor):
                pd.DataFrame(np.array(value.cpu().detach())).to_csv(
                    filename, header=None, index=None
                )
            elif value is not None:
                pd.DataFrame(np.array([value])).to_csv(
                    filename, header=None, index=None
                )

    @property
    def counts(self):
        """
        Property representing the counts.

        Returns
        -------
        torch.Tensor or None
            The counts or None.
        """
        return self._cpu_attribute_or_none("_counts")

    @property
    def offsets(self):
        """
        Property representing the offsets.

        Returns
        -------
        torch.Tensor or None
            The offsets or None.
        """
        return self._cpu_attribute_or_none("_offsets")

    @property
    def covariates(self):
        """
        Property representing the covariates.

        Returns
        -------
        torch.Tensor or None
            The covariates or None.
        """
        return self._cpu_attribute_or_none("_covariates")

    @counts.setter
    @_array2tensor
    def counts(self, counts):
        """
        Setter for the counts property.

        Parameters
        ----------
        counts : torch.Tensor
            The counts.

        Raises
        ------
        ValueError
            If the shape of the counts is incorrect or if the input is negative.
        """
        if self.counts.shape != counts.shape:
            raise ValueError(
                f"Wrong shape for the counts. Expected {self.counts.shape}, got {counts.shape}"
            )
        if torch.min(counts) < 0:
            raise ValueError("Input should be non-negative only.")
        self._counts = counts

    @offsets.setter
    @_array2tensor
    def offsets(self, offsets):
        """
        Setter for the offsets property.

        Parameters
        ----------
        offsets : torch.Tensor
            The offsets.

        Raises
        ------
        ValueError
            If the shape of the offsets is incorrect.
        """
        if self.offsets.shape != offsets.shape:
            raise ValueError(
                f"Wrong shape for the offsets. Expected {self.offsets.shape}, got {offsets.shape}"
            )
        self._offsets = offsets

    @covariates.setter
    @_array2tensor
    def covariates(self, covariates):
        """
        Setter for the covariates property.

        Parameters
        ----------
        covariates : torch.Tensor
            The covariates.

        Raises
        ------
        ValueError
            If the shape of the covariates or counts is incorrect.
        """
        _check_data_shape(self.counts, covariates, self.offsets)
        self._covariates = covariates

    @coef.setter
    @_array2tensor
    def coef(self, coef):
        """
        Setter for the coef property.

        Parameters
        ----------
        coef : torch.Tensor or None
            The coefficients.

        Raises
        ------
        ValueError
            If the shape of the coef is incorrect.
        """
        if coef is None:
            pass
        elif coef.shape != (self.nb_cov, self.dim):
            raise ValueError(
                f"Wrong shape for the coef. Expected {(self.nb_cov, self.dim)}, got {coef.shape}"
            )
        self._coef = coef

    @property
    def _dict_for_printing(self):
        """
        Property representing the dictionary for printing.

        Returns
        -------
        dict
            The dictionary for printing.
        """
        return {
            "Loglike": np.round(self.loglike, 2),
            "Dimension": self.dim,
            "Nb param": int(self.number_of_parameters),
            "BIC": int(self.BIC),
            "AIC": int(self.AIC),
        }

    @property
    def optim_parameters(self):
        """
        Property representing the optimization parameters.

        Returns
        -------
        dict
            The dictionary of optimization parameters.
        """
        return {"Number of iterations done": self.nb_iteration_done}

    @property
    def _useful_properties_string(self):
        """
        Property representing the useful properties as a string.

        Returns
        -------
        str
            The string representation of the useful properties.
        """
        return ".latent_variables, .model_parameters, .latent_parameters, .optim_parameters"

    @property
    def _useful_methods_strings(self):
        """
        Property representing the useful methods as a string.

        Returns
        -------
        str
            The string representation of the useful methods.
        """
        return ".show(), .coef() .transform(), .sigma(), .predict(), .pca_projected_latent_variables(), .plot_pca_correlation_graph(), .viz(), .scatter_pca_matrix()"

    def sigma(self):
        """
        Method returning the covariance matrix.

        Returns
        -------
        torch.Tensor or None
            The covariance matrix or None.
        """
        return self.covariance

    def predict(self, covariates=None):
        """
        Method for making predictions.

        Parameters
        ----------
        covariates : torch.Tensor, optional
            The covariates, by default None.

        Returns
        -------
        torch.Tensor or None
            The predicted values or None.

        Raises
        ------
        AttributeError
            If there are no covariates in the model.
        RuntimeError
            If the shape of the covariates is incorrect.

        Notes
        -----
        - If `covariates` is not provided and there are no covariates in the model, None is returned.
        - If `covariates` is provided, it should have the shape `(_, nb_cov)`, where `nb_cov` is the number of covariates.
        - The predicted values are obtained by multiplying the covariates by the coefficients.

        """
        if covariates is not None and self.nb_cov == 0:
            raise AttributeError("No covariates in the model, can't predict")
        if covariates is None:
            if self.covariates is None:
                print("No covariates in the model.")
                return None
            return self.covariates @ self.coef
        if covariates.shape[-1] != self.nb_cov:
            error_string = f"X has wrong shape ({covariates.shape}). Should"
            error_string += f" be ({self.n_samples, self.nb_cov})."
            raise RuntimeError(error_string)
        return covariates @ self.coef

    @property
    def _directory_name(self):
        """
        Property representing the directory name.

        Returns
        -------
        str
            The directory name.
        """
        return f"{self._NAME}_nbcov_{self.nb_cov}_dim_{self.dim}"

    @property
    def _path_to_directory(self):
        """
        Property representing the path to the directory.

        Returns
        -------
        str
            The path to the directory.
        """
        return ""


# need to do a good init for M and S
class Pln(model):
    _NAME = "Pln"
    coef: torch.Tensor

    @property
    def _description(self):
        """
        Property representing the description of the model.

        Returns
        -------
        str
            The description of the model.
        """
        return "full covariance model."

    @property
    def coef(self):
        """
        Property representing the coefficients.

        Returns
        -------
        torch.Tensor or None
            The coefficients or None.
        """
        if (
            hasattr(self, "_latent_mean")
            and hasattr(self, "_covariates")
            and self.nb_cov > 0
        ):
            return self._coef.detach().cpu()
        return None

    @coef.setter
    def coef(self, coef):
        """
        Setter for the coef property.

        Parameters
        ----------
        coef : torch.Tensor
            The coefficients.
        """
        pass

    def _smart_init_latent_parameters(self):
        """
        Method for smartly initializing the latent parameters.
        """
        self._random_init_latent_parameters()

    def _random_init_latent_parameters(self):
        """
        Method for randomly initializing the latent parameters.
        """
        if not hasattr(self, "_latent_sqrt_var"):
            self._latent_sqrt_var = (
                1 / 2 * torch.ones((self.n_samples, self.dim)).to(DEVICE)
            )
        if not hasattr(self, "_latent_mean"):
            self._latent_mean = torch.ones((self.n_samples, self.dim)).to(DEVICE)

    @property
    def _list_of_parameters_needing_gradient(self):
        """
        Property representing the list of parameters needing gradient.

        Returns
        -------
        list
            The list of parameters needing gradient.
        """
        return [self._latent_mean, self._latent_sqrt_var]

    def _get_max_components(self):
        """
        Method for getting the maximum number of components.

        Returns
        -------
        int
            The maximum number of components.
        """
        return self.dim

    def compute_elbo(self):
        """
        Method for computing the evidence lower bound (ELBO).

        Returns
        -------
        torch.Tensor
            The computed ELBO.
        """
        return profiled_elbo_pln(
            self._counts,
            self._covariates,
            self._offsets,
            self._latent_mean,
            self._latent_sqrt_var,
        )

    def _smart_init_model_parameters(self):
        """
        Method for smartly initializing the model parameters.
        """
        # no model parameters since we are doing a profiled ELBO
        pass

    def _random_init_model_parameters(self):
        """
        Method for randomly initializing the model parameters.
        """
        # no model parameters since we are doing a profiled ELBO
        pass

    @property
    def _coef(self):
        """
        Property representing the coefficients.

        Returns
        -------
        torch.Tensor
            The coefficients.
        """
        return _closed_formula_coef(self._covariates, self._latent_mean)

    @property
    def _covariance(self):
        """
        Property representing the covariance matrix.

        Returns
        -------
        torch.Tensor or None
            The covariance matrix or None.
        """
        return _closed_formula_covariance(
            self._covariates,
            self._latent_mean,
            self._latent_sqrt_var,
            self._coef,
            self.n_samples,
        )

    def _get_pca_low_dim_covariances(self, sk_components):
        components_var = (self._latent_sqrt_var**2).unsqueeze(
            1
        ) * sk_components.unsqueeze(0)
        covariances = components_var @ (sk_components.T.unsqueeze(0))
        return covariances

    def _pring_beginning_message(self):
        """
        Method for printing the beginning message.
        """
        print(f"Fitting a Pln model with {self._description}")

    @property
    def latent_variables(self):
        """
        Property representing the latent variables.

        Returns
        -------
        torch.Tensor
            The latent variables.
        """
        return self.latent_mean.detach()

    @property
    def number_of_parameters(self):
        """
        Property representing the number of parameters.

        Returns
        -------
        int
            The number of parameters.
        """
        return self.dim * (self.dim + self.nb_cov)

    def transform(self):
        """
        Method for transforming the model.

        Returns
        -------
        torch.Tensor
            The transformed model.
        """
        return self.latent_variables

    @property
    def covariance(self):
        """
        Property representing the covariance matrix.

        Returns
        -------
        torch.Tensor or None
            The covariance matrix or None.
        """
        if all(
            hasattr(self, attr)
            for attr in [
                "_covariates",
                "_latent_mean",
                "_latent_sqrt_var",
                "_coef",
                "n_samples",
            ]
        ):
            return self._covariance.cpu().detach()
        return None

    @covariance.setter
    def covariance(self, covariance):
        """
        Setter for the covariance property.

        Parameters
        ----------
        covariance : torch.Tensor
            The covariance matrix.
        """
        pass


class PlnPCAcollection:
    _NAME = "PlnPCAcollection"
    _dict_models: dict

    def __init__(
        self,
        counts: torch.Tensor,
        covariates: Optional[torch.Tensor] = None,
        offsets: Optional[torch.Tensor] = None,
        offsets_formula: str = "logsum",
        ranks: Iterable[int] = range(3, 5),
        dict_of_dict_initialization: Optional[dict] = None,
        take_log_offsets: bool = False,
        add_const: bool = True,
    ):
        """
        Constructor for PlnPCAcollection.

        Parameters
        ----------
        counts : torch.Tensor
            The counts.
        covariates : torch.Tensor, optional
            The covariates, by default None.
        offsets : torch.Tensor, optional
            The offsets, by default None.
        offsets_formula : str, optional
            The formula for offsets, by default "logsum".
        ranks : Iterable[int], optional
            The range of ranks, by default range(3, 5).
        dict_of_dict_initialization : dict, optional
            The dictionary of initialization, by default None.
        take_log_offsets : bool, optional
            Whether to take the logarithm of offsets, by default False.
        add_const: bool, optional
            Whether to add a column of one in the covariates. Defaults to True.
        """
        self._dict_models = {}
        (
            self._counts,
            self._covariates,
            self._offsets,
            self.column_counts,
        ) = _handle_data(
            counts, covariates, offsets, offsets_formula, take_log_offsets, add_const
        )
        self._fitted = False
        self._init_models(ranks, dict_of_dict_initialization)

    @classmethod
    def from_formula(
        cls,
        formula: str,
        data: dict,
        offsets_formula: str = "logsum",
        ranks: Iterable[int] = range(3, 5),
        dict_of_dict_initialization: Optional[dict] = None,
        take_log_offsets: bool = False,
    ) -> "PlnPCAcollection":
        """
        Create an instance of PlnPCAcollection from a formula.

        Parameters
        ----------
        formula : str
            The formula.
        data : dict
            The data dictionary.
        offsets_formula : str, optional
            The formula for offsets, by default "logsum".
        ranks : Iterable[int], optional
            The range of ranks, by default range(3, 5).
        dict_of_dict_initialization : dict, optional
            The dictionary of initialization, by default None.
        take_log_offsets : bool, optional
            Whether to take the logarithm of offsets, by default False.
        Returns
        -------
        PlnPCAcollection
            The created PlnPCAcollection instance.
        Examples
        --------
            >>> from pyPLNmodels import PlnPCAcollection, get_real_count_data
            >>> counts = get_real_count_data()
            >>> data = {"counts": counts}
            >>> pca_col = PlnPCAcollection.from_formula("counts ~ 1", data = data, ranks = [5,6])
        """
        counts, covariates, offsets = _extract_data_from_formula(formula, data)
        return cls(
            counts,
            covariates,
            offsets,
            offsets_formula,
            ranks,
            dict_of_dict_initialization,
            take_log_offsets,
            add_const=False,
        )

    @property
    def covariates(self) -> torch.Tensor:
        """
        Property representing the covariates.

        Returns
        -------
        torch.Tensor
            The covariates.
        """
        return self[self.ranks[0]].covariates

    @property
    def counts(self) -> torch.Tensor:
        """
        Property representing the counts.

        Returns
        -------
        torch.Tensor
            The counts.
        """
        return self[self.ranks[0]].counts

    @property
    def coef(self) -> Dict[int, torch.Tensor]:
        """
        Property representing the coefficients.

        Returns
        -------
        Dict[int, torch.Tensor]
            The coefficients.
        """
        return {model.rank: model.coef for model in self.values()}

    @property
    def components(self) -> Dict[int, torch.Tensor]:
        """
        Property representing the components.

        Returns
        -------
        Dict[int, torch.Tensor]
            The components.
        """
        return {model.rank: model.components for model in self.values()}

    @property
    def latent_mean(self) -> Dict[int, torch.Tensor]:
        """
        Property representing the latent means.

        Returns
        -------
        Dict[int, torch.Tensor]
            The latent means.
        """
        return {model.rank: model.latent_mean for model in self.values()}

    @property
    def latent_sqrt_var(self) -> Dict[int, torch.Tensor]:
        """
        Property representing the latent variances.

        Returns
        -------
        Dict[int, torch.Tensor]
            The latent variances.
        """
        return {model.rank: model.latent_sqrt_var for model in self.values()}

    @counts.setter
    @_array2tensor
    def counts(self, counts: torch.Tensor):
        """
        Setter for the counts property.

        Parameters
        ----------
        counts : torch.Tensor
            The counts.
        """
        for model in self.values():
            model.counts = counts

    @coef.setter
    @_array2tensor
    def coef(self, coef: torch.Tensor):
        """
        Setter for the coef property.

        Parameters
        ----------
        coef : torch.Tensor
            The coefficients.
        """
        for model in self.values():
            model.coef = coef

    @covariates.setter
    @_array2tensor
    def covariates(self, covariates: torch.Tensor):
        """
        Setter for the covariates property.

        Parameters
        ----------
        covariates : torch.Tensor
            The covariates.
        """
        for model in self.values():
            model.covariates = covariates

    @property
    def offsets(self) -> torch.Tensor:
        """
        Property representing the offsets.

        Returns
        -------
        torch.Tensor
            The offsets.
        """
        return self[self.ranks[0]].offsets

    @offsets.setter
    @_array2tensor
    def offsets(self, offsets: torch.Tensor):
        """
        Setter for the offsets property.

        Parameters
        ----------
        offsets : torch.Tensor
            The offsets.
        """
        for model in self.values():
            model.offsets = offsets

    def _init_models(
        self, ranks: Iterable[int], dict_of_dict_initialization: Optional[dict]
    ):
        """
        Method for initializing the models.

        Parameters
        ----------
        ranks : Iterable[int]
            The range of ranks.
        dict_of_dict_initialization : dict, optional
            The dictionary of initialization.
        """
        if isinstance(ranks, (Iterable, np.ndarray)):
            for rank in ranks:
                if isinstance(rank, (int, np.integer)):
                    dict_initialization = _get_dict_initialization(
                        rank, dict_of_dict_initialization
                    )
                    self._dict_models[rank] = PlnPCA(
                        counts=self._counts,
                        covariates=self._covariates,
                        offsets=self._offsets,
                        rank=rank,
                        dict_initialization=dict_initialization,
                    )
                else:
                    raise TypeError(
                        "Please instantiate with either a list "
                        "of integers or an integer."
                    )
        elif isinstance(ranks, (int, np.integer)):
            dict_initialization = _get_dict_initialization(
                ranks, dict_of_dict_initialization
            )
            self._dict_models[rank] = PlnPCA(
                self._counts,
                self._covariates,
                self._offsets,
                ranks,
                dict_initialization,
            )

        else:
            raise TypeError(
                f"Please instantiate with either a list " f"of integers or an integer."
            )

    @property
    def ranks(self) -> List[int]:
        """
        Property representing the ranks.

        Returns
        -------
        List[int]
            The ranks.
        """
        return [model.rank for model in self.values()]

    def _pring_beginning_message(self) -> str:
        """
        Method for printing the beginning message.

        Returns
        -------
        str
            The beginning message.
        """
        return f"Adjusting {len(self.ranks)} Pln models for PCA analysis \n"

    @property
    def dim(self) -> int:
        """
        Property representing the dimension.

        Returns
        -------
        int
            The dimension.
        """
        return self[self.ranks[0]].dim

    @property
    def nb_cov(self) -> int:
        """
        Property representing the number of covariates.

        Returns
        -------
        int
            The number of covariates.
        """
        return self[self.ranks[0]].nb_cov

    def fit(
        self,
        nb_max_iteration: int = 100000,
        lr: float = 0.01,
        class_optimizer: Type[torch.optim.Optimizer] = torch.optim.Rprop,
        tol: float = 1e-3,
        do_smart_init: bool = True,
        verbose: bool = False,
    ):
        """
        Fit the PlnPCAcollection.

        Parameters
        ----------
        nb_max_iteration : int, optional
            The maximum number of iterations, by default 100000.
        lr : float, optional
            The learning rate, by default 0.01.
        class_optimizer : Type[torch.optim.Optimizer], optional
            The optimizer class, by default torch.optim.Rprop.
        tol : float, optional
            The tolerance, by default 1e-3.
        do_smart_init : bool, optional
            Whether to do smart initialization, by default True.
        verbose : bool, optional
            Whether to print verbose output, by default False.
        """
        self._pring_beginning_message()
        for i in range(len(self.values())):
            model = self[self.ranks[i]]
            model.fit(
                nb_max_iteration,
                lr,
                class_optimizer,
                tol,
                do_smart_init,
                verbose,
            )
            if i < len(self.values()) - 1:
                next_model = self[self.ranks[i + 1]]
                self._init_next_model_with_current_model(next_model, model)
        self._print_ending_message()

    def _init_next_model_with_current_model(self, next_model: Any, current_model: Any):
        """
        Initialize the next model with the parameters of the current model.

        Parameters
        ----------
        next_model : Any
            The next model to initialize.
        current_model : Any
            The current model.
        """
        next_model.coef = current_model.coef
        next_model.components = torch.zeros(self.dim, next_model.rank)
        with torch.no_grad():
            next_model._components[:, : current_model.rank] = current_model._components

    def _print_ending_message(self):
        """
        Method for printing the ending message.
        """
        delimiter = "=" * NB_CHARACTERS_FOR_NICE_PLOT
        print(f"{delimiter}\n")
        print("DONE!")
        print(f"    Best model(lower BIC): {self._criterion_dict('BIC')}\n ")
        print(f"    Best model(lower AIC): {self._criterion_dict('AIC')}\n ")
        print(f"{delimiter}\n")

    def _criterion_dict(self, criterion: str = "AIC") -> int:
        """
        Return the rank of the best model according to the specified criterion.

        Parameters
        ----------
        criterion : str, optional
            The criterion to use ('AIC' or 'BIC'), by default 'AIC'.

        Returns
        -------
        int
            The rank of the best model.
        """
        return self.best_model(criterion).rank

    def __getitem__(self, rank: int) -> Any:
        """
        Get the model with the specified rank.

        Parameters
        ----------
        rank : int
            The rank of the model.

        Returns
        -------
        Any
            The model with the specified rank.
        """
        return self._dict_models[rank]

    def __len__(self) -> int:
        """
        Get the number of models in the collection.

        Returns
        -------
        int
            The number of models in the collection.
        """
        return len(self._dict_models)

    def __iter__(self):
        """
        Iterate over the models in the collection.

        Returns
        -------
        Iterator
            Iterator over the models.
        """
        return iter(self._dict_models)

    def __contains__(self, rank: int) -> bool:
        """
        Check if a model with the specified rank exists in the collection.

        Parameters
        ----------
        rank : int
            The rank to check.

        Returns
        -------
        bool
            True if a model with the specified rank exists, False otherwise.
        """
        return rank in self._dict_models.keys()

    def keys(self):
        """
        Get the ranks of the models in the collection.

        Returns
        -------
        KeysView
            The ranks of the models.
        """
        return self._dict_models.keys()

    def get(self, key: Any, default: Any) -> Any:
        """
        Get the model with the specified key, or return a default value if the key does not exist.

        Parameters
        ----------
        key : Any
            The key to search for.
        default : Any
            The default value to return if the key does not exist.

        Returns
        -------
        Any
            The model with the specified key, or the default value if the key does not exist.
        """
        if key in self:
            return self[key]
        else:
            return default

    def values(self):
        """
        Get the models in the collection.

        Returns
        -------
        ValuesView
            The models in the collection.
        """
        return self._dict_models.values()

    def items(self):
        """
        Get the key-value pairs of the models in the collection.

        Returns
        -------
        ItemsView
            The key-value pairs of the models.
        """
        return self._dict_models.items()

    @property
    def BIC(self) -> Dict[int, int]:
        """
        Property representing the BIC scores of the models in the collection.

        Returns
        -------
        Dict[int, int]
            The BIC scores of the models.
        """
        return {model.rank: int(model.BIC) for model in self.values()}

    @property
    def AIC(self) -> Dict[int, int]:
        """
        Property representing the AIC scores of the models in the collection.

        Returns
        -------
        Dict[int, int]
            The AIC scores of the models.
        """
        return {model.rank: int(model.AIC) for model in self.values()}

    @property
    def loglikes(self) -> Dict[int, Any]:
        """
        Property representing the log-likelihoods of the models in the collection.

        Returns
        -------
        Dict[int, Any]
            The log-likelihoods of the models.
        """
        return {model.rank: model.loglike for model in self.values()}

    def show(self):
        """
        Show a plot with BIC scores, AIC scores, and negative log-likelihoods of the models.
        """
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
    def best_BIC_model_rank(self) -> int:
        """
        Property representing the rank of the best model according to the BIC criterion.

        Returns
        -------
        int
            The rank of the best model.
        """
        return self.ranks[np.argmin(list(self.BIC.values()))]

    @property
    def best_AIC_model_rank(self) -> int:
        """
        Property representing the rank of the best model according to the AIC criterion.

        Returns
        -------
        int
            The rank of the best model.
        """
        return self.ranks[np.argmin(list(self.AIC.values()))]

    def best_model(self, criterion: str = "AIC") -> Any:
        """
        Get the best model according to the specified criterion.

        Parameters
        ----------
        criterion : str, optional
            The criterion to use ('AIC' or 'BIC'), by default 'AIC'.

        Returns
        -------
        Any
            The best model.
        """
        if criterion == "BIC":
            return self[self.best_BIC_model_rank]
        if criterion == "AIC":
            return self[self.best_AIC_model_rank]
        raise ValueError(f"Unknown criterion {criterion}")

    def save(self, path_of_directory: str = "./", ranks: Optional[List[int]] = None):
        """
        Save the models in the specified directory.

        Parameters
        ----------
        path_of_directory : str, optional
            The path of the directory to save the models, by default "./".
        ranks : Optional[List[int]], optional
            The ranks of the models to save, by default None.
        """
        if ranks is None:
            ranks = self.ranks
        for model in self.values():
            if model.rank in ranks:
                model.save(f"{self._directory_name}/PlnPCA_rank_{model.rank}")

    @property
    def _directory_name(self) -> str:
        """
        Property representing the directory name.

        Returns
        -------
        str
            The directory name.
        """
        return f"{self._NAME}_nbcov_{self.nb_cov}_dim_{self.dim}"

    @property
    def n_samples(self) -> int:
        """
        Property representing the number of samples.

        Returns
        -------
        int
            The number of samples.
        """
        return self[self.ranks[0]].n_samples

    def __repr__(self) -> str:
        """
        Return a string representation of the PlnPCAcollection object.

        Returns
        -------
        str
            The string representation of the PlnPCAcollection object.
        """
        nb_models = len(self)
        delimiter = "\n" + "-" * NB_CHARACTERS_FOR_NICE_PLOT + "\n"
        to_print = delimiter
        to_print += f"Collection of {nb_models} PlnPCAcollection models with \
                    {self.dim} variables."
        to_print += delimiter
        to_print += f" - Ranks considered:{self.ranks}\n"
        dict_bic = {"rank": "criterion"} | self.BIC
        to_print += f" - BIC metric:\n{_nice_string_of_dict(dict_bic)}\n"
        rank_bic = self.best_model(criterion="BIC")._rank
        to_print += f"   Best model(lower BIC): {rank_bic}\n \n"
        dict_aic = {"rank": "criterion"} | self.AIC
        to_print += f" - AIC metric:\n{_nice_string_of_dict(dict_aic)}\n"
        rank_aic = self.best_model(criterion="AIC")._rank
        to_print += f"   Best model(lower AIC): {rank_aic}"
        to_print += delimiter
        to_print += "* Useful properties\n"
        to_print += f"    {self._useful_properties_string}\n"
        to_print += "* Useful methods \n"
        to_print += f"    {self._useful_methods_strings}"
        to_print += delimiter
        return to_print

    @property
    def _useful_methods_strings(self) -> str:
        """
        Property representing the useful methods.

        Returns
        -------
        str
            The string representation of the useful methods.
        """
        return ".show(), .best_model()"

    @property
    def _useful_properties_string(self) -> str:
        """
        Property representing the useful properties.

        Returns
        -------
        str
            The string representation of the useful properties.
        """
        return ".BIC, .AIC, .loglikes"


# Here, setting the value for each key in _dict_parameters
class PlnPCA(model):
    _NAME: str = "PlnPCA"
    _components: torch.Tensor

    def __init__(
        self,
        counts: torch.Tensor,
        covariates: Optional[torch.Tensor] = None,
        offsets: Optional[torch.Tensor] = None,
        offsets_formula: str = "logsum",
        rank: int = 5,
        dict_initialization: Optional[Dict[str, torch.Tensor]] = None,
        take_log_offsets: bool = False,
        add_const: bool = True,
    ):
        """
        Initialize the PlnPCA object.

        Parameters
        ----------
        counts : torch.Tensor
            The counts tensor.
        covariates : torch.Tensor, optional
            The covariates tensor, by default None.
        offsets : torch.Tensor, optional
            The offsets tensor, by default None.
        offsets_formula : str, optional
            The offsets formula, by default "logsum".
        rank : int, optional
            The rank of the approximation, by default 5.
        dict_initialization : Dict[str, torch.Tensor], optional
            The dictionary for initialization, by default None.
        take_log_offsets : bool, optional
            Whether to take the log of offsets. Defaults to False.
        add_const: bool, optional
            Whether to add a column of one in the covariates. Defaults to True.
        """
        self._rank = rank
        super().__init__(
            counts=counts,
            covariates=covariates,
            offsets=offsets,
            offsets_formula=offsets_formula,
            dict_initialization=dict_initialization,
            take_log_offsets=take_log_offsets,
            add_const=add_const,
        )

    @classmethod
    def from_formula(
        cls,
        formula: str,
        data: Any,
        rank: int = 5,
        offsets_formula: str = "logsum",
        dict_initialization: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """
        Create a PlnPCA object from a formula.

        Parameters
        ----------
        formula : str
            The formula.
        data : Any
            The data.
        rank : int, optional
            The rank of the approximation, by default 5.
        offsets_formula : str, optional
            The offsets formula, by default "logsum".
        dict_initialization : Dict[str, torch.Tensor], optional
            The dictionary for initialization, by default None.

        Returns
        -------
        PlnPCA
            The created PlnPCA object.
        Examples
        --------
            >>> from pyPLNmodels import PlnPCA, get_real_count_data
            >>> counts = get_real_count_data()
            >>> data = {"counts": counts}
            >>> pca_col = PlnPCA.from_formula("counts ~ 1", data = data, rank = [5,6])
        """
        counts, covariates, offsets = _extract_data_from_formula(formula, data)
        return cls(
            counts,
            covariates,
            offsets,
            offsets_formula,
            rank,
            dict_initialization,
            add_const=False,
        )

    def _check_if_rank_is_too_high(self):
        """
        Check if the rank is too high and issue a warning if necessary.
        """
        if self.dim < self.rank:
            warning_string = (
                f"\nThe requested rank of approximation {self.rank} "
                f"is greater than the number of variables {self.dim}. "
                f"Setting rank to {self.dim}"
            )
            warnings.warn(warning_string)
            self._rank = self.dim

    @property
    def latent_mean(self) -> torch.Tensor:
        """
        Property representing the latent mean.

        Returns
        -------
        torch.Tensor
            The latent mean tensor.
        """
        return self._cpu_attribute_or_none("_latent_mean")

    @property
    def latent_sqrt_var(self) -> torch.Tensor:
        """
        Property representing the unsigned square root of the latent variance.

        Returns
        -------
        torch.Tensor
            The latent variance tensor.
        """
        return self._cpu_attribute_or_none("_latent_sqrt_var")

    @property
    def _latent_var(self) -> torch.Tensor:
        """
        Property representing the latent variance.

        Returns
        -------
        torch.Tensor
            The latent variance tensor.
        """
        return self._latent_sqrt_var**2

    @latent_mean.setter
    @_array2tensor
    def latent_mean(self, latent_mean: torch.Tensor):
        """
        Setter for the latent mean.

        Parameters
        ----------
        latent_mean : torch.Tensor
            The latent mean tensor.
        """
        if latent_mean.shape != (self.n_samples, self.rank):
            raise ValueError(
                f"Wrong shape. Expected {self.n_samples, self.rank}, got {latent_mean.shape}"
            )
        self._latent_mean = latent_mean

    @latent_sqrt_var.setter
    @_array2tensor
    def latent_sqrt_var(self, latent_sqrt_var: torch.Tensor):
        """
        Setter for the latent variance.

        Parameters
        ----------
        latent_sqrt_var : torch.Tensor
            The latent variance tensor.
        """
        if latent_sqrt_var.shape != (self.n_samples, self.rank):
            raise ValueError(
                f"Wrong shape. Expected {self.n_samples, self.rank}, got {latent_sqrt_var.shape}"
            )
        self._latent_sqrt_var = latent_sqrt_var

    @property
    def _directory_name(self) -> str:
        """
        Property representing the directory name.

        Returns
        -------
        str
            The directory name.
        """
        return f"{self._NAME}_nbcov_{self.nb_cov}_rank_{self._rank}"

    @property
    def covariates(self) -> torch.Tensor:
        """
        Property representing the covariates.

        Returns
        -------
        torch.Tensor
            The covariates tensor.
        """
        return self._cpu_attribute_or_none("_covariates")

    @covariates.setter
    @_array2tensor
    def covariates(self, covariates: torch.Tensor):
        """
        Setter for the covariates.

        Parameters
        ----------
        covariates : torch.Tensor
            The covariates tensor.
        """
        _check_data_shape(self.counts, covariates, self.offsets)
        self._covariates = covariates
        print("Setting coef to initialization")
        self._smart_init_coef()

    def _get_pca_low_dim_covariances(self, sk_components):
        C_tilde_C = sk_components @ self._components
        C_tilde_C_latent_var = C_tilde_C.unsqueeze(0) * (self._latent_var.unsqueeze(1))
        covariances = (C_tilde_C_latent_var) @ (C_tilde_C.T.unsqueeze(0))
        return covariances

    @property
    def rank(self) -> int:
        """
        Property representing the rank.

        Returns
        -------
        int
            The rank.
        """
        return self._rank

    def _get_max_components(self) -> int:
        """
        Get the maximum number of components.

        Returns
        -------
        int
            The maximum number of components.
        """
        return self._rank

    def _pring_beginning_message(self):
        """
        Print the beginning message.
        """
        print("-" * NB_CHARACTERS_FOR_NICE_PLOT)
        print(f"Fitting a PlnPCAcollection model with {self._rank} components")

    @property
    def model_parameters(self) -> Dict[str, torch.Tensor]:
        """
        Property representing the model parameters.

        Returns
        -------
        Dict[str, torch.Tensor]
            The model parameters.
        """
        return {"coef": self.coef, "components": self.components}

    def _smart_init_model_parameters(self):
        """
        Initialize the model parameters smartly.
        """
        if not hasattr(self, "_coef"):
            super()._smart_init_coef()
        if not hasattr(self, "_components"):
            self._components = _init_components(
                self._counts, self._covariates, self._coef, self._rank
            )

    def _random_init_model_parameters(self):
        """
        Randomly initialize the model parameters.
        """
        super()._random_init_coef()
        self._components = torch.randn((self.dim, self._rank)).to(DEVICE)

    def _random_init_latent_parameters(self):
        """
        Randomly initialize the latent parameters.
        """
        self._latent_sqrt_var = (
            1 / 2 * torch.ones((self.n_samples, self._rank)).to(DEVICE)
        )
        self._latent_mean = torch.ones((self.n_samples, self._rank)).to(DEVICE)

    def _smart_init_latent_parameters(self):
        """
        Initialize the latent parameters smartly.
        """
        if not hasattr(self, "_latent_mean"):
            self._latent_mean = (
                _init_latent_mean(
                    self._counts,
                    self._covariates,
                    self._offsets,
                    self._coef,
                    self._components,
                )
                .to(DEVICE)
                .detach()
            )
        if not hasattr(self, "_latent_sqrt_var"):
            self._latent_sqrt_var = (
                1 / 2 * torch.ones((self.n_samples, self._rank)).to(DEVICE)
            )

    @property
    def _list_of_parameters_needing_gradient(self):
        """
        Property representing the list of parameters needing gradient.

        Returns
        -------
        List[torch.Tensor]
            The list of parameters needing gradient.
        """
        if self._coef is None:
            return [self._components, self._latent_mean, self._latent_sqrt_var]
        return [self._components, self._coef, self._latent_mean, self._latent_sqrt_var]

    def compute_elbo(self) -> torch.Tensor:
        """
        Compute the evidence lower bound (ELBO).

        Returns
        -------
        torch.Tensor
            The ELBO value.
        """
        return elbo_plnpca(
            self._counts,
            self._covariates,
            self._offsets,
            self._latent_mean,
            self._latent_sqrt_var,
            self._components,
            self._coef,
        )

    @property
    def number_of_parameters(self) -> int:
        """
        Property representing the number of parameters.

        Returns
        -------
        int
            The number of parameters.
        """
        return self.dim * (self.nb_cov + self._rank) - self._rank * (self._rank - 1) / 2

    @property
    def _additional_properties_string(self) -> str:
        """
        Property representing the additional properties string.

        Returns
        -------
        str
            The additional properties string.
        """
        return ".projected_latent_variables"

    @property
    def _additional_methods_string(self) -> str:
        """
        Property representing the additional methods string.

        Returns
        -------
        str
            The additional methods string.
        """
        string = " .projected_latent_variables"
        return string

    @property
    def covariance(self) -> Optional[torch.Tensor]:
        """
        Property representing the covariance.

        Returns
        -------
        Optional[torch.Tensor]
            The covariance tensor or None if components are not present.
        """
        if hasattr(self, "_components"):
            cov_latent = self._latent_mean.T @ self._latent_mean
            cov_latent += torch.diag(
                torch.sum(torch.square(self._latent_sqrt_var), dim=0)
            )
            cov_latent /= self.n_samples
            return (self._components @ cov_latent @ self._components.T).cpu().detach()
        return None

    @property
    def _description(self) -> str:
        """
        Property representing the description.

        Returns
        -------
        str
            The description string.
        """
        return f" {self.rank} principal component."

    @property
    def latent_variables(self) -> torch.Tensor:
        """
        Property representing the latent variables.

        Returns
        -------
        torch.Tensor
            The latent variables of size (n_samples, dim).
        """
        return torch.matmul(self._latent_mean, self._components.T).detach()

    @property
    def projected_latent_variables(self) -> torch.Tensor:
        """
        Property representing the projected latent variables.

        Returns
        -------
        torch.Tensor
            The projected latent variables.
        """
        return torch.mm(self.latent_variables, self.ortho_components).detach().cpu()

    @property
    def ortho_components(self):
        """
        Orthogonal components of the model.
        """
        return torch.linalg.qr(self._components, "reduced")[0]

    def pca_projected_latent_variables(
        self, n_components: Optional[int] = None
    ) -> np.ndarray:
        """
        Perform PCA on projected latent variables.

        Parameters
        ----------
        n_components : Optional[int]
            Number of components to keep. Defaults to None.

        Returns
        -------
        np.ndarray
            The transformed projected latent variables.
        Raises
        ------
        ValueError
           If the number of components asked is greater than the number of dimensions.
        """
        if n_components is None:
            n_components = self._get_max_components()
        if n_components > self.rank:
            raise ValueError(
                f"You ask more components ({n_components}) than maximum rank ({self.rank})"
            )
        pca = PCA(n_components=n_components)
        return pca.fit_transform(self.latent_variables.cpu())

    @property
    def components(self) -> torch.Tensor:
        """
        Property representing the components.

        Returns
        -------
        torch.Tensor
            The components.
        """
        return self._cpu_attribute_or_none("_components")

    @components.setter
    @_array2tensor
    def components(self, components: torch.Tensor):
        """
        Setter for the components.

        Parameters
        ----------
        components : torch.Tensor
            The components to set.

        Raises
        ------
        ValueError
            If the components have an invalid shape.
        """
        if components.shape != (self.dim, self.rank):
            raise ValueError(
                f"Wrong shape. Expected {self.dim, self.rank}, got {components.shape}"
            )
        self._components = components

    def transform(self, project: bool = True) -> torch.Tensor:
        """
        Transform the model.

        Parameters
        ----------
        project : bool, optional
            Whether to project the latent variables, by default True.

        Returns
        -------
        torch.Tensor
            The transformed model.
        """
        if project is True:
            return self.projected_latent_variables
        return self.latent_variables


class ZIPln(Pln):
    _NAME = "ZIPln"

    _pi: torch.Tensor
    _coef_inflation: torch.Tensor
    _dirac: torch.Tensor

    @property
    def _description(self):
        return "with full covariance model and zero-inflation."

    def _random_init_model_parameters(self):
        super()._random_init_model_parameters()
        self._coef_inflation = torch.randn(self.nb_cov, self.dim)
        self._covariance = torch.diag(torch.ones(self.dim)).to(DEVICE)

    # should change the good initialization, especially for _coef_inflation
    def _smart_init_model_parameters(self):
        super()._smart_init_model_parameters()
        if not hasattr(self, "_covariance"):
            self._covariance = _init_covariance(
                self._counts, self._covariates, self._coef
            )
        if not hasattr(self, "_coef_inflation"):
            self._coef_inflation = torch.randn(self.nb_cov, self.dim)

    def _random_init_latent_parameters(self):
        self._dirac = self._counts == 0
        self._latent_mean = torch.randn(self.n_samples, self.dim)
        self._latent_sqrt_var = torch.randn(self.n_samples, self.dim)
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
            self._latent_sqrt_var,
            self._pi,
            self._covariance,
            self._coef,
            self._coef_inflation,
            self._dirac,
        )

    @property
    def _list_of_parameters_needing_gradient(self):
        return [self._latent_mean, self._latent_sqrt_var, self._coef_inflation]

    def _update_closed_forms(self):
        self._coef = _closed_formula_coef(self._covariates, self._latent_mean)
        self._covariance = _closed_formula_covariance(
            self._covariates,
            self._latent_mean,
            self._latent_sqrt_var,
            self._coef,
            self.n_samples,
        )
        self._pi = _closed_formula_pi(
            self._offsets,
            self._latent_mean,
            self._latent_sqrt_var,
            self._dirac,
            self._covariates,
            self._coef_inflation,
        )

    @property
    def number_of_parameters(self):
        return self.dim * (2 * self.nb_cov + (self.dim + 1) / 2)
