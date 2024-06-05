import time
from abc import ABC, abstractmethod
import warnings
import os
from typing import Optional, Dict, List, Type, Any, Iterable, Union, Literal
from tqdm import tqdm

import pandas as pd
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib
from scipy import stats

from pyPLNmodels._closed_forms import (
    _closed_formula_coef,
    _closed_formula_covariance,
    _closed_formula_latent_prob,
    _closed_formula_zero_grad_prob,
)
from pyPLNmodels.elbos import (
    elbo_plnpca,
    elbo_zi_pln,
    profiled_elbo_pln,
    elbo_brute_zipln_components,
    elbo_brute_zipln_covariance,
    per_sample_elbo_plnpca,
)
from pyPLNmodels._utils import (
    _CriterionArgs,
    _format_data,
    _nice_string_of_dict,
    _plot_ellipse,
    _check_data_shape,
    _extract_data_from_formula_no_infla,
    _extract_data_from_formula_with_infla,
    _get_dict_initialization,
    _array2tensor,
    _handle_data,
    _handle_data_with_inflation,
    _add_doc,
    plot_correlation_circle,
    _check_formula,
    _pca_pairplot,
    _check_right_exog_inflation_shape,
    mse,
)

from pyPLNmodels._initialization import (
    _init_components,
    _init_coef,
    _init_latent_mean,
    _init_coef_coef_inflation,
)

if torch.cuda.is_available():
    DEVICE = "cuda"
    print("Using a GPU.")
else:
    DEVICE = "cpu"
# shoudl add a good init for M. for pln we should not put
# the maximum of the log posterior, for plnpca it may be ok.

NB_CHARACTERS_FOR_NICE_PLOT = 70


class _model(ABC):
    """
    Base class for all the Pln models. Should be inherited.
    """

    _endog: torch.Tensor
    _exog: torch.Tensor
    _offsets: torch.Tensor
    _coef: torch.Tensor
    _beginning_time: float
    _latent_sqrt_var: torch.Tensor
    _latent_mean: torch.Tensor
    _batch_size: int = None

    def __init__(
        self,
        endog: Union[torch.Tensor, np.ndarray, pd.DataFrame],
        *,
        exog: Optional[Union[torch.Tensor, np.ndarray, pd.DataFrame]] = None,
        offsets: Optional[Union[torch.Tensor, np.ndarray, pd.DataFrame]] = None,
        offsets_formula: {"zero", "logsum"} = "zero",
        dict_initialization: Optional[dict] = None,
        take_log_offsets: bool = False,
        add_const: bool = True,
        batch_size: int = None,
    ):
        """
        Initializes the model class.

        Parameters
        ----------
        endog : Union[torch.Tensor, np.ndarray, pd.DataFrame]
            The count data.
        exog : Union[torch.Tensor, np.ndarray, pd.DataFrame], optional(keyword-only)
            The covariate data. Defaults to None.
        offsets : Union[torch.Tensor, np.ndarray, pd.DataFrame], optional(keyword-only)
            The offsets data. Defaults to None.
        offsets_formula : str, optional(keyword-only)
            The formula for offsets. Defaults to "zero". Can be also "logsum" where we take the logarithm of the sum (of each line) of the counts.
            Overriden (useless) if offsets is not None.
        dict_initialization : dict, optional(keyword-only)
            The initialization dictionary. Defaults to None.
        take_log_offsets : bool, optional(keyword-only)
            Whether to take the log of offsets. Defaults to False.
        add_const: bool, optional(keyword-only)
            Whether to add a column of one in the exog. Defaults to True.
        batch_size: int, optional(keyword-only)
            The batch size when optimizing the elbo. If None,
            batch gradient descent will be performed (i.e. batch_size = n_samples).
        """
        (
            self._endog,
            self._exog,
            self._offsets,
            self.column_endog,
            self.samples_only_zeros,
            _,
            self._batch_size,
        ) = _handle_data(
            endog,
            exog,
            offsets,
            offsets_formula,
            take_log_offsets,
            add_const,
            batch_size,
        )
        self._fitted = False
        self._criterion_args = _CriterionArgs()
        if dict_initialization is not None:
            self._set_init_parameters(dict_initialization)

    @classmethod
    def from_formula(
        cls,
        formula: str,
        data: dict[str : Union[torch.Tensor, np.ndarray, pd.DataFrame]],
        *,
        offsets_formula: {"zero", "logsum"} = "zero",
        dict_initialization: Optional[dict] = None,
        take_log_offsets: bool = False,
        batch_size: int = None,
    ):
        """
        Create a model instance from a formula and data.

        Parameters
        ----------
        formula : str
            The formula.
        data : dict
            The data dictionary. Each value can be either a torch.Tensor,
            a np.ndarray or pd.DataFrame
        offsets_formula : str, optional(keyword-only)
            The formula for offsets. Defaults to "zero". Can be also "logsum" where we take
            the logarithm of the sum (of each line) of the counts.
            Overriden (useless) if data["offsets"] is not None.
        dict_initialization : dict, optional(keyword-only)
            The initialization dictionary. Defaults to None.
        take_log_offsets : bool, optional(keyword-only)
            Whether to take the log of offsets. Defaults to False.
        batch_size: int, optional(keyword-only)
            The batch size when optimizing the elbo. If None,
            batch gradient descent will be performed (i.e. batch_size = n_samples).
        """
        endog, exog, offsets = _extract_data_from_formula_no_infla(formula, data)
        return cls(
            endog,
            exog=exog,
            offsets=offsets,
            offsets_formula=offsets_formula,
            dict_initialization=dict_initialization,
            take_log_offsets=take_log_offsets,
            add_const=False,
            batch_size=batch_size,
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
            dict_initialization["coef"] = None
        if self._NAME == "Pln":
            del dict_initialization["covariance"]
            del dict_initialization["coef"]
        for key, array in dict_initialization.items():
            if array is not None:
                print(array.shape)
            array = _format_data(array)
            setattr(self, key, array)
        self._fitted = True

    @property
    def batch_size(self) -> int:
        """
        The batch size of the model. Should not be greater than the number of samples.
        """
        if self._batch_size is None:
            return self.n_samples
        return self._batch_size

    @property
    def _onlyonebatch(self):
        return self._batch_size == self.n_samples

    @property
    def useful_indices(self):
        return ~self.samples_only_zeros

    @batch_size.setter
    def batch_size(self, batch_size: int):
        raise ValueError("Can not set the batch size.")

    def viz(self, *, ax=None, colors=None, show_cov: bool = False):
        """
        Visualize the gaussian latent variables with a classic PCA.

        Parameters
        ----------
        ax : Optional[matplotlib.axes.Axes], optional(keyword-only)
            The matplotlib axis to use. If None, the current axis is used, by default None.
        colors : Optional[np.ndarray], optional(keyword-only)
            The colors to use for plotting, by default None.
        show_cov: bool, Optional(keyword-only)
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
        variables = self.transform()
        return self._viz_variables(variables, ax=ax, colors=colors, show_cov=show_cov)

    def viz_positions(self, *, ax=None, colors=None, show_cov: bool = False):
        variables = self.latent_position
        return self._viz_variables(variables, ax=ax, colors=colors, show_cov=show_cov)

    @property
    def latent_position(self):
        return self.transform() - self.mean_gaussian

    def _viz_variables(
        self, variables, *, ax=None, colors=None, show_cov: bool = False
    ):
        """
        Visualize variables with a classic PCA.

        Parameters
        ----------
        variables: torch.Tensor
            The variables that need to be visualize
        ax : Optional[matplotlib.axes.Axes], optional(keyword-only)
            The matplotlib axis to use. If None, the current axis is used, by default None.
        colors : Optional[np.ndarray], optional(keyword-only)
            The colors to use for plotting, by default None.
        show_cov: bool, Optional(keyword-only)
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
        if self._get_max_components() < 2:
            raise RuntimeError("Can't perform visualization for dim < 2.")
        pca = PCA(n_components=2)
        pca.fit(variables)
        proj_variables = pca.transform(self.transform())
        x = proj_variables[:, 0]
        y = proj_variables[:, 1]
        if ax is None:
            ax = plt.gca()
        sns.scatterplot(x=x, y=y, hue=colors, ax=ax)
        if show_cov is True:
            sk_components = torch.from_numpy(pca.components_)
            covariances = self._get_pca_low_dim_covariances(sk_components).detach()
            for i in range(covariances.shape[0]):
                _plot_ellipse(x[i], y[i], cov=covariances[i], ax=ax)
        plt.show()
        return ax

    def _project_parameters(self):
        pass

    @property
    def nb_iteration_done(self) -> int:
        """
        The number of iterations done.

        Returns
        -------
        int
            The number of iterations done.
        """
        return len(self._criterion_args._elbos_list) * self.nb_batches

    @property
    def n_samples(self) -> int:
        """
        The number of samples, i.e. the first dimension of the endog.

        Returns
        -------
        int
            The number of samples.
        """
        return self._endog.shape[0]

    @property
    def dim(self) -> int:
        """
        The second dimension of the endog.

        Returns
        -------
        int
            The second dimension of the endog.
        """
        return self._endog.shape[1]

    @property
    def nb_cov(self) -> int:
        """
        The number of exog.

        Returns
        -------
        int
            The number of exog.
        """
        if self.exog is None:
            return 0
        return self.exog.shape[1]

    def _smart_init_coef(self):
        """
        Initialize coefficients smartly.
        """
        coef = _init_coef(self._endog, self._exog, self._offsets)
        if coef is not None:
            self._coef = coef.detach().to(DEVICE)
        else:
            self._coef = None

    def _random_init_coef(self):
        """
        Randomly initialize coefficients.
        """
        if self.nb_cov == 0:
            self._coef = None
        self._coef = torch.randn((self.nb_cov, self.dim)).to(DEVICE)

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

    def _set_requiring_grad_true(self):
        """
        Move parameters to the GPU device if present.
        """
        for parameter in self._list_of_parameters_needing_gradient:
            parameter.requires_grad_(True)

    def fit(
        self,
        nb_max_epoch: int = 400,
        *,
        lr: float = 0.01,
        tol: float = 1e-5,
        do_smart_init: bool = True,
        verbose: bool = False,
    ):
        """
        Fit the model. The lower tol, the more accurate the model.

        Parameters
        ----------
        nb_max_epoch : int, optional
            The maximum number of epochs (iterations). Defaults to 400.
        lr : float, optional(keyword-only)
            The learning rate. Defaults to 0.01.
        tol : float, optional(keyword-only)
            The tolerance for convergence. Defaults to 1e-5.
        do_smart_init : bool, optional(keyword-only)
            Whether to perform smart initialization. Defaults to True.
        verbose : bool, optional(keyword-only)
            Whether to print training progress.  Defaults to False.
        Raises
        ------
        ValueError
            If 'nb_max_epoch' is not an int.
        """
        if not isinstance(nb_max_epoch, int):
            raise ValueError("The argument 'nb_max_epoch' should be an int.")
        self._print_beginning_message()
        self._beginning_time = time.time()
        if self._fitted is False:
            self._init_parameters(do_smart_init)
        elif len(self._criterion_args.running_times) > 0:
            self._beginning_time -= self._criterion_args.running_times[-1]
        self._set_requiring_grad_true()
        self._handle_optimizer(lr, nb_max_epoch)
        stop_condition = False
        self._dict_mse = {name_model: [] for name_model in self.model_parameters.keys()}
        nb_epoch_done = 0
        pbar = tqdm(desc="Estimated remaining time", total=nb_max_epoch)
        while nb_epoch_done < nb_max_epoch and not stop_condition:
            loss = self._trainstep()
            criterion = self._update_criterion_args(loss)
            if abs(criterion) < tol and self._onlyonebatch is True:
                stop_condition = True
            if nb_epoch_done % 25 == 0:
                for name_param, param in self.model_parameters.items():
                    mse_param = 0 if param is None else mse(param).detach().item()
                    self._dict_mse[name_param].append(mse_param)
                if nb_epoch_done % 50 == 0 and verbose is True:
                    self._print_stats(nb_epoch_done, nb_max_epoch, tol)
            nb_epoch_done += 1
            pbar.update(1)

        self._print_end_of_fitting_message(stop_condition, tol)
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

    def _display_norm(self, ax=None, label=None, color=None):
        if ax is None:
            _, ax = plt.subplots(1)
        x = np.arange(0, len(self._dict_mse[list(self._dict_mse.keys())[0]]))
        for key, value in self._dict_mse.items():
            ax.plot(x, value, label=key)
        ax.legend()
        ax.set_title("Norm of each parameter.")

    def _handle_optimizer(self, lr, nb_max_epoch):
        if self.batch_size < self.n_samples:
            self.optim = torch.optim.Adam(
                self._list_of_parameters_needing_gradient, lr=lr
            )
            wrns = "No criterion for convergence is computed when batch_size < n_samples, the algorithm will stop "
            wrns += f"only after {nb_max_epoch} iterations (epochs). You can monitor the norm "
            wrns += " of each parameter calling .show() method after a fit. You can "
            wrns += " also save the model by calling .save() and fit it again after."
            warnings.warn(wrns)
        else:
            self.optim = torch.optim.Rprop(
                self._list_of_parameters_needing_gradient, lr=lr, step_sizes=(1e-10, 50)
            )

    def _get_batch(self, shuffle=False):
        """Get the batches required to do a  minibatch gradient ascent.

        Args:
            batch_size: int. The batch size. Should be lower than n.

        Returns: A generator. Will generate n//batch_size + 1 batches of
            size batch_size (except the last one since the rest of the
            division is not always 0)
        """
        indices = np.arange(self.n_samples)
        if shuffle:
            np.random.shuffle(indices)
        for i in range(self._nb_full_batch):
            batch = self._return_batch(
                indices, i * self._batch_size, (i + 1) * self._batch_size
            )
            yield batch

        # Last batch
        if self._last_batch_size != 0:
            yield self._return_batch(indices, -self._last_batch_size, self.n_samples)

    def _return_batch(self, indices, beginning, end):
        self.to_take = self._smart_device(torch.tensor(indices[beginning:end]))
        if self._exog is not None:
            exog_b = torch.index_select(self._exog, 0, self.to_take)
        else:
            exog_b = None
        return (
            torch.index_select(self._endog, 0, self.to_take),
            exog_b,
            torch.index_select(self._offsets, 0, self.to_take),
            torch.index_select(self._latent_mean, 0, self.to_take),
            torch.index_select(self._latent_sqrt_var, 0, self.to_take),
        )

    @property
    def _nb_full_batch(self):
        return self.n_samples // self.batch_size

    @property
    def _last_batch_size(self):
        return self.n_samples % self.batch_size

    @property
    def nb_batches(self):
        return self._nb_full_batch + (self._last_batch_size > 0)

    def _trainstep(self):
        """
        Perform a single pass of the data.

        Returns
        -------
        torch.Tensor
            The loss value.
        """
        elbo = 0
        t = time.time()
        for batch in self._get_batch(shuffle=False):
            self._extract_batch(batch)
            self.optim.zero_grad()
            loss = -self._compute_elbo_b()
            if torch.sum(torch.isnan(loss)):
                print("There are nan in the loss:", loss)
                raise ValueError("The ELBO contains nan values.")
            loss.backward()
            self.optim.step()
            elbo += loss.item()
        self._update_closed_forms()
        self._project_parameters()
        return elbo / self.nb_batches

    def _update_closed_forms(self):
        pass

    def _extract_batch(self, batch):
        self._endog_b = batch[0]
        self._exog_b = batch[1]
        self._offsets_b = batch[2]
        self._latent_mean_b = batch[3]
        self._latent_sqrt_var_b = batch[4]

    def transform(self):
        """
        Method for transforming the endog. Can be seen as a normalization of the endog.
        """
        return self.latent_variables

    def _qq_plots(self):
        centered_latent = self.latent_variables - torch.mean(
            self.latent_variables, axis=0
        )
        chol = torch.linalg.cholesky(torch.inverse(self.covariance_a_posteriori))
        residus = torch.matmul(centered_latent.unsqueeze(1), chol.unsqueeze(0))
        stats.probplot(residus.ravel(), plot=plt)
        plt.show()

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
        """
        pca = self.sk_PCA(n_components=n_components)
        return pca.transform(self.latent_variables)

    def sk_PCA(self, n_components=None):
        """
        Perform the scikit-learn PCA on the latent variables.

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
        latent_variables = self.transform()
        pca = PCA(n_components=n_components)
        pca.fit(latent_variables)
        return pca

    @property
    def latent_variance(self) -> torch.Tensor:
        """
        Property representing the latent variance.

        Returns
        -------
        torch.Tensor
            The latent variance tensor.
        """
        return (self.latent_sqrt_var**2).detach()

    @property
    def mean_gaussian(self):
        """Unconditional mean of the latent variable Z."""
        mean_gaussian = 0 if self.exog is None else self.exog @ self.coef
        if isinstance(mean_gaussian, int):
            return mean_gaussian
        return mean_gaussian.detach().cpu()

    def pca_pairplot(self, n_components=None, colors=None):
        """
        Generates a scatter matrix plot based on Principal Component Analysis (PCA) on the latent variables.

        Parameters
        ----------
            n_components (int, optional): The number of components to consider for plotting.
                If not specified, the maximum number of components will be used. Note that
                it will not display more than 10 graphs.
                Defaults to None.

            colors (np.ndarray): An array with one label for each
                sample in the endog property of the object.
                Defaults to None.
        Raises
        ------
            ValueError: If the number of components requested is greater than the number of variables in the dataset.
        """
        n_components = self._threshold_n_components(n_components)
        array = self.transform().numpy()
        _pca_pairplot(array, n_components, self.dim, colors)

    def _threshold_n_components(self, n_components):
        if n_components is None:
            n_components = self._get_max_components()

        if n_components > self.dim:
            raise ValueError(
                f"You ask more components ({n_components}) than variables ({self.dim})"
            )
        if n_components > 10:
            msg = f"Can not display a scatter matrix with {n_components}*"
            msg += f"{n_components} = {n_components*n_components} graphs."
            msg += f" Setting the number of components to 10."
            warnings.warn(msg)
            n_components = 10
        return n_components

    def plot_pca_correlation_circle(
        self, variables_names, indices_of_variables=None, title: str = ""
    ):
        """
        Visualizes variables using PCA and plots a correlation circle.

        Parameters
        ----------
            variables_names : List[str]
                A list of variable names to visualize.
            indices_of_variables : Optional[List[int]], optional
                A list of indices corresponding to the variables.
                If None, indices are determined based on `column_endog`, by default None
            title : str
                An additional title for the plot.

        Raises
        ------
            ValueError
                If `indices_of_variables` is None and `column_endog` is not set.
            ValueError
                If the length of `indices_of_variables` is different from the length of `variables_names`.

        Returns
        -------
            None
        """
        if indices_of_variables is None:
            if self.column_endog is None:
                raise ValueError(
                    "No names have been given to the column of "
                    "endog. Please set the column_endog to the"
                    "needed names or instantiate a new model with"
                    "a pd.DataFrame with appropriate column names"
                )
            indices_of_variables = []
            for variables_name in variables_names:
                index = self.column_endog.get_loc(variables_name)
                indices_of_variables.append(index)
        else:
            if len(indices_of_variables) != len(variables_names):
                raise ValueError(
                    f"Number of variables {len(indices_of_variables)} should be the same as the number of variables_names {len(variables_names)}"
                )
        plot_correlation_circle(
            self.transform(), variables_names, indices_of_variables, title=title
        )

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

    @property
    def latent_var(self) -> torch.Tensor:
        """
        Property representing the latent variance.

        Returns
        -------
        torch.Tensor
            The latent variance tensor.
        """
        return self.latent_sqrt_var**2

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
                f"in {self._criterion_args.iteration_number} iterations"
            )
        else:
            print(
                "Maximum number of iterations reached : ",
                self._criterion_args.iteration_number,
                ". Last criterion = ",
                np.round(self._criterion_args.criterion_list[-1], 8),
                f". Required tolerance = {tol}",
            )

    def _print_stats(self, nb_epoch_done, nb_max_epoch, tol):
        """
        Print the training statistics.
        """
        print("-------UPDATE-------")
        print(
            "Iteration ",
            nb_epoch_done,
            "out of ",
            nb_max_epoch,
            "iterations.",
        )
        msg_criterion = "Current criterion: " + str(
            np.round(self._criterion_args.criterion_list[-1], 6)
        )
        if self._onlyonebatch is True:
            msg_criterion += ". Stop if lower than " + str(tol)
        print(msg_criterion)
        print("ELBO:", np.round(self._criterion_args._elbos_list[-1], 6))

    def _update_criterion_args(self, loss):
        """
        Compute the convergence criterion and update the plot arguments.

        Parameters
        ----------
        loss : torch.Tensor
            The loss value.

        Returns
        -------
        float
            The computed criterion.
        """
        current_running_time = time.time() - self._beginning_time
        self._criterion_args.update_criterion(-loss, current_running_time)
        return self._criterion_args.criterion

    def display_covariance(self, ax=None, savefig=False, name_file="", display=True):
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
            sigma = self.covariance[:400, :400]
            sns.heatmap(self.covariance[:400, :400], ax=ax)
        else:
            sns.heatmap(self.covariance, ax=ax)
        ax.set_title("Covariance Matrix")
        plt.legend()
        if savefig:
            plt.savefig(name_file + self._NAME)
        if display is True:
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

    def show(self, axes=None, display=True):
        """
        Show 3 plots. The first one is the covariance of the model.
        The second one is the stopping criterion with the runtime in abscisse.
        The third one is the elbo.

        Parameters
        ----------
        axes : numpy.ndarray, optional
            The axes to plot on. If None, a new figure will be created. Defaults to None.
        display : bool, optional.
            If True, will display the graph.
        """
        print("Likelihood:", self.loglike)
        if self._fitted is False:
            nb_axes = 1
        else:
            nb_axes = 3
        if axes is None:
            _, axes = plt.subplots(1, nb_axes, figsize=(23, 5))
        if self._fitted is True:
            self._criterion_args._show_loss(ax=axes[2])
            self._display_norm(ax=axes[1])
            self.display_covariance(ax=axes[0], display=False)
        else:
            self.display_covariance(ax=axes)

        if display is True:
            plt.show()

    @property
    def _elbos_list(self):
        """
        Property representing the list of ELBO values.
        """
        return self._criterion_args._elbos_list

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
            self._criterion_args._elbos_list.append(self.compute_elbo().item())
            self._criterion_args.running_times.append(time.time() - t0)
        return self.n_samples * self._elbos_list[-1]

    def compute_loglike(self):
        """
        Computes the log likelihood.
        """
        return self.n_samples * self.compute_elbo()

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
    def dict_data(self):
        """
        Property representing the data dictionary.

        Returns
        -------
        dict
            The dictionary of data.
        """
        return {
            "endog": self.endog,
            "exog": self.exog,
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
            The latent mean or None if it has not yet been initialized.
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
    def latent_mean(self, latent_mean: Union[torch.Tensor, np.ndarray, pd.DataFrame]):
        """
        Setter for the latent mean property.

        Parameters
        ----------
        latent_mean : Union[torch.Tensor, np.ndarray, pd.DataFrame]
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
        self._latent_mean = self._smart_device(latent_mean)

    def _smart_device(self, tensor):
        if tensor is None:
            return None
        if self._onlyonebatch is True:
            return tensor.to(DEVICE)
        return tensor

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
    def endog(self):
        """
        Property representing the endog.

        Returns
        -------
        torch.Tensor or None
            The endog or None.
        """
        return self._cpu_attribute_or_none("_endog")

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
    def exog(self):
        """
        Property representing the exog.

        Returns
        -------
        torch.Tensor or None
            The exog or None.
        """
        return self._cpu_attribute_or_none("_exog")

    @property
    def _exog_b_device(self):
        if self._exog is None:
            return None
        return self._exog_b.to(DEVICE)

    @endog.setter
    @_array2tensor
    def endog(self, endog: Union[torch.Tensor, np.ndarray, pd.DataFrame]):
        """
        Setter for the endog property.

        Parameters
        ----------
        endog : Union[torch.Tensor, np.ndarray, pd.DataFrame]
            The endog.

        Raises
        ------
        ValueError
            If the shape of the endog is incorrect or if the input is negative.
        """
        if self.endog.shape != endog.shape:
            raise ValueError(
                f"Wrong shape for the endog. Expected {self.endog.shape}, got {endog.shape}"
            )
        if torch.min(endog) < 0:
            raise ValueError("Input should be non-negative only.")
        self._endog = endog

    @offsets.setter
    @_array2tensor
    def offsets(self, offsets: Union[torch.Tensor, np.ndarray, pd.DataFrame]):
        """
        Setter for the offsets property.

        Parameters
        ----------
        offsets : Union[torch.Tensor, np.ndarray, pd.DataFrame]
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

    @exog.setter
    @_array2tensor
    def exog(self, exog: Union[torch.Tensor, np.ndarray, pd.DataFrame]):
        """
        Setter for the exog property.

        Parameters
        ----------
        exog : Union[torch.Tensor, np.ndarray, pd.DataFrame]
            The exog.

        Raises
        ------
        ValueError
            If the shape of the exog or endog is incorrect.
        """
        _check_data_shape(self.endog, exog, self.offsets)
        self._exog = exog

    @coef.setter
    @_array2tensor
    def coef(self, coef: Union[torch.Tensor, np.ndarray, pd.DataFrame]):
        """
        Setter for the coef property.

        Parameters
        ----------
        coef : Union[torch.Tensor, np.ndarray, pd.DataFrame]
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
        self._coef = self._smart_device(coef)

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
        return ".show(), .transform(), .sigma(), .predict(), .pca_projected_latent_variables(), .plot_pca_correlation_circle(), .viz(), .pca_pairplot(), .plot_expected_vs_true()"

    def sigma(self):
        """
        Method returning the covariance matrix.

        Returns
        -------
        torch.Tensor or None
            The covariance matrix or None.
        """
        return self.covariance

    def predict(self, exog: Union[torch.Tensor, np.ndarray, pd.DataFrame] = None):
        """
        Method for making predictions.

        Parameters
        ----------
        exog : Union[torch.Tensor, np.ndarray, pd.DataFrame], optional
            The exog, by default None.

        Returns
        -------
        torch.Tensor or None
            The predicted values or None.

        Raises
        ------
        AttributeError
            If there are no exog in the model but some are provided.
        RuntimeError
            If the shape of the exog is incorrect.

        Notes
        -----
        - If `exog` is not provided and there are no exog in the model, None is returned.
            If there are exog in the model, then the mean exog @ coef is returned.
        - If `exog` is provided, it should have the shape `(_, nb_cov)`, where `nb_cov` is the number of exog.
        - The predicted values are obtained by multiplying the exog by the coefficients.
        """
        exog = _format_data(exog)
        if exog is not None and self.nb_cov == 0:
            msg = "No exog in the model, can't predict with exog"
            raise AttributeError(msg)
        if exog is None:
            if self.exog is None:
                print("No exog in the model.")
                return None
            return self.exog @ self.coef
        if exog.shape[-1] != self.nb_cov:
            error_string = f"X has wrong shape ({exog.shape}). Should"
            error_string += f" be ({self.n_samples, self.nb_cov})."
            raise RuntimeError(error_string)
        return exog @ self.coef

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
    def reconstruction_error(self):
        """Recontstruction error between the predictions and the endog."""
        predictions = self._endog_predictions().ravel().detach()
        return torch.sqrt(torch.mean((predictions - self._endog.ravel().cpu()) ** 2))

    @property
    def elbo(self):
        """Alias for loglike"""
        return self.loglike

    def plot_expected_vs_true(self, ax=None, colors=None):
        """
        Plot the predicted value of the endog against the endog.

        Parameters
        ----------
        ax : Optional[matplotlib.axes.Axes], optional
            The matplotlib axis to use. If None, the current axis is used, by default None.

        colors : Optional[Any], optional
            The colors to use for plotting, by default None.

        Returns
        -------
        matplotlib.axes.Axes
            The matplotlib axis.
        """
        if self._fitted is None:
            raise RuntimeError("Please fit the model before.")
        if ax is None:
            ax = plt.gca()
            to_show = True
        else:
            to_show = False
        predictions = self._endog_predictions().ravel().detach()
        if colors is not None:
            colors = np.repeat(np.array(colors), repeats=self.dim).ravel()
        sns.scatterplot(x=self.endog.ravel(), y=predictions, hue=colors, ax=ax)
        max_y = int(torch.max(self.endog.ravel()).item())
        y = np.linspace(0, max_y, max_y)
        ax.plot(y, y, c="red")
        ax.set_yscale("log")
        ax.set_title(
            f"Reconstruction error (RMSE):{np.round(self.reconstruction_error.item(), 3)}"
        )
        ax.set_xscale("log")
        ax.set_ylabel("Predicted values")
        ax.set_xlabel("Counts")
        ax.set_ylim(top=max_y, bottom=0.001)
        ax.legend()
        if to_show is True:
            plt.show()
        return ax

    def _print_beginning_message(self):
        """
        Method for printing the beginning message.
        """
        print(f"Fitting a {self._NAME} model with {self._description}")

    @property
    @abstractmethod
    def latent_variables(self) -> torch.Tensor:
        """
        Property representing the latent variables.

        Returns
        -------
        torch.Tensor
            The latent variables of size (n_samples, dim).
        """

    @abstractmethod
    def compute_elbo(self):
        """
        Compute the Evidence Lower BOund (ELBO) that will be maximized
        by pytorch.

        Returns
        -------
        torch.Tensor
            The computed ELBO.
        """

    @abstractmethod
    def _compute_elbo_b(self) -> torch.Tensor:
        """
        Compute the Evidence Lower BOund (ELBO) for the current mini-batch.
        Returns
        -------
        torch.Tensor
            The computed ELBO on the current batch.
        """

    @abstractmethod
    def _random_init_model_parameters(self):
        """
        Abstract method to randomly initialize model parameters.
        """

    @abstractmethod
    def _random_init_latent_parameters(self):
        """
        Abstract method to randomly initialize latent parameters.
        """

    @abstractmethod
    def _smart_init_latent_parameters(self):
        """
        Method for smartly initializing the latent parameters.
        """

    @abstractmethod
    def _smart_init_model_parameters(self):
        """
        Method for smartly initializing the model parameters.
        """

    @property
    @abstractmethod
    def _list_of_parameters_needing_gradient(self):
        """
        A list containing all the parameters that need to be upgraded via a gradient step.

        Returns
        -------
        List[torch.Tensor]
            List of parameters needing gradient.
        """

    @property
    @abstractmethod
    def _description(self):
        pass

    @property
    @abstractmethod
    def number_of_parameters(self) -> int:
        """
        Number of parameters of the model.
        """

    @property
    @abstractmethod
    def model_parameters(self) -> Dict[str, torch.Tensor]:
        """
        Property representing the model parameters.

        Returns
        -------
        dict
            The dictionary of model parameters.
        """

    @property
    @abstractmethod
    def latent_parameters(self) -> Dict[str, torch.Tensor]:
        """
        Property representing the latent parameters.

        Returns
        -------
        dict
            The dictionary of latent parameters.
        """


class Pln(_model):
    """
    Pln class.

    Examples
    --------
    >>> from pyPLNmodels import Pln, load_scrna
    >>> endog, labels = load_scrna(return_labels = True, for_formula = False)
    >>> pln = Pln(endog,add_const = True)
    >>> pln.fit()
    >>> print(pln)
    >>> pln.viz(colors = labels)

    >>> from pyPLNmodels import Pln, get_simulation_parameters, sample_pln
    >>> param = get_simulation_parameters()
    >>> endog = sample_pln(param)
    >>> data = {"endog": endog}
    >>> pln = Pln.from_formula("endog ~ 1", data)
    >>> pln.fit()
    >>> print(pln)
    """

    _NAME = "Pln"
    coef: torch.Tensor

    @_add_doc(
        _model,
        example="""
            >>> from pyPLNmodels import Pln, load_scrna
            >>> data = load_scrna()
            >>> pln = Pln.from_formula("endog ~ 1", data)
            >>> pln.fit()
            >>> print(pln)
        """,
        returns="""
            Pln
        """,
        see_also="""
        :func:`pyPLNmodels.Pln.from_formula`
        """,
    )
    def __init__(
        self,
        endog: Optional[Union[torch.Tensor, np.ndarray, pd.DataFrame]],
        *,
        exog: Optional[Union[torch.Tensor, np.ndarray, pd.DataFrame]] = None,
        offsets: Optional[Union[torch.Tensor, np.ndarray, pd.DataFrame]] = None,
        offsets_formula: {"zero", "logsum"} = "zero",
        dict_initialization: Optional[Dict[str, torch.Tensor]] = None,
        take_log_offsets: bool = False,
        add_const: bool = True,
        batch_size: int = None,
    ):
        super().__init__(
            endog=endog,
            exog=exog,
            offsets=offsets,
            offsets_formula=offsets_formula,
            dict_initialization=dict_initialization,
            take_log_offsets=take_log_offsets,
            add_const=add_const,
            batch_size=batch_size,
        )

    @classmethod
    @_add_doc(
        _model,
        example="""
            >>> from pyPLNmodels import Pln, load_scrna
            >>> data = load_scrna()
            >>> pln = Pln.from_formula("endog ~ 1", data = data)
        """,
        returns="""
            Pln
        """,
        see_also="""
        :class:`pyPLNmodels.Pln`
        :func:`pyPLNmodels.Pln.__init__`
    """,
    )
    def from_formula(
        cls,
        formula: str,
        data: Dict[str, Union[torch.Tensor, np.ndarray, pd.DataFrame]],
        *,
        offsets_formula: {"zero", "logsum"} = "zero",
        dict_initialization: Optional[Dict[str, torch.Tensor]] = None,
        take_log_offsets: bool = False,
        batch_size: int = None,
    ):
        return super().from_formula(
            formula=formula,
            data=data,
            offsets_formula=offsets_formula,
            dict_initialization=dict_initialization,
            take_log_offsets=take_log_offsets,
            batch_size=batch_size,
        )

    @_add_doc(
        _model,
        example="""
        >>> from pyPLNmodels import Pln, load_scrna
        >>> data = load_scrna()
        >>> pln = Pln.from_formula("endog ~ 1",data = data)
        >>> pln.fit()
        >>> print(pln)
        """,
    )
    def fit(
        self,
        nb_max_epoch: int = 400,
        *,
        lr: float = 0.01,
        tol: float = 1e-5,
        do_smart_init: bool = True,
        verbose: bool = False,
    ):
        super().fit(
            nb_max_epoch,
            lr=lr,
            tol=tol,
            do_smart_init=do_smart_init,
            verbose=verbose,
        )

    @_add_doc(
        _model,
        example="""
            >>> import matplotlib.pyplot as plt
            >>> from pyPLNmodels import Pln, load_scrna
            >>> endog, labels = load_scrna(return_labels = True, for_formula = False)
            >>> pln = Pln(endog,add_const = True)
            >>> pln.fit()
            >>> pln.plot_expected_vs_true()
            >>> plt.show()
            >>> pln.plot_expected_vs_true(colors = labels)
            >>> plt.show()
            """,
    )
    def plot_expected_vs_true(self, ax=None, colors=None):
        super().plot_expected_vs_true(ax=ax, colors=colors)

    @_add_doc(
        _model,
        example="""
            >>> import matplotlib.pyplot as plt
            >>> from pyPLNmodels import Pln, load_scrna
            >>> data = load_scrna(return_labels = True)
            >>> pln = Pln.from_formula("endog ~ 1", data = data)
            >>> pln.fit()
            >>> pln.viz()
            >>> plt.show()
            >>> pln.viz(colors = data["labels"])
            >>> plt.show()
            >>> pln.viz(show_cov = True)
            >>> plt.show()
            """,
    )
    def viz(self, ax=None, colors=None, show_cov: bool = False):
        super().viz(ax=ax, colors=colors, show_cov=show_cov)

    @_add_doc(
        _model,
        example="""
        >>> from pyPLNmodels import Pln, load_scrna
        >>> data = load_scrna()
        >>> pln = Pln.from_formula("endog ~ 1", data = data)
        >>> pln.fit()
        >>> pca_proj = pln.pca_projected_latent_variables()
        >>> print(pca_proj.shape)
        """,
    )
    def pca_projected_latent_variables(self, n_components: Optional[int] = None):
        return super().pca_projected_latent_variables(n_components=n_components)

    @_add_doc(
        _model,
        example="""
        >>> from pyPLNmodels import Pln, load_scrna
        >>> data = load_scrna()
        >>> pln = Pln.from_formula("endog ~ 1", data = data)
        >>> pln.fit()
        >>> pln.pca_pairplot(n_components = 5)
        """,
    )
    def pca_pairplot(self, n_components=None, colors=None):
        super().pca_pairplot(n_components=n_components, colors=colors)

    @_add_doc(
        _model,
        example="""
        >>> from pyPLNmodels import Pln, load_scrna
        >>> data = load_scrna()
        >>> pln = Pln.from_formula("endog ~ 1", data = data)
        >>> pln.fit()
        >>> pln.plot_pca_correlation_circle(["a","b"], indices_of_variables = [4,8])
        """,
    )
    def plot_pca_correlation_circle(self, variables_names, indices_of_variables=None):
        super().plot_pca_correlation_circle(
            variables_names=variables_names, indices_of_variables=indices_of_variables
        )

    @_add_doc(
        _model,
        returns="""
        torch.Tensor
            The transformed endog (latent variables of the model).
        """,
        example="""
              >>> from pyPLNmodels import Pln, load_scrna
              >>> data = load_scrna()
              >>> pln = Pln.from_formula("endog ~ 1", data = data)
              >>> pln.fit()
              >>> transformed_endog = pln.transform()
              >>> print(transformed_endog.shape)
              """,
    )
    def transform(self):
        return super().transform()

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
        if hasattr(self, "_latent_mean") and hasattr(self, "_exog") and self.nb_cov > 0:
            return self._coef.detach().cpu()
        return None

    @coef.setter
    def coef(self, coef: Union[torch.Tensor, np.ndarray, pd.DataFrame]):
        """
        Setter for the coef property.

        Parameters
        ----------
        coef : Union[torch.Tensor, np.ndarray, pd.DataFrame]
            The regression coefficients of the gaussian latent variables.
        Raises
        ------
        AttributeError since you can not set the coef in the Pln model.
        """
        msg = "You can not set the coef in the Pln model."
        warnings.warn(msg)

    def _endog_predictions(self):
        return torch.exp(
            self.offsets + self.latent_mean + 1 / 2 * self.latent_sqrt_var**2
        )

    def _get_max_components(self):
        """
        Method for getting the maximum number of components.

        Returns
        -------
        int
            The maximum number of components.
        """
        return min(self.dim, self.n_samples)

    @property
    def _coef(self):
        """
        Property representing the coefficients.

        Returns
        -------
        torch.Tensor
            The coefficients.
        """
        return _closed_formula_coef(self._exog, self._latent_mean)

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
            self._exog,
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

    @_model.latent_sqrt_var.setter
    @_array2tensor
    def latent_sqrt_var(
        self, latent_sqrt_var: Union[torch.Tensor, np.ndarray, pd.DataFrame]
    ):
        """
        Setter for the latent variance property.

        Parameters
        ----------
        latent_sqrt_var : Union[torch.Tensor, np.ndarray, pd.DataFrame]
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
        self._latent_sqrt_var = self._smart_device(latent_sqrt_var)

    @property
    def number_of_parameters(self) -> int:
        """
        Property representing the number of parameters.

        Returns
        -------
        int
            The number of parameters.
        """
        return self.dim * (self.dim + 2 * self.nb_cov + 1) / 2

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
                "_exog",
                "_latent_mean",
                "_latent_sqrt_var",
                "_coef",
                "n_samples",
            ]
        ):
            return self._covariance.detach().cpu()
        return None

    @covariance.setter
    def covariance(self, covariance):
        """
        Setter for the covariance property. Only here for completeness, since
        this function does nothing

        Parameters
        ----------
        covariance : torch.Tensor
            The covariance matrix.
        """
        warnings.warn("You can not set the covariance for the Pln model.")

    def _random_init_latent_sqrt_var(self):
        if not hasattr(self, "_latent_sqrt_var"):
            self._latent_sqrt_var = (
                1 / 2 * self._smart_device(torch.ones((self.n_samples, self.dim)))
            )

    @_add_doc(_model)
    def _smart_init_model_parameters(self):
        pass
        # no model parameters since we are doing a profiled ELBO

    @_add_doc(_model)
    def _random_init_model_parameters(self):
        pass
        # no model parameters since we are doing a profiled ELBO

    @property
    @_add_doc(_model)
    def _list_of_parameters_needing_gradient(self):
        return [self._latent_mean, self._latent_sqrt_var]

    @property
    @_add_doc(_model)
    def model_parameters(self) -> Dict[str, torch.Tensor]:
        return {"coef": self.coef, "covariance": self.covariance}

    @property
    @_add_doc(_model)
    def latent_parameters(self):
        return {
            "latent_sqrt_var": self.latent_sqrt_var,
            "latent_mean": self.latent_mean,
        }

    def _random_init_latent_sqrt_var(self):
        if not hasattr(self, "_latent_sqrt_var"):
            self._latent_sqrt_var = (
                1 / 2 * self._smart_device(torch.ones((self.n_samples, self.dim)))
            )

    @_add_doc(_model)
    def _smart_init_latent_parameters(self):
        self._random_init_latent_sqrt_var()
        if not hasattr(self, "_latent_mean"):
            # we should do a poisson regression and initialise M with it
            self._latent_mean = self._smart_device(
                torch.log(self._endog + (self._endog == 0))
            )

    @property
    @_add_doc(
        _model,
        example="""
        >>> from pyPLNmodels import Pln, load_scrna
        >>> data = load_scrna()
        >>> pln = Pln.from_formula(" endog ~ 1", data)
        >>> pln.fit()
        >>> print(pln.latent_variables.shape)
        """,
    )
    def latent_variables(self):
        return self.latent_mean.detach()

    @_add_doc(
        _model,
        example="""
            >>> from pyPLNmodels import Pln, load_scrna
            >>> data = load_scrna()
            >>> pln = Pln.from_formula("endog ~ 1", data)
            >>> pln.fit()
            >>> elbo = pln.compute_elbo()
            >>> print("elbo: ", elbo)
            >>> print("loglike/n: ", pln.loglike/pln.n_samples)
            """,
    )
    def compute_elbo(self):
        return profiled_elbo_pln(
            self._endog,
            self._exog,
            self._offsets,
            self._latent_mean,
            self._latent_sqrt_var,
        )

    @_add_doc(_model)
    def _compute_elbo_b(self) -> torch.Tensor:
        elbo = profiled_elbo_pln(
            self._endog_b.to(DEVICE),
            self._exog_b_device,
            self._offsets_b.to(DEVICE),
            self._latent_mean_b.to(DEVICE),
            self._latent_sqrt_var_b.to(DEVICE),
        )
        return elbo

    def _smart_init_model_parameters(self):
        pass
        # no model parameters since we are doing a profiled ELBO

    @_add_doc(_model)
    def _random_init_model_parameters(self):
        pass
        # no model parameters since we are doing a profiled ELBO

    @_add_doc(_model)
    def _random_init_latent_parameters(self):
        self._random_init_latent_sqrt_var()
        if not hasattr(self, "_latent_mean"):
            self._latent_mean = self._smart_device(
                torch.ones((self.n_samples, self.dim))
            )

    @property
    @_add_doc(_model)
    def _list_of_parameters_needing_gradient(self):
        return [self._latent_mean, self._latent_sqrt_var]


class PlnPCAcollection:
    """
    A collection where value q corresponds to a PlnPCA object with rank q.

    Examples
    --------
    >>> from pyPLNmodels import PlnPCAcollection, load_scrna, get_simulation_parameters, sample_pln
    >>> data = load_scrna()
    >>> plnpcas = PlnPCAcollection.from_formula("endog ~ 1", data = data, ranks = [5,8, 12])
    >>> plnpcas.fit()
    >>> print(plnpcas)
    >>> plnpcas.show()
    >>> print(plnpcas.best_model())
    >>> print(plnpcas[5])

    >>> plnparam = get_simulation_parameters(n_samples =100, dim = 60, nb_cov = 2, rank = 8)
    >>> endog = sample_pln(plnparam)
    >>> data = {"endog":endog, "cov": plnparam.exog, "offsets": plnparam.offsets}
    >>> plnpcas = PlnPCAcollection.from_formula("endog ~ 0 + cov", data = data, ranks = [5,8,12])
    >>> plnpcas.fit()
    >>> print(plnpcas)
    >>> plnpcas.show()
    See also
    --------
    :class:`~pyPLNmodels.PlnPCA`
    """

    _NAME = "PlnPCAcollection"
    _dict_models: dict

    def __init__(
        self,
        endog: Union[torch.Tensor, np.ndarray, pd.DataFrame],
        *,
        exog: Union[torch.Tensor, np.ndarray, pd.DataFrame] = None,
        offsets: Union[torch.Tensor, np.ndarray, pd.DataFrame] = None,
        offsets_formula: {"zero", "logsum"} = "zero",
        ranks: Iterable[int] = range(3, 5),
        dict_of_dict_initialization: Optional[dict] = None,
        take_log_offsets: bool = False,
        add_const: bool = True,
        batch_size: int = None,
    ):
        """
        Constructor for PlnPCAcollection.

        Parameters
        ----------
        endog :Union[torch.Tensor, np.ndarray, pd.DataFrame]
            The endog.
        exog : Union[torch.Tensor, np.ndarray, pd.DataFrame], optional(keyword-only)
            The exog, by default None.
        offsets : Union[torch.Tensor, np.ndarray, pd.DataFrame], optional(keyword-only)
            The offsets, by default None.
        offsets_formula : str, optional(keyword-only)
            The formula for offsets, by default "zero". Can be also "logsum" where we take the logarithm of the sum (of each line) of the counts.
            Overriden (useless) if offsets is not None.
        ranks : Iterable[int], optional(keyword-only)
            The range of ranks, by default range(3, 5).
        dict_of_dict_initialization : dict, optional(keyword-only)
            The dictionary of initialization, by default None.
        take_log_offsets : bool, optional(keyword-only)
            Whether to take the logarithm of offsets, by default False.
        add_const: bool, optional(keyword-only)
            Whether to add a column of one in the exog. Defaults to True.
        batch_size: int, optional(keyword-only)
            The batch size when optimizing the elbo. If None,
            batch gradient descent will be performed (i.e. batch_size = n_samples).
        Returns
        -------
        PlnPCAcollection
        See also
        --------
        :class:`~pyPLNmodels.PlnPCA`
        :meth:`~pyPLNmodels.PlnPCAcollection.from_formula`
        """
        self._dict_models = {}
        (
            self._endog,
            self._exog,
            self._offsets,
            self.column_endog,
            _,
            _,
            self._batch_size,
        ) = _handle_data(
            endog,
            exog,
            offsets,
            offsets_formula,
            take_log_offsets,
            add_const,
            batch_size,
        )
        self._fitted = False
        self._init_models(ranks, dict_of_dict_initialization, add_const=add_const)

    @classmethod
    def from_formula(
        cls,
        formula: str,
        data: Dict[str, Union[torch.Tensor, np.ndarray, pd.DataFrame]],
        *,
        offsets_formula: {"zero", "logsum"} = "zero",
        ranks: Iterable[int] = range(3, 5),
        dict_of_dict_initialization: Optional[dict] = None,
        take_log_offsets: bool = False,
        batch_size: int = None,
    ) -> "PlnPCAcollection":
        """
        Create an instance of PlnPCAcollection from a formula.

        Parameters
        ----------
        formula : str
            The formula.
        data : dict
            The data dictionary. Each value can be either
            a torch.Tensor, np.ndarray or pd.DataFrame
        offsets_formula : str, optional(keyword-only)
            The formula for offsets, by default "zero". Can be also "logsum" where we take the logarithm of the sum (of each line) of the counts.
            Overriden (useless) if data["offsets"] is not None.
        ranks : Iterable[int], optional(keyword-only)
            The range of ranks, by default range(3, 5).
        dict_of_dict_initialization : dict, optional(keyword-only)
            The dictionary of initialization, by default None.
        take_log_offsets : bool, optional(keyword-only)
            Whether to take the logarithm of offsets, by default False.
        batch_size: int, optional(keyword-only)
            The batch size when optimizing the elbo. If None,
            batch gradient descent will be performed (i.e. batch_size = n_samples).

        Returns
        -------
        PlnPCAcollection
            The created PlnPCAcollection instance.
        Examples
        --------
        >>> from pyPLNmodels import PlnPCAcollection, load_scrna
        >>> data = load_scrna()
        >>> pca_col = PlnPCAcollection.from_formula("endog ~ 1", data = data, ranks = [5,6])
        See also
        --------
        :class:`~pyPLNmodels.PlnPCA`
        :func:`~pyPLNmodels.PlnPCAcollection.__init__`
        """
        endog, exog, offsets = _extract_data_from_formula_no_infla(formula, data)
        return cls(
            endog,
            exog=exog,
            offsets=offsets,
            offsets_formula=offsets_formula,
            ranks=ranks,
            dict_of_dict_initialization=dict_of_dict_initialization,
            take_log_offsets=take_log_offsets,
            add_const=False,
            batch_size=batch_size,
        )

    @property
    def exog(self) -> torch.Tensor:
        """
        Property representing the exog.

        Returns
        -------
        torch.Tensor
            The exog.
        """
        return self[self.ranks[0]].exog

    @property
    def batch_size(self) -> torch.Tensor:
        """
        Property representing the batch_size.

        Returns
        -------
        torch.Tensor
            The batch_size.
        """
        return self[self.ranks[0]].batch_size

    @property
    def endog(self) -> torch.Tensor:
        """
        Property representing the endog.

        Returns
        -------
        torch.Tensor
            The endog.
        """
        return self[self.ranks[0]].endog

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

    @endog.setter
    @_array2tensor
    def endog(self, endog: Union[torch.Tensor, np.ndarray, pd.DataFrame]):
        """
        Setter for the endog property.

        Parameters
        ----------
        endog : Union[torch.Tensor, np.ndarray, pd.DataFrame]
            The endog.
        """
        for model in self.values():
            model.endog = endog

    @batch_size.setter
    def batch_size(self, batch_size: int):
        """
        Does not allow to set the batch size.
        """
        raise ValueError("Can not set the batch size after initialization")

    @coef.setter
    @_array2tensor
    def coef(self, coef: Union[torch.Tensor, np.ndarray, pd.DataFrame]):
        """
        Setter for the coef property.

        Parameters
        ----------
        coef : Union[torch.Tensor, np.ndarray, pd.DataFrame]
            The coefficients.
        """
        for model in self.values():
            model.coef = coef

    @exog.setter
    @_array2tensor
    def exog(self, exog: Union[torch.Tensor, np.ndarray, pd.DataFrame]):
        """
        Setter for the exog property.

        Parameters
        ----------
        exog : Union[torch.Tensor, np.ndarray, pd.DataFrame]
            The exog.
        """
        for model in self.values():
            model.exog = exog

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
    def offsets(self, offsets: Union[torch.Tensor, np.ndarray, pd.DataFrame]):
        """
        Setter for the offsets property.

        Parameters
        ----------
        offsets : Union[torch.Tensor, np.ndarray, pd.DataFrame]
            The offsets.
        """
        for model in self.values():
            model.offsets = offsets

    def _init_models(
        self,
        ranks: Iterable[int],
        dict_of_dict_initialization: Optional[dict],
        add_const: bool,
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
                        endog=self._endog,
                        exog=self._exog,
                        offsets=self._offsets,
                        rank=rank,
                        dict_initialization=dict_initialization,
                        add_const=add_const,
                        batch_size=self._batch_size,
                    )
                else:
                    raise TypeError(
                        "Please instantiate with either a list "
                        "of integers or an integer."
                    )
            if dict_of_dict_initialization is not None:
                if ranks != dict_of_dict_initialization["ranks"]:
                    msg = (
                        "The given ranks in the dict_initialization are loaded but"
                        " you should fit the model once again or instantiate the"
                        " model with the ranks loaded."
                    )
                    warnings.warn(msg)

        elif isinstance(ranks, (int, np.integer)):
            dict_initialization = _get_dict_initialization(
                ranks, dict_of_dict_initialization
            )
            self._dict_models[rank] = PlnPCA(
                self._endog,
                self._exog,
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

    def _print_beginning_message(self) -> str:
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
        Property representing the number of exog.

        Returns
        -------
        int
            The number of exog.
        """
        return self[self.ranks[0]].nb_cov

    def fit(
        self,
        nb_max_epoch: int = 400,
        *,
        lr: float = 0.01,
        tol: float = 1e-5,
        do_smart_init: bool = True,
        verbose: bool = False,
    ):
        """
        Fit each model in the PlnPCAcollection.

        Parameters
        ----------
        nb_max_epoch : int, optional
            The maximum number of epochs (iterations), by default 400.
        lr : float, optional(keyword-only)
            The learning rate, by default 0.01.
        tol : float, optional(keyword-only)
            The tolerance, by default 1e-5.
        do_smart_init : bool, optional(keyword-only)
            Whether to do smart initialization, by default True.
        verbose : bool, optional(keyword-only)
            Whether to print verbose output, by default False.
        Raises
        ------
        """
        self._print_beginning_message()
        for i in range(len(self.values())):
            model = self[self.ranks[i]]
            model.fit(
                nb_max_epoch,
                lr=lr,
                tol=tol,
                do_smart_init=do_smart_init,
                verbose=verbose,
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
        next_model.coef = current_model._coef
        next_model.components = torch.zeros(self.dim, next_model.rank).to(DEVICE)
        with torch.no_grad():
            next_model._components[:, : current_model.rank] = (
                current_model._components
            ).to(DEVICE)

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
        colors = {"BIC": "blue", "AIC": "red", "Negative log likelihood": "orange"}
        for criterion, values in zip(
            ["BIC", "AIC", "Negative log likelihood"], [bic, aic, loglikes]
        ):
            plt.scatter(
                values.keys(),
                values.values(),
                label=f"{criterion} criterion",
                c=colors[criterion],
            )
            plt.plot(values.keys(), values.values(), c=colors[criterion])
            if criterion == "BIC":
                plt.axvline(
                    self.best_BIC_model_rank, c=colors[criterion], linestyle="dotted"
                )
            elif criterion == "AIC":
                plt.axvline(
                    self.best_AIC_model_rank, c=colors[criterion], linestyle="dotted"
                )
                plt.xticks(list(values.keys()))
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
        return ".show(), .best_model(), .keys(), .items(), .values()"

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


# Here, setting the value for each key  _dict_parameters
class PlnPCA(_model):
    """
    PlnPCA object where the covariance has low rank.

    Examples
    --------
    >>> from pyPLNmodels import PlnPCA, load_scrna, get_simulation_parameters, sample_pln
    >>> endog, labels = load_scrna(return_labels = True, for_formula = False)
    >>> data = {"endog": endog}
    >>> pca = PlnPCA.from_formula("endog ~ 1", data = data, rank = 5)
    >>> pca.fit()
    >>> print(pca)
    >>> pca.viz(colors = labels)

    >>> plnparam = get_simulation_parameters(n_samples =100, dim = 60, nb_cov = 2, rank = 8)
    >>> endog = sample_pln(plnparam)
    >>> data = {"endog": endog, "cov": plnparam.exog, "offsets": plnparam.offsets}
    >>> plnpca = PlnPCA.from_formula("endog ~ 0 + cov", data = data, rank = 5)
    >>> plnpca.fit()
    >>> print(plnpca)

    See also
    --------
    :class:`pyPLNmodels.Pln`
    """

    _NAME: str = "PlnPCA"
    _components: torch.Tensor

    @_add_doc(
        _model,
        params="""
            rank : int, optional(keyword-only)
                The rank of the approximation, by default 5.
            """,
        example="""
            >>> from pyPLNmodels import PlnPCA, load_scrna
            >>> data = load_scrna()
            >>> pca = PlnPCA.from_formula("endog ~ 1", data)
            >>> pca.fit()
            >>> print(pca)
        """,
        returns="""
            PlnPCA
        """,
        see_also="""
        :func:`pyPLNmodels.PlnPCA.from_formula`
        """,
    )
    def __init__(
        self,
        endog: Union[torch.Tensor, np.ndarray, pd.DataFrame],
        *,
        exog: Optional[Union[torch.Tensor, np.ndarray, pd.DataFrame]] = None,
        offsets: Optional[Union[torch.Tensor, np.ndarray, pd.DataFrame]] = None,
        offsets_formula: {"zero", "logsum"} = "zero",
        rank: int = 5,
        dict_initialization: Optional[Dict[str, torch.Tensor]] = None,
        take_log_offsets: bool = False,
        add_const: bool = True,
        batch_size: int = None,
    ):
        self._rank = rank
        super().__init__(
            endog=endog,
            exog=exog,
            offsets=offsets,
            offsets_formula=offsets_formula,
            dict_initialization=dict_initialization,
            take_log_offsets=take_log_offsets,
            add_const=add_const,
            batch_size=batch_size,
        )

    @classmethod
    @_add_doc(
        _model,
        params="""
            rank : int, optional(keyword-only)
                The rank of the approximation, by default 5.
            """,
        example="""
            >>> from pyPLNmodels import PlnPCA, load_scrna
            >>> data = load_scrna()
            >>> pca = PlnPCA.from_formula("endog ~ 1", data = data, rank = 5)
        """,
        returns="""
            PlnPCA
        """,
        see_also="""
        :class:`pyPLNmodels.Pln`
        :func:`pyPLNmodels.PlnPCA.__init__`
    """,
    )
    def from_formula(
        cls,
        formula: str,
        data: Dict[str, Union[torch.Tensor, np.ndarray, pd.DataFrame]],
        *,
        rank: int = 5,
        offsets_formula: {"zero", "logsum"} = "zero",
        dict_initialization: Optional[Dict[str, torch.Tensor]] = None,
        batch_size: int = None,
    ):
        endog, exog, offsets = _extract_data_from_formula_no_infla(formula, data)
        return cls(
            endog,
            exog=exog,
            offsets=offsets,
            offsets_formula=offsets_formula,
            rank=rank,
            dict_initialization=dict_initialization,
            add_const=False,
            batch_size=batch_size,
        )

    @_add_doc(
        _model,
        example="""
        >>> from pyPLNmodels import PlnPCA, load_scrna
        >>> endog = load_scrna(for_formula = False)
        >>> plnpca = PlnPCA(endog,add_const = True, rank = 6)
        >>> plnpca.fit()
        >>> print(plnpca)
        """,
    )
    def fit(
        self,
        nb_max_epoch: int = 400,
        *,
        lr: float = 0.01,
        tol: float = 1e-5,
        do_smart_init: bool = True,
        verbose: bool = False,
    ):
        super().fit(
            nb_max_epoch,
            lr=lr,
            tol=tol,
            do_smart_init=do_smart_init,
            verbose=verbose,
        )

    @_add_doc(
        _model,
        example="""
            >>> import matplotlib.pyplot as plt
            >>> from pyPLNmodels import PlnPCA, load_scrna
            >>> endog, labels = load_scrna(return_labels = True, for_formula = False)
            >>> plnpca = PlnPCA(endog,add_const = True)
            >>> plnpca.fit()
            >>> plnpca.plot_expected_vs_true()
            >>> plt.show()
            >>> plnpca.plot_expected_vs_true(colors = labels)
            >>> plt.show()
            """,
    )
    def plot_expected_vs_true(self, ax=None, colors=None):
        super().plot_expected_vs_true(ax=ax, colors=colors)

    @_add_doc(
        _model,
        example="""
            >>> import matplotlib.pyplot as plt
            >>> from pyPLNmodels import PlnPCA, load_scrna
            >>> endog, labels = load_scrna(return_labels = True, for_formula = False)
            >>> plnpca = PlnPCA(endog,add_const = True)
            >>> plnpca.fit()
            >>> plnpca.viz()
            >>> plt.show()
            >>> plnpca.viz(colors = labels)
            >>> plt.show()
            >>> plnpca.viz(show_cov = True)
            >>> plt.show()
            """,
    )
    def viz(self, ax: matplotlib.axes.Axes = None, colors=None, show_cov: bool = False):
        super().viz(ax=ax, colors=colors, show_cov=show_cov)

    @_add_doc(
        _model,
        example="""
        >>> from pyPLNmodels import PlnPCA, load_scrna
        >>> data = load_scrna()
        >>> plnpca = PlnPCA.from_formula("endog ~ 1", data = data)
        >>> plnpca.fit()
        >>> pca_proj = plnpca.pca_projected_latent_variables()
        >>> print(pca_proj.shape)
        """,
    )
    def pca_projected_latent_variables(self, n_components: Optional[int] = None):
        return super().pca_projected_latent_variables(n_components=n_components)

    @_add_doc(
        _model,
        example="""
        >>> from pyPLNmodels import PlnPCA, load_scrna
        >>> data = load_scrna()
        >>> plnpca = PlnPCA.from_formula("endog ~ 1", data = data)
        >>> plnpca.fit()
        >>> plnpca.pca_pairplot(n_components = 5)
        """,
    )
    def pca_pairplot(self, n_components=None, colors=None):
        super().pca_pairplot(n_components=n_components, colors=colors)

    @_add_doc(
        _model,
        example="""
        >>> from pyPLNmodels import PlnPCA, load_scrna
        >>> data = load_scrna()
        >>> plnpca = PlnPCA.from_formula("endog ~ 1", data = data)
        >>> plnpca.fit()
        >>> plnpca.plot_pca_correlation_circle(["a","b"], indices_of_variables = [4,8])
        """,
    )
    def plot_pca_correlation_circle(
        self, variables_names: List[str], indices_of_variables=None
    ):
        super().plot_pca_correlation_circle(
            variables_names=variables_names,
            indices_of_variables=indices_of_variables,
            title=f", which are {self.rank} dimensional.",
        )

    @property
    @_add_doc(
        _model,
        example="""
        >>> from pyPLNmodels import PlnPCA, load_scrna
        >>> data = load_scrna()
        >>> plnpca = PlnPCA.from_formula("endog ~ 1", data = data)
        >>> plnpca.fit()
        >>> print(plnpca.latent_mean.shape)
        """,
    )
    def latent_mean(self) -> torch.Tensor:
        return super().latent_mean

    def _endog_predictions(self):
        covariance_a_posteriori = torch.sum(
            (self.components**2).unsqueeze(0) * (self.latent_sqrt_var**2).unsqueeze(1),
            axis=2,
        )
        if self.exog is not None:
            XB = self.exog @ self.coef
        else:
            XB = 0
        return torch.exp(
            self.offsets + XB + self.latent_variables + 1 / 2 * covariance_a_posteriori
        )

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
        self._latent_mean = self._smart_device(latent_mean)

    @_model.latent_sqrt_var.setter
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
        self._latent_sqrt_var = self._smart_device(latent_sqrt_var)

    @property
    def _directory_name(self) -> str:
        """
        Property representing the directory name.

        Returns
        -------
        str
            The directory name.
        """
        return f"{super()._directory_name}_rank_{self._rank}"

    @property
    def exog(self) -> torch.Tensor:
        """
        Property representing the exog.

        Returns
        -------
        torch.Tensor
            The exog tensor.
        """
        return self._cpu_attribute_or_none("_exog")

    @exog.setter
    @_array2tensor
    def exog(self, exog: Union[torch.Tensor, np.ndarray, pd.DataFrame]):
        """
        Setter for the exog.

        Parameters
        ----------
        exog : Union[torch.Tensor, np.ndarray, pd.DataFrame]
            The exog tensor.
        """
        _check_data_shape(self.endog, exog, self.offsets)
        self._exog = exog
        print("Setting coef to initialization")
        self._smart_init_coef()

    def _get_pca_low_dim_covariances(self, sk_components):
        C_tilde_C = sk_components @ self.components
        C_tilde_C_latent_var = C_tilde_C.unsqueeze(0) * (self.latent_var.unsqueeze(1))
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
        Get the maximum number of components possible by the model.
        """
        return self._rank

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
        pass

    @property
    def covariance(self) -> torch.Tensor:
        """
        Property representing the covariance of the latent variables.

        Returns
        -------
        Optional[torch.Tensor]
            The covariance tensor or None if components are not present.
        """
        if hasattr(self, "_components"):
            return self.components @ (self.components.T)
        return None

    @property
    def covariance_a_posteriori(self) -> Optional[torch.Tensor]:
        """
        Property representing the covariance a posteriori of the latent variables.

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
            return (self._components @ cov_latent @ self._components.T).detach()
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
    def projected_latent_variables(self) -> torch.Tensor:
        """
        Property representing the projected latent variables.

        Returns
        -------
        torch.Tensor
            The projected latent variables.
        """
        return torch.mm(self.latent_variables, self.ortho_components).detach()

    @property
    def ortho_components(self):
        """
        Orthogonal components of the model.
        """
        return torch.linalg.qr(self._components, "reduced")[0]

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
        self._components = self._smart_device(components)

    @_add_doc(
        _model,
        params="""
        Parameters
        ----------
        project : bool, optional
            Whether to project the latent variables, by default False.
        """,
        returns="""
        torch.Tensor
            The transformed endog (latent variables of the model).
        """,
        example="""
            >>> from pyPLNmodels import PlnPCA, load_scrna
            >>> data = load_scrna()
            >>> pca = PlnPCA.from_formula("endog ~ 1", data)
            >>> pca.fit()
            >>> transformed_endog_low_dim = pca.transform()
            >>> transformed_endog_high_dim = pca.transform(project = False)
            >>> print(transformed_endog_low_dim.shape)
            >>> print(transformed_endog_high_dim.shape)
            """,
    )
    def transform(self, project: bool = False) -> torch.Tensor:
        if project is True:
            return self.projected_latent_variables
        return self.latent_variables

    @property
    @_add_doc(
        _model,
        example="""
        >>> from pyPLNmodels import PlnPCA, load_scrna
        >>> data = load_scrna()
        >>> pca = PlnPCA.from_formula("endog ~ 1", data)
        >>> pca.fit()
        >>> print(pca.latent_variables.shape)
        """,
    )
    def latent_variables(self) -> torch.Tensor:
        return torch.matmul(self.latent_mean, self.components.T) + self.mean_gaussian

    @_add_doc(
        _model,
        example="""
            >>> from pyPLNmodels import PlnPCA, load_scrna
            >>> data = load_scrna()
            >>> pca = PlnPCA.from_formula("endog ~ 1", data)
            >>> pca.fit()
            >>> elbo = pca.compute_elbo()
            >>> print("elbo", elbo)
            >>> print("loglike/n", pca.loglike/pca.n_samples)
            """,
    )
    def compute_elbo(self) -> torch.Tensor:
        return elbo_plnpca(
            self._endog,
            self._exog,
            self._offsets,
            self._latent_mean,
            self._latent_sqrt_var,
            self._components,
            self._coef,
        )

    @_add_doc(_model)
    def _compute_elbo_b(self) -> torch.Tensor:
        return elbo_plnpca(
            self._endog_b.to(DEVICE),
            self._exog_b_device,
            self._offsets_b.to(DEVICE),
            self._latent_mean_b.to(DEVICE),
            self._latent_sqrt_var_b.to(DEVICE),
            self._components,
            self._coef,
        )

    @property
    def _per_sample_elbo(self) -> torch.Tensor:
        return per_sample_elbo_plnpca(
            self._endog,
            self._exog,
            self._offsets,
            self._latent_mean,
            self._latent_sqrt_var,
            self._components,
            self._coef,
        )

    @_add_doc(_model)
    def _random_init_model_parameters(self):
        super()._random_init_coef()
        self._components = torch.randn((self.dim, self._rank)).to(DEVICE)

    @_add_doc(_model)
    def _smart_init_model_parameters(self):
        if not hasattr(self, "_coef"):
            super()._smart_init_coef()
        if not hasattr(self, "_components"):
            self._components = _init_components(self._endog, self._rank).to(DEVICE)

    @_add_doc(_model)
    def _random_init_latent_parameters(self):
        """
        Randomly initialize the latent parameters.
        """
        self._latent_sqrt_var = (
            1 / 2 * self._smart_device(torch.ones((self.n_samples, self._rank)))
        )
        self._latent_mean = self._smart_device(torch.ones((self.n_samples, self._rank)))

    @_add_doc(_model)
    def _smart_init_latent_parameters(self):
        if not hasattr(self, "_latent_mean"):
            self._latent_mean = _init_latent_mean(
                self._endog,
                self._exog,
                self._offsets,
                self._coef,
                self._components,
            ).detach()
            self._latent_mean = self._smart_device(self._latent_mean)
        if not hasattr(self, "_latent_sqrt_var"):
            self._latent_sqrt_var = (
                1 / 2 * self._smart_device(torch.ones((self.n_samples, self._rank)))
            )

    @property
    @_add_doc(_model)
    def _list_of_parameters_needing_gradient(self):
        if self._coef is None:
            return [self._components, self._latent_mean, self._latent_sqrt_var]
        return [self._components, self._coef, self._latent_mean, self._latent_sqrt_var]

    @property
    @_add_doc(_model)
    def model_parameters(self) -> Dict[str, torch.Tensor]:
        return {"coef": self.coef, "components": self.components}

    @property
    @_add_doc(_model)
    def latent_parameters(self):
        return {
            "latent_sqrt_var": self.latent_sqrt_var,
            "latent_mean": self.latent_mean,
        }


class ZIPln(_model):
    """
    Zero-Inflated Pln (ZIPln) class. Like a Pln but adds zero-inflation
    modelled as row-wise (one inflation parameter per sample), column-wise
    (one inflation per variable) or global (one and only one inflation parameter).
    Fitting such a model is slower than fitting a Pln.

    Examples
    --------
    >>> from pyPLNmodels import ZIPln, Pln, load_microcosm
    >>> data = load_microcosm() # microcosm are higly zero-inflated (96% of zeros)
    >>> zi = ZIPln.from_formula("endog ~ 1 + site", data)
    >>> zi.fit()
    >>> zi.viz(colors = data["site"])
    >>> # Here Pln is not appropriate:
    >>> pln = Pln.from_formula("endog ~ 1 + site", data)
    >>> pln.fit()
    >>> pln.viz(colors = data["site"])
    >>> # Can also give different covariates:
    >>> zi_diff = ZIPln.from_formula("endog ~ 1 + site | 1 + time", data)
    >>> zi.fit()
    >>> zi.viz(colors = data["site"])
    >>> ## Or take all the covariates
    >>> zi_all = ZIPln.from_formula("endog ~ 1 + site*time | 1 + site*time", data)
    >>> zi_all.fit()

    >>> from pyPLNmodels import ZIPln, get_simulation_parameters, sample_zipln
    >>> param = get_simulation_parameters(nb_cov_inflation = 1, zero_inflation_formula = "column-wise")
    >>> endog = sample_zipln(param)
    >>> data = {"endog": endog, "exog": param.exog, "exog_infla": param.exog_inflation}
    >>> zi = ZIPln.from_formula("endog ~ 0 + exog | 0+ exog_infla", data)
    >>> zi.fit()
    >>> print(zi)
    """

    _NAME = "ZIPln"

    _latent_prob: torch.Tensor
    _coef_inflation: torch.Tensor
    _dirac: torch.Tensor

    def __init__(
        self,
        endog: Optional[Union[torch.Tensor, np.ndarray, pd.DataFrame]],
        *,
        exog: Optional[Union[torch.Tensor, np.ndarray, pd.DataFrame]] = None,
        exog_inflation: Optional[Union[torch.Tensor, np.ndarray, pd.DataFrame]] = None,
        offsets: Optional[Union[torch.Tensor, np.ndarray, pd.DataFrame]] = None,
        offsets_formula: {"zero", "logsum"} = "zero",
        zero_inflation_formula: {"column-wise", "row-wise", "global"} = "column-wise",
        dict_initialization: Optional[Dict[str, torch.Tensor]] = None,
        take_log_offsets: bool = False,
        add_const: bool = True,
        add_const_inflation: bool = True,
        use_closed_form_prob: bool = True,
        batch_size: int = None,
    ):
        """
        Initializes the ZIPln class.

        Parameters
        ----------
        endog : Union[torch.Tensor, np.ndarray, pd.DataFrame]
            The count data.
        exog : Union[torch.Tensor, np.ndarray, pd.DataFrame], optional(keyword-only)
            The covariate data. Defaults to None.
        exog_inflation : Union[torch.Tensor, np.ndarray, pd.DataFrame], optional(keyword-only)
            The covariate data for the inflation part. Defaults to None. If None,
            will automatically add a vector of one if zero_inflation_formula is
            either "column-wise" or "row-wise".
        offsets : Union[torch.Tensor, np.ndarray, pd.DataFrame], optional(keyword-only)
            The offsets data. Defaults to None.
        offsets_formula : str, optional(keyword-only)
            The formula for offsets. Defaults to "zero". Can be also "logsum"
            where we take the logarithm of the sum (of each line) of the counts.
            Overriden (useless) if offsets is not None.
        zero_inflation_formula: str {"column-wise", "row-wise", "global"}
            The modelling of the zero_inflation. Either "column-wise", "row-wise"
            or "global". Default to "column-wise".
        dict_initialization : dict, optional(keyword-only)
            The initialization dictionary loading a previously saved model.
            Defaults to None.
        take_log_offsets : bool, optional(keyword-only)
            Whether to take the log of offsets. Defaults to False.
        add_const : bool, optional(keyword-only)
            Whether to add a column of one in the exog. Defaults to True.
        add_const_inflation : bool, optional(keyword-only)
            Whether to add a column of one in the exog_inflation. Defaults to True.
            If exog_inflation is None and zero_inflation_formula is not "global",
            add_const_inflation is set to True anyway and a warnings
            is launched.
        use_closed_form_prob : bool, optional
            Whether or not use the closed formula for the latent probability.
            Default is True.
        batch_size: int, optional(keyword-only)
            The batch size when optimizing the elbo. If None,
            batch gradient descent will be performed (i.e. batch_size = n_samples).
        Raises
        ------
        Returns
        -------
        A ZIPln object
        See also
        --------
        :func:`pyPLNmodels.ZIPln.from_formula`
        Examples
        --------
        >>> from pyPLNmodels import ZIPln, load_scrna
        >>> endog = load_scrna(for_formula = False)
        >>> zi = ZIPln(endog, add_const = True)
        >>> zi.fit()
        >>> print(zi)
        """
        self._use_closed_form_prob = use_closed_form_prob
        _check_formula(zero_inflation_formula)
        self._zero_inflation_formula = zero_inflation_formula
        (
            self._endog,
            self._exog,
            self._exog_inflation,
            self._offsets,
            self.column_endog,
            self._dirac,
            self._batch_size,
            self.samples_only_zeros,
        ) = _handle_data_with_inflation(
            endog,
            exog,
            exog_inflation,
            offsets,
            offsets_formula,
            self._zero_inflation_formula,
            take_log_offsets,
            add_const,
            add_const_inflation,
            batch_size,
        )
        self._fitted = False
        self._criterion_args = _CriterionArgs()
        if dict_initialization is not None:
            self._set_init_parameters(dict_initialization)

    def _extract_batch(self, batch):
        super()._extract_batch(batch)
        self._dirac_b = batch[5]
        if self._use_closed_form_prob is False:
            self._latent_prob_b = batch[6]

    def _return_batch(self, indices, beginning, end):
        pln_batch = super()._return_batch(indices, beginning, end)
        dirac_b = torch.index_select(self._dirac, 0, self.to_take)
        batch = pln_batch + (dirac_b,)
        if self._use_closed_form_prob is False:
            to_return = torch.index_select(self._latent_prob, 0, self.to_take)
            return batch + (torch.index_select(self._latent_prob, 0, self.to_take),)
        return batch

    @_add_doc(
        _model,
        example="""
            >>> import matplotlib.pyplot as plt
            >>> from pyPLNmodels import ZIPln, load_microcosm
            >>> data = load_microcosm()
            >>> zi = ZIPln.from_formula("endog ~ 1 + site", data = data)
            >>> zi.fit()
            >>> zi.viz()
            >>> plt.show()
            """,
    )
    def viz(self, ax=None, colors=None, show_cov: bool = False):
        super().viz(ax=ax, colors=colors, show_cov=show_cov)

    @classmethod
    def from_formula(
        cls,
        formula: str,
        data: Dict[str, Union[torch.Tensor, np.ndarray, pd.DataFrame]],
        zero_inflation_formula: {"column-wise", "row-wise", "global"} = "column-wise",
        *,
        offsets_formula: {"zero", "logsum"} = "zero",
        dict_initialization: Optional[Dict[str, torch.Tensor]] = None,
        take_log_offsets: bool = False,
        use_closed_form_prob: bool = True,
        batch_size: int = None,
    ):
        """
        Create a ZIPln instance from a formula and data.

        Parameters
        ----------
        formula : str
            The formula. Can not take into account the exog_inflation.
            They are automatically set to exog. If separate exog are needed,
            do not use the from_formula classmethod.
        data : dict
            The data dictionary. Each value can be either a torch.Tensor,
            a np.ndarray or pd.DataFrame
        zero_inflation_formula: str {"column-wise", "row-wise", "global"}
            The modelling of the zero_inflation. Either "column-wise", "row-wise"
            or "global". Default to "column-wise".
        offsets_formula : str, optional(keyword-only)
            The formula for offsets. Defaults to "zero". Can be also "logsum" where
            we take the logarithm of the sum (of each line) of the counts. Overriden (useless)
            if data["offsets"] is not None.
        dict_initialization : dict, optional(keyword-only)
            The initialization dictionary. Defaults to None.
        take_log_offsets : bool, optional(keyword-only)
            Whether to take the log of offsets. Defaults to False.
        use_closed_form_prob : bool, optional
            Whether or not use the closed formula for the latent probability.
            Default is True.
        batch_size: int, optional(keyword-only)
            The batch size when optimizing the elbo. If None,
            batch gradient descent will be performed (i.e. batch_size = n_samples).
        Returns
        -------
        A ZIPln object
        See also
        --------
        :class:`pyPLNmodels.ZIPln`
        :func:`pyPLNmodels.ZIPln.__init__`
        Examples
        --------
        >>> from pyPLNmodels import ZIPln, load_microcosm
        >>> data = load_microcosm()
        >>> zi = ZIPln.from_formula("endog ~ 1", data = data)

        >>> from pyPLNmodels import ZIPln, load_scrna
        >>> data = load_scrna()
        >>> zi = ZIPln.from_formula("endog ~ 1", data = data)
        """
        ### add_const is inside the formula so it is set to False for the initialisation of
        ### the class.
        add_const_inflation = False
        if "|" not in formula:
            msg = "exog_inflation are set to exog (if any). If you need different exog_inflation, "
            msg += "specify it with a | like in the following: endog ~ 1 +x | x + y "
            print(msg)
            endog, exog, offsets = _extract_data_from_formula_no_infla(formula, data)
            exog_infla = exog
        else:
            endog, exog, exog_infla, offsets = _extract_data_from_formula_with_infla(
                formula, data
            )
            ## Problem if the exog inflation is 1, we can not infer the shape.
            if exog_infla is None:
                add_const_inflation = True
        return cls(
            endog,
            exog=exog,
            exog_inflation=exog_infla,
            offsets=offsets,
            offsets_formula=offsets_formula,
            zero_inflation_formula=zero_inflation_formula,
            dict_initialization=dict_initialization,
            take_log_offsets=take_log_offsets,
            add_const=False,
            add_const_inflation=add_const_inflation,
            use_closed_form_prob=use_closed_form_prob,
            batch_size=batch_size,
        )

    @_add_doc(
        _model,
        example="""
        >>> from pyPLNmodels import ZIPln, load_scrna
        >>> data = load_scrna()
        >>> zi = ZIPln.from_formula("endog ~ 1", data)
        >>> zi.fit()
        >>> print(zi)

        >>> from pyPLNmodels import ZIPln, load_scrna
        >>> data = load_scrna()
        >>> zi = ZIPln.from_formula("endog ~ 1", data)
        >>> zi.fit( nb_max_epoch = 500, verbose = True)
        >>> print(zi)
        """,
    )
    def fit(
        self,
        nb_max_epoch: int = 400,
        *,
        lr: float = 0.01,
        tol: float = 1e-5,
        do_smart_init: bool = True,
        verbose: bool = False,
    ):
        super().fit(
            nb_max_epoch,
            lr=lr,
            tol=tol,
            do_smart_init=do_smart_init,
            verbose=verbose,
        )

    @_add_doc(
        _model,
        example="""
            >>> import matplotlib.pyplot as plt
            >>> from pyPLNmodels import ZIPln, load_scrna
            >>> endog, labels = load_scrna(return_labels = True, for_formula = False)
            >>> zi = ZIPln(endog,add_const = True)
            >>> zi.fit()
            >>> zi.plot_expected_vs_true()
            >>> plt.show()
            >>> zi.plot_expected_vs_true(colors = labels)
            >>> plt.show()
            """,
    )
    def plot_expected_vs_true(self, ax=None, colors=None):
        super().plot_expected_vs_true(ax=ax, colors=colors)

    @property
    def _description(self):
        msg = "full covariance model with "
        msg += f"{self._zero_inflation_formula} zero-inflation"
        if self._use_closed_form_prob is True:
            msg += f" and closed form for latent prob."
        else:
            msg += f" and NO closed form for latent prob."
        return msg

    @property
    def nb_cov_inflation(self):
        """
        Number of covariates for the inflation part in the model.
        If the zero_inflation_formula is "global", return 0.
        """
        if self._zero_inflation_formula == "global":
            return 0
        elif self._zero_inflation_formula == "column-wise":
            return self.exog_inflation.shape[1]
        return self.exog_inflation.shape[0]

    def _random_init_model_parameters(self):
        if self._zero_inflation_formula == "global":
            self._coef_inflation = torch.tensor([0.5]).to(DEVICE)
        elif self._zero_inflation_formula == "row-wise":
            self._coef_inflation = self._smart_device(
                torch.randn(self.n_samples, self.nb_cov_inflation)
            )
        elif self._zero_inflation_formula == "column-wise":
            self._coef_inflation = torch.randn(self.nb_cov_inflation, self.dim).to(
                DEVICE
            )

        if self.nb_cov == 0:
            self._coef = None
        else:
            self._coef = torch.randn(self.nb_cov, self.dim).to(DEVICE)
        self._components = torch.randn(self.dim, self.dim).to(DEVICE)

    # should change the good initialization for _coef_inflation
    def _smart_init_model_parameters(self):
        """
        Zero Inflated Poisson is fitted for the coef and coef_inflation.
        For the components, PCA on the log counts.
        """
        coef, coef_inflation, self.rec_error_init = _init_coef_coef_inflation(
            self.endog,
            self.exog,
            self.exog_inflation,
            self.offsets,
            self._zero_inflation_formula,
        )
        if not hasattr(self, "_coef_inflation"):
            self._coef_inflation = self._smart_device(coef_inflation)
        if not hasattr(self, "_coef"):
            self._coef = coef.to(DEVICE) if coef is not None else None
        if not hasattr(self, "_components"):
            self._components = torch.clone(_init_components(self._endog, self.dim)).to(
                DEVICE
            )

    def _random_init_latent_parameters(self):
        self._latent_mean = self._smart_device(torch.randn(self.n_samples, self.dim))
        self._latent_sqrt_var = self._smart_device(
            torch.randn(self.n_samples, self.dim)
        )
        if self._use_closed_form_prob is False:
            self._latent_prob = self._smart_device(
                (
                    torch.empty(self.n_samples, self.dim).uniform_(0, 1) * self._dirac
                ).double()
            )

    def _smart_init_latent_parameters(self):
        if not hasattr(self, "_latent_mean"):
            if self.exog is not None:
                self._latent_mean = self._smart_device(self.mean_gaussian)
            else:
                self._latent_mean = self._smart_device(
                    torch.log(self._endog + (self._endog == 0))
                )

        if not hasattr(self, "_latent_sqrt_var"):
            self._latent_sqrt_var = self._smart_device(
                torch.randn(self.n_samples, self.dim)
            )
        if not hasattr(self, "_latent_prob"):
            if self._use_closed_form_prob is False:
                self._latent_prob = self._smart_device(
                    self._proba_inflation * (self._dirac)
                )

    @property
    def _covariance(self):
        return self._components @ (self._components.T)

    def _get_max_components(self):
        """
        Method for getting the maximum number of components.

        Returns
        -------
        int
            The maximum number of components.
        """
        return min(self.dim, self.n_samples)

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

    @property
    def latent_variables(self) -> tuple([torch.Tensor, torch.Tensor]):
        """
        Property representing the latent variables. Two latent
        variables are available if exog is not None

        Returns
        -------
        tuple(torch.Tensor, torch.Tensor)
            The latent variables of a classic Pln model (size (n_samples, dim))
            and zero inflated latent variables of size (n_samples, dim).
        Examples
        --------
        >>> from pyPLNmodels import ZIPln, load_scrna
        >>> data = load_scrna()
        >>> zi = ZIPln.from_formula("endog ~ 1", data)
        >>> zi.fit()
        >>> latent_mean, latent_inflated = zi.latent_variables
        >>> print(latent_mean.shape)
        >>> print(latent_inflated.shape)
        """
        return self.latent_mean, self.latent_prob

    def transform(self, return_latent_prob=False):
        """
        Method for transforming the endog. Can be seen as a
        normalization of the endog. Can return a the latent probability
        if required.

        Parameters
        ----------
        return_latent_prob: bool, optional
            Wheter to return or not the latent_probability of zero inflation.
        Returns
        -------
        The latent mean if `return_latent_prob` is False and (latent_mean, latent_prob) else.
        """
        latent_gaussian = (
            1 - self.latent_prob
        ) * self.latent_mean + self.mean_gaussian * self.latent_prob
        if return_latent_prob is True:
            return latent_gaussian, self.latent_prob
        return latent_gaussian

    def _endog_predictions(self):
        return torch.exp(
            self.offsets + self.latent_mean + 1 / 2 * self.latent_sqrt_var**2
        ) * (1 - self.latent_prob)

    @property
    def coef_inflation(self):
        """
        Property representing the coefficients of the inflation.

        Returns
        -------
        torch.Tensor or None
            The coefficients or None.
        """
        return self._cpu_attribute_or_none("_coef_inflation")

    @property
    def exog_inflation(self):
        """
        Property representing the exog of the inflation.

        Returns
        -------
        torch.Tensor or None
            The exog_inflation or None.
        """
        return self._cpu_attribute_or_none("_exog_inflation")

    @exog_inflation.setter
    @_array2tensor
    def exog_inflation(
        self, exog_inflation: Union[torch.Tensor, np.ndarray, pd.DataFrame]
    ):
        """
        Setter for the exog_inflation property.

        Parameters
        ----------
        exog_inflation : Union[torch.Tensor, np.ndarray, pd.DataFrame]
            The exog for the inflation part.
        """
        # _check_data_shape(self.endog, exog, self.offsets)
        _check_right_exog_inflation_shape(
            exog_inflation, self.n_samples, self.dim, self._zero_inflation_formula
        )
        self._exog_inflation = exog_inflation
        print("Setting coef_inflation to initialization")
        _, self._coef_inflation, _ = _init_coef_coef_inflation(
            self.endog,
            self.exog,
            self.exog_inflation,
            self.offsets,
            self._zero_inflation_formula,
        )

    @coef_inflation.setter
    @_array2tensor
    def coef_inflation(
        self, coef_inflation: Union[torch.Tensor, np.ndarray, pd.DataFrame]
    ):
        """
        Setter for the coef_inflation property.

        Parameters
        ----------
        coef : Union[torch.Tensor, np.ndarray, pd.DataFrame]
            The coefficients of size (nb_cov_inflation,dim) if zero_inflation_formula
            is "column-wise", (n_samples, nb_cov_inflation) if zero_inflation_formula
            is "row-wsie" and a scalar if zero_inflation_formula is "global".

        Raises
        ------
        ValueError
            If the shape of the coef_inflation is incorrect.
        """
        if not self._has_right_coef_infla_shape(coef_inflation.shape):
            msg = "Wrong shape for the coef_inflation. Expected "
            msg += f"{self._shape_coef_infla}, got {coef_inflation.shape}"
            raise ValueError(msg)
        self._coef_inflation = self._smart_device(coef_inflation)

    def _has_right_coef_infla_shape(self, shape):
        if self._zero_inflation_formula == "global":
            return shape.numel() < 2
        return shape == self._shape_coef_infla

    @property
    def _shape_coef_infla(self):
        if self._zero_inflation_formula == "global":
            return torch.Size([])
        if self._zero_inflation_formula == "column-wise":
            return (self.nb_cov_inflation, self.dim)
        return (self.n_samples, self.nb_cov_inflation)

    @_model.latent_sqrt_var.setter
    @_array2tensor
    def latent_sqrt_var(
        self, latent_sqrt_var: Union[torch.Tensor, np.ndarray, pd.DataFrame]
    ):
        """
        Setter for the latent variance property.

        Parameters
        ----------
        latent_sqrt_var : Union[torch.Tensor, np.ndarray, pd.DataFrame]
            The latent square root of the variance.

        Raises
        ------
        ValueError
            If the shape of the latent variance is incorrect
            (i.e. should be (n_samples, dim)).
        """
        if latent_sqrt_var.shape != self.endog.shape:
            raise ValueError(
                f"Wrong shape. Expected {self.endog.shape}, got {latent_sqrt_var.shape}"
            )
        self._latent_sqrt_var = self._smart_device(latent_sqrt_var)

    def _project_parameters(self):
        self._project_latent_prob()

    def _project_latent_prob(self):
        """Ensure the latent probabilites stays in [0,1]."""
        if self._use_closed_form_prob is False:
            with torch.no_grad():
                self._latent_prob = torch.maximum(
                    self._latent_prob,
                    self._smart_device(torch.tensor([0])),
                    out=self._latent_prob,
                )
                self._latent_prob = torch.minimum(
                    self._latent_prob,
                    self._smart_device(torch.tensor([1])),
                    out=self._latent_prob,
                )
                self._latent_prob *= self._dirac

    @property
    def covariance(self) -> torch.Tensor:
        """
        Property representing the covariance of the latent variables.

        Returns
        -------
        Optional[torch.Tensor]
            The covariance tensor or None if components are not present.
        """
        return self._cpu_attribute_or_none("_covariance")

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
            If the components have an invalid shape
            (i.e. should be (dim,dim)).
        """
        if components.shape != (self.dim, self.dim):
            raise ValueError(
                f"Wrong shape. Expected {self.dim, self.dim}, got {components.shape}"
            )
        self._components = self._smart_device(components)

    @property
    def latent_prob(self):
        """
        The latent probability i.e. the probabilities that the zero inflation
        component is 0 given Y.
        """
        if self._use_closed_form_prob is True:
            return self.closed_formula_latent_prob.detach().cpu()
        return self._cpu_attribute_or_none("_latent_prob")

    @latent_prob.setter
    @_array2tensor
    def latent_prob(self, latent_prob: Union[torch.Tensor, np.ndarray, pd.DataFrame]):
        """
        Setter for the latent_probabilities.

        Parameters
        ----------
        latent_prob : torch.Tensor
            The latent_probabilities to set.

        Raises
        ------
        ValueError
            If the latent_prob have an invalid shape
            (i.e. should be (n_samples,dim)), or
            if you assign probabilites greater than 1 or lower
            than 0, and if you assign non-zero probabilities
            to non-zero counts.
        >>> from pyPLNmodels import ZIPln, load_scrna
        >>> endog = load_scrna(for_formula = False)
        >>> zi = ZIPln(endog,add_const = True, use_closed_form_prob = False)
        >>> zi.fit()
        >>> latent_prob = zi.latent_prob
        >>> zi.latent_prob = latent_prob*0.5
        """
        if self._use_closed_form_prob is True:
            raise ValueError(
                "Can not set the latent prob when the closed form is used."
            )
        if latent_prob.shape != (self.n_samples, self.dim):
            raise ValueError(
                f"Wrong shape. Expected {self.n_samples, self.dim}, got {latent_prob.shape}"
            )
        if torch.max(latent_prob) > 1 or torch.min(latent_prob) < 0:
            raise ValueError(f"Wrong value. All values should be between 0 and 1.")
        if torch.norm(latent_prob * (self._endog == 0) - latent_prob) > 0.00000001:
            raise ValueError(
                "You can not assign non zeros inflation probabilities to non zero counts."
            )
        self._latent_prob = latent_prob

    @property
    def closed_formula_latent_prob(self):
        """
        The closed form for the latent probability.
        Uses the exponential moment of a log gaussian variable.
        """
        return _closed_formula_latent_prob(
            self._exog_device,
            self._coef,
            self._offsets.to(DEVICE),
            self.xinflacoefinfla.to(DEVICE),
            self._covariance,
            self._dirac.to(DEVICE),
        )

    @property
    def _exog_device(self):
        if self._exog is None:
            return None
        else:
            return self._exog.to(DEVICE)

    @property
    def closed_formula_latent_prob_b(self):
        """
        The closed form for the latent probability for the batch.
        Uses the exponential moment of a log gaussian variable.
        """
        return _closed_formula_latent_prob(
            self._exog_b_device,
            self._coef,
            self._offsets_b.to(DEVICE),
            self._xinflacoefinfla_b,
            self._covariance,
            self._dirac_b.to(DEVICE),
        )

    @_add_doc(
        _model,
        example="""
            >>> from pyPLNmodels import ZIPln, load_scrna
            >>> data = load_scrna()
            >>> zi = ZIPln.from_formula("endog ~ 1", data)
            >>> zi.fit()
            >>> elbo = zi.compute_elbo()
            >>> print("elbo", elbo)
            >>> print("loglike/n", zi.loglike/zi.n_samples)
            """,
        see_also="""
        :func:`pyPLNmodels.elbos.elbo_zi_pln`
        """,
    )
    def compute_elbo(self):
        if self._use_closed_form_prob is True:
            latent_prob = self.closed_formula_latent_prob
        else:
            latent_prob = self._latent_prob
        return elbo_zi_pln(
            self._endog,
            self._exog,
            self._offsets,
            self._latent_mean,
            self._latent_sqrt_var,
            latent_prob,
            self._components,
            self._coef,
            self._xinflacoefinfla,
            self._dirac,
        )

    @property
    def _xinflacoefinfla_b(self):
        """Computes the term exog_infla_b @ coef_infla
        or coef_infla_b @ exog_infla depending on the
        zero_inflation_formula.
        """
        if self._zero_inflation_formula == "global":
            return self._coef_inflation
        if self._zero_inflation_formula == "column-wise":
            exog_infla_b = torch.index_select(self._exog_inflation, 0, self.to_take)
            return exog_infla_b.to(DEVICE) @ self._coef_inflation
        coef_infla_b = torch.index_select(self._coef_inflation, 0, self.to_take).to(
            DEVICE
        )
        return coef_infla_b @ self._exog_inflation

    @property
    def _xinflacoefinfla(self):
        """Computes the term exog_infla @ coef_infla
        or coef_infla @ exog_infla depending on the
        zero_inflation_formula.
        """
        if self._zero_inflation_formula == "global":
            return self._smart_device(self._coef_inflation)
        elif self._zero_inflation_formula == "column-wise":
            return self._exog_inflation @ (self._smart_device(self._coef_inflation))
        elif self._zero_inflation_formula == "row-wise":
            return self._smart_device(self._coef_inflation) @ (self._exog_inflation)

    @property
    def proba_inflation(self):
        """
        Probability of observing a zero inflation.
        Even if the counts are non-zero, the probability of observing
        a zero inflation can be positive.
        """
        return self._proba_inflation.detach().cpu()

    @property
    def _proba_inflation(self):
        """
        Probability of observing a zero inflation.
        Even if the counts are non-zero, the probability of observing
        a zero inflation can be positive.
        """
        return torch.sigmoid(self._xinflacoefinfla)

    def _compute_elbo_b(self) -> torch.Tensor:
        if self._use_closed_form_prob is True:
            latent_prob_b = self.closed_formula_latent_prob_b
        else:
            latent_prob_b = self._latent_prob_b

        return elbo_zi_pln(
            self._endog_b.to(DEVICE),
            self._exog_b_device,
            self._offsets_b.to(DEVICE),
            self._latent_mean_b.to(DEVICE),
            self._latent_sqrt_var_b.to(DEVICE),
            latent_prob_b,
            self._components,
            self._coef,
            self._xinflacoefinfla_b,
            self._dirac_b.to(DEVICE),
        )

    @property
    def xinflacoefinfla(self):
        """Computes the term exog_infla @ coef_infla
        or coef_infla @ exog_infla depending on the
        zero_inflation_formula.
        """
        if self._zero_inflation_formula == "global":
            return self.coef_inflation
        elif self._zero_inflation_formula == "column-wise":
            return self.exog_inflation @ self.coef_inflation
        elif self._zero_inflation_formula == "row-wise":
            return self.coef_inflation @ self.exog_inflation

    @property
    @_add_doc(_model)
    def number_of_parameters(self) -> int:
        nb_param = self.dim * (self.nb_cov + (self.dim + 1) / 2 + self.nb_cov_inflation)
        if self._zero_inflation_formula == "global":
            return nb_param + 1
        else:
            return nb_param

    @property
    @_add_doc(_model)
    def _list_of_parameters_needing_gradient(self):
        list_parameters = [
            self._latent_mean,
            self._latent_sqrt_var,
            self._components,
            self._coef_inflation,
        ]
        if self._use_closed_form_prob is False:
            list_parameters.append(self._latent_prob)
        if self._exog is not None:
            list_parameters.append(self._coef)
        return list_parameters

    @property
    @_add_doc(_model)
    def model_parameters(self) -> Dict[str, torch.Tensor]:
        return {
            "coef": self.coef,
            "components": self.components,
            "coef_inflation": self.coef_inflation,
        }

    def predict_prob_inflation(
        self, exog_infla: Union[torch.Tensor, np.ndarray, pd.DataFrame] = None
    ):
        """
        Method for estimating the probability of a zero coming from the zero inflated component.

        Parameters
        ----------
        exog_infla : Union[torch.Tensor, np.ndarray, pd.DataFrame]
            The exog.

        Returns
        -------
        torch.Tensor
            The predicted values.

        Raises
        ------
        RuntimeError
            If the shape of the exog is incorrect.

        Notes
        -----
        - The mean sigmoid(exog @ coef_inflation) is returned.
        - `exog_infla` should have the shape `(_, nb_cov)`, where `nb_cov` is the number of exog variables.
        """
        exog_infla = _format_data(exog_infla)
        if self._zero_inflation_formula == "global":
            if exog_infla is not None:
                msg = "Can t predict with exog_infla. Exog_infla should be None"
                raise AttributeError(msg)
            return torch.sigmoid(self.coef_inflation)

        if self._zero_inflation_formula == "column-wise":
            if exog_infla.shape[-1] != self.nb_cov_inflation:
                error_string = f"X has wrong shape:({exog_infla.shape}). Should"
                error_string += f" be (_, {self.nb_cov_inflation})."
                raise RuntimeError(error_string)
            xb = exog_infla @ self.coef_inflation

        if self._zero_inflation_formula == "row-wise":
            if exog_infla.shape != self._exog_inflation.shape:
                error_string = f"X has wrong shape:({exog_infla.shape}). Should"
                error_string += f" be ({self._exog_inflation.shape})."
                raise RuntimeError(error_string)
            xb = self.coef_inflation @ exog_infla
        return torch.sigmoid(xb)

    @property
    @_add_doc(_model)
    def latent_parameters(self):
        latent_param = {
            "latent_sqrt_var": self.latent_sqrt_var,
            "latent_mean": self.latent_mean,
        }
        if self._use_closed_form_prob is False:
            latent_param["latent_prob"] = self.latent_prob
        return latent_param

    @property
    def _additional_methods_string(self):
        """
        Abstract property representing the additional methods string.
        """
        return (
            ".visualize_latent_prob(), .pca_pairplot_prob(), .predict_prob_inflation() "
        )

    @property
    def _additional_properties_string(self) -> str:
        """
        Property representing the additional properties string.

        Returns
        -------
        str
            The additional properties string.
        """
        return ".projected_latent_variables, .latent_prob, .proba_inflation"

    def visualize_latent_prob(self, indices_of_samples=None, indices_of_variables=None):
        """Visualize the latent probabilities via a heatmap."""
        latent_prob = self.latent_prob
        fig, ax = plt.subplots(figsize=(20, 20))
        if indices_of_samples is None:
            if self.n_samples > 1000:
                mess = "Visualization of the whole dataset not possible "
                mess += f"as n_samples ={self.n_samples} is too big (>1000). "
                mess += "Please provide the argument 'indices_of_samples', "
                mess += "with the needed samples number."
                raise ValueError(mess)
            indices_of_samples = np.arange(self.n_samples)
        elif indices_of_variables is None:
            if self.dim > 1000:
                mess = "Visualization of all variables not possible "
                mess += f"as dim ={self.dim} is too big(>1000). "
                mess += "Please provide the argument 'indices_of_variables', "
                mess += "with the needed variables number."
                raise ValueError(mess)
            indices_of_variables = np.arange(self.dim)
        latent_prob = latent_prob[indices_of_samples][:, indices_of_variables].squeeze()
        sns.heatmap(latent_prob, ax=ax)
        ax.set_title("Latent probability to be zero inflated.")
        ax.set_xlabel("Variable number")
        ax.set_ylabel("Sample number")
        plt.show()

    def pca_pairplot_prob(self, n_components=None, colors=None):
        """
        Generates a scatter matrix plot based on Principal Component Analysis (PCA)
        on the latent probabilitiess.

        Parameters
        ----------
            n_components (int, optional): The number of components to consider for plotting.
                If not specified, the maximum number of components will be used. Note that
                it will not display more than 10 graphs.
                Defaults to None.

            colors (np.ndarray): An array with one label for each
                sample in the endog property of the object.
                Defaults to None.
        Raises
        ------
            ValueError: If the number of components requested is greater than
                the number of variables in the dataset.
        """
        n_components = self._threshold_n_components(n_components)
        array = self.latent_prob.detach()
        _pca_pairplot(array.numpy(), n_components, self.dim, colors)

    @property
    def _directory_name(self):
        """
        Property representing the directory name.

        Returns
        -------
        str
            The directory name.
        """
        return f"{self._NAME}_nbcov_{self.nb_cov}_dim_{self.dim}_nbcovinfla_{self.nb_cov_inflation}_zero_infla_{self.writable_zero_formula}"

    @property
    def writable_zero_formula(self):
        return self._zero_inflation_formula.replace("-", "")

    def viz_prob(self, *, colors=None, ax=None):
        """
        Visualize the latent probabilites with a classic PCA.

        Parameters
        ----------
        ax : Optional[matplotlib.axes.Axes], optional(keyword-only)
            The matplotlib axis to use. If None, the current axis is used, by default None.
        colors : Optional[np.ndarray], optional(keyword-only)
            The colors to use for plotting, by default None.
        Raises
        ------

        Returns
        -------
        Any
            The matplotlib axis.
        """
        variables = self.latent_prob
        return self._viz_variables(variables, colors=colors, ax=ax, show_cov=False)


class Brute_ZIPln(ZIPln):
    @property
    def _description(self):
        msg = "full covariance model and brute zero-inflation with"
        msg += f" {self._zero_inflation_formula} inflation"
        if self._use_closed_form_prob is True:
            msg += " and closed form for latent prob."
        else:
            msg += " and NO closed form for latent prob."
        return msg

    def _compute_elbo_b(self) -> torch.Tensor:
        if self._use_closed_form_prob is True:
            latent_prob_b = self.closed_formula_latent_prob_b
            tocompute = elbo_brute_zipln_components
            cov_or_components = self._components
        else:
            latent_prob_b = self._closed_formula_zero_grad_prob_b
            tocompute = elbo_brute_zipln_covariance
            cov_or_components = self._covariance
        return tocompute(
            self._endog_b.to(DEVICE),
            self._exog_b_device,
            self._offsets_b.to(DEVICE),
            self._latent_mean_b.to(DEVICE),
            self._latent_sqrt_var_b.to(DEVICE),
            latent_prob_b.to(DEVICE),
            cov_or_components,
            self._coef,
            self._xinflacoefinfla_b,
            self._dirac_b.to(DEVICE),
        )

    @property
    def _closed_formula_zero_grad_prob_b(self):
        return _closed_formula_zero_grad_prob(
            self._offsets_b.to(DEVICE),
            self._latent_mean_b.to(DEVICE),
            self._latent_sqrt_var_b.to(DEVICE),
            self._dirac_b.to(DEVICE),
            self._xinflacoefinfla_b,
        )

    @property
    def _closed_formula_zero_grad_prob(self):
        return _closed_formula_zero_grad_prob(
            self._offsets,
            self._latent_mean,
            self._latent_sqrt_var,
            self._dirac,
            self._xinflacoefinfla,
        )

    @property
    def _covariance(self):
        if self._use_closed_form_prob is False:
            return _closed_formula_covariance(
                self._exog,
                self._latent_mean,
                self._latent_sqrt_var,
                self._coef,
                self.n_samples,
            )
        return self._components @ (self._components.T)

    @property
    def _list_of_parameters_needing_gradient(self):
        list_parameters = [
            self._latent_mean,
            self._latent_sqrt_var,
            self._coef_inflation,
        ]
        if self._use_closed_form_prob is True:
            list_parameters.append(self._coef)
            list_parameters.append(self._components)
        return list_parameters

    def _update_closed_forms(self):
        if self._use_closed_form_prob is True:
            self._latent_prob = self.closed_formula_latent_prob
        else:
            self._coef = _closed_formula_coef(self._exog, self._latent_mean)

    @property
    def _components_(self):
        if self._use_closed_form_prob is True:
            return self._components
        return torch.linalg.cholesky(self._covariance)

    @property
    @_add_doc(_model)
    def model_parameters(self) -> Dict[str, torch.Tensor]:
        return {
            "coef": self.coef,
            "components": self._components_,
            "coef_inflation": self.coef_inflation,
        }

    @property
    def latent_prob(self):
        """
        The latent probability i.e. the probabilities that the zero inflation
        component is 0 given Y.
        """
        if self._use_closed_form_prob is True:
            return self.closed_formula_latent_prob.detach().cpu()
        return self._closed_formula_zero_grad_prob.detach().cpu()
