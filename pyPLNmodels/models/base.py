# pylint:disable = too-many-lines
from abc import ABC, abstractmethod
from typing import Union, Optional
import warnings

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.decomposition import PCA

from pyPLNmodels.utils._data_handler import (
    _handle_data,
    _extract_data_from_formula,
    _array2tensor,
)
from pyPLNmodels.utils._criterion import _ElboCriterionMonitor
from pyPLNmodels.utils._utils import (
    _TimeRecorder,
    _nice_string_of_dict,
    _process_column_index,
    _shouldbefitted,
)
from pyPLNmodels.utils._viz import (
    _viz_variables,
    BaseModelViz,
    plot_correlation_circle,
    _pca_pairplot,
    _plot_expected_vs_true,
    _biplot,
)


DEFAULT_TOL = 1e-6


class BaseModel(
    ABC
):  # pylint: disable=too-many-instance-attributes, too-many-public-methods
    """
    Abstract base class for all the PLN based models that will be derived.
    """

    _time_recorder: _TimeRecorder
    optim: torch.optim.Optimizer
    _dict_list_mse: dict

    _latent_mean: torch.Tensor
    _latent_sqrt_variance: torch.Tensor

    _coef: torch.Tensor
    _covariance: torch.Tensor

    _ModelViz = BaseModelViz
    remove_zero_columns = True

    def __init__(
        self,
        endog: Union[torch.Tensor, np.ndarray, pd.DataFrame],
        *,
        exog: Optional[Union[torch.Tensor, np.ndarray, pd.DataFrame]] = None,
        offsets: Optional[Union[torch.Tensor, np.ndarray, pd.DataFrame]] = None,
        compute_offsets_method: {"zero", "logsum"} = "zero",
        add_const: bool = True,
    ):  # pylint: disable=too-many-arguments
        """
        Initializes the model class.

        Parameters
        ----------
        endog : Union[torch.Tensor, np.ndarray, pd.DataFrame]
            The count data.
        exog : Union[torch.Tensor, np.ndarray, pd.DataFrame], optional(keyword-only)
            The covariate data. Defaults to `None`.
        offsets : Union[torch.Tensor, np.ndarray, pd.DataFrame], optional(keyword-only)
            The offsets data. Defaults to `None`.
        compute_offsets_method : str, optional(keyword-only)
            Method to compute offsets if not provided. Options are:
                - "zero" that will set the offsets to zero.
                - "logsum" that will take the logarithm of the sum (per line) of the counts.
            Overridden (useless) if `offsets` is not None.
        add_const: bool, optional(keyword-only)
            Whether to add a column of one in the `exog`. Defaults to `True`.
        """

        (
            self._endog,
            self._exog,
            self._offsets,
            self.column_names_endog,
            self.column_names_exog,
        ) = _handle_data(
            endog,
            exog,
            offsets,
            compute_offsets_method,
            add_const,
            remove_zero_columns=self.remove_zero_columns,
        )

        self._elbo_criterion_monitor = _ElboCriterionMonitor()
        self._fitted = False

    @classmethod
    @abstractmethod
    def from_formula(
        cls,
        formula: str,
        data: dict[str : Union[torch.Tensor, np.ndarray, pd.DataFrame, pd.Series]],
        *,
        compute_offsets_method: {"zero", "logsum"} = "zero",
    ):
        """
        Create an instance from a formula and data.

        Parameters
        ----------
        formula : str
            The formula.
        data : dict
            The data dictionary. Each value can be either a torch.Tensor,
            `np.ndarray`, `pd.DataFrame` or `pd.Series`. The categorical exogenous
            variables should be 1-dimensional.
        compute_offsets_method : str, optional(keyword-only)
            Method to compute offsets if not provided. Options are:
                - "zero" that will set the offsets to zero.
                - "logsum" that will take the logarithm of the sum (per line) of the counts.
            Overridden (useless) if data["offsets"] is not `None`.
        """
        endog, exog, offsets = _extract_data_from_formula(formula, data)
        return cls(
            endog,
            exog=exog,
            offsets=offsets,
            compute_offsets_method=compute_offsets_method,
            add_const=False,
        )

    def fit(
        self,
        *,
        maxiter: int = 400,
        lr: float = 0.01,
        tol: float = DEFAULT_TOL,
        verbose: bool = False,
    ):
        """
        Fit the model using variational inference.
        The lower the `tol` (tolerance), the more accurate the model.

        Parameters
        ----------
        maxiter : int, optional
            The maximum number of iterations to be done. Defaults to 400.
        lr : float, optional(keyword-only)
            The learning rate. Defaults to 0.01.
        tol : float, optional(keyword-only)
            The tolerance for convergence. Defaults to 1e-6.
        verbose : bool, optional(keyword-only)
            Whether to print training progress. Defaults to `False`.

        Raises
        ------
        ValueError
            If `maxiter` is not an `int`.
        """
        self._fitting_initialization(lr, maxiter)
        iterdone = 0
        stop_condition = False
        pbar = tqdm(desc="Upper bound on the fitting time", total=maxiter)
        while iterdone < maxiter and not stop_condition:
            elbo = self._trainstep()
            self._elbo_criterion_monitor.update_criterion(elbo)
            if abs(self._elbo_criterion_monitor.criterion) < tol:
                stop_condition = True
            if iterdone % 25 == 0:
                self._track_mse()
                if verbose is True:
                    self._print_stats(iterdone, maxiter, tol)
            self._time_recorder.track_running_time()
            pbar.update(1)
            iterdone += 1
        self._print_end_of_fitting_message(stop_condition, tol)
        self._fitted = True
        return self

    def show(self, savefig=False, name_file="", figsize: tuple = (10, 10)):
        """
        Display the model parameters, norm evolution of the parameters and the criterion.

        Parameters
        ----------
        savefig : bool, optional
            If `True`, the figure will be saved to a file. Default is `False`.
        name_file : str, optional
            The name of the file to save the figure. Only used if savefig is `True`.
            Default is an empty string.
        figsize : tuple of two positive floats.
            Size of the figure that will be created. By default (10,10)

        """
        model_viz = self._get_model_viz()
        model_viz.show(savefig=savefig, name_file=name_file, figsize=figsize)

    def _get_model_viz(self):
        return self._ModelViz(self)

    @abstractmethod
    def plot_correlation_circle(self, column_names, column_index=None, title: str = ""):
        """
        Visualizes variables using PCA and plots a correlation circle. If the `endog` has been
        given as a pd.DataFrame, the `column_names` have been stored and may be indicated with the
        `column_names` argument. Else, one should provide the indices of variables.

        Parameters
        ----------
        column_names : List[str]
            A list of variable names to visualize.
            If `column_index` is `None`, the variables plotted are the
            ones in `column_names`. If `column_index` is not `None`,
            this only serves as a legend.
            Check the attribute `column_names_endog`.
        column_index : Optional[List[int]], optional
            A list of indices corresponding to the variables that should be plotted.
            If `None`, the indices are determined based on `column_names_endog`
            given the `column_names`, by default None.
            If not None, should have the same length as `column_names`.
        title : str
            An additional title for the plot.

        Raises
        ------
        ValueError
            If `column_index` is None and `column_names_endog` is not set,
            that has been set if the model has been initialized with a pd.DataFrame as `endog`.
        ValueError
            If the length of `column_index` is different from the length
            of `column_names`.

        """
        column_index = _process_column_index(
            column_names, column_index, self.column_names_endog
        )
        plot_correlation_circle(
            self.transform(), column_names, column_index, title=title
        )

    @abstractmethod
    def biplot(
        self,
        column_names,
        *,
        column_index: np.ndarray = None,
        remove_exog_effect: bool = False,
        colors: np.ndarray = None,
        title: str = "",
    ):  # pylint:disable=too-many-arguments
        """
        Visualizes variables using the correlation circle along with the pca transformed samples.
        If the `endog` has been given as a pd.DataFrame, the `column_names` have been stored and
        may be indicated with the `column_names` argument. Else, one should provide the
        indices of variables.

        Parameters
        ----------
        column_names : List[str]
            A list of variable names to visualize.
            If `column_index` is `None`, the variables plotted
            are the ones in `column_names`. If `column_index`
            is not `None`, this only serves as a legend.
            Check the attribute `column_names_endog`.
        column_index : Optional[List[int]], optional keyword-only
            A list of indices corresponding to the variables that should be plotted.
            If `None`, the indices are determined based on `column_names_endog`
            given the `column_names`, by default `None`.
            If not None, should have the same length as `column_names`.
        remove_exog_effect: bool, optional
            Whether to remove or not the effect of exogenous variables. Default to `False`.
        title : str optional, keyword-only
            An additional title for the plot.
        colors : list, optional, keyword-only
            The labels to color the samples, of size `n_samples`.

        Raises
        ------
        ValueError
            If `column_index` is None and `column_names_endog` is not set,
            that has been set if the model has been initialized with a pd.DataFrame as `endog`.
        ValueError
            If the length of `column_index` is different
            from the length of `column_names`.

        """
        column_index = _process_column_index(
            column_names, column_index, self.column_names_endog
        )
        return _biplot(
            self.transform(remove_exog_effect=remove_exog_effect),
            column_names,
            column_index=column_index,
            colors=colors,
            title=title,
        )

    def _trainstep(self):
        """
        Compute the elbo and do a gradient step.

        Returns
        -------
        torch.Tensor
            The loss value.
        """
        self.optim.zero_grad()
        elbo = self.compute_elbo()
        if torch.isfinite(elbo).item() is False:
            raise ValueError(
                "The ELBO contains non-finite values. Please raise an issue on Github."
            )

        loss = self._compute_loss(elbo)
        loss.backward()
        self.optim.step()
        self._update_closed_forms()
        self._project_parameters()
        return elbo.detach().cpu()

    def _compute_loss(self, elbo):
        return -elbo

    def _initialize_timing(self):
        self._print_beginning_message()
        if self._fitted is True:
            time_to_remove_from_beginning = self._time_recorder.running_times[-1]
            running_times = self._time_recorder.running_times
        else:
            time_to_remove_from_beginning = 0
            running_times = []
        self._time_recorder = _TimeRecorder(
            time_to_remove_from_beginning, running_times
        )

    def _print_beginning_message(self):
        print(f"Fitting a {self._name} model with {self._description}")

    @property
    def _name(self):
        return str(type(self).__name__)

    def _print_end_of_fitting_message(self, stop_condition: bool, tol: float):
        if stop_condition is True:
            print(
                f"Tolerance {tol} reached "
                f"in {self._elbo_criterion_monitor.iteration_number} iterations"
            )
        else:
            print(
                f"Maximum number of iterations ({self._elbo_criterion_monitor.iteration_number})",
                f" reached in {self._time_recorder.running_times[-1]:.1f} seconds.\nLast ",
                f"criterion = {np.round(self._elbo_criterion_monitor.criterion, 8)}",
                f". Required tolerance = {tol}",
            )

    def _init_parameters(self):
        self._print_start_init()
        self._init_model_parameters()
        self._init_latent_parameters()
        self._print_end_init()

    def _print_start_init(self):
        print("Intializing parameters ...")

    def _print_end_init(self):
        print("Initialization finished.")

    @abstractmethod
    def _init_model_parameters(self):
        """Initialization of model parameters."""

    @abstractmethod
    def _init_latent_parameters(self):
        """Initialization of latent parameters."""

    @property
    @abstractmethod
    def _description(self):
        """Description of the model."""

    def _set_requiring_grad_true(self):
        """
        Move parameters to the GPU device if present.
        """
        for parameter in self.list_of_parameters_needing_gradient:
            parameter.requires_grad_(True)

    @property
    @abstractmethod
    def list_of_parameters_needing_gradient(
        self,
    ):
        """
        The list of all the parameters of the model that needs to be updated at each iteration.
        """

    @property
    @abstractmethod
    def dict_model_parameters(self):
        """The parameters of the model."""

    @property
    def model_parameters(self):
        """Alias for dict_model_parameters."""
        return self.dict_model_parameters

    @property
    def _default_dict_model_parameters(self):
        return {"coef": self.coef, "covariance": self.covariance}

    @property
    @abstractmethod
    def dict_latent_parameters(self):
        """The latent parameters of the model."""

    @property
    def latent_parameters(self):
        """Alias for dict_latent_parameters."""
        return self.dict_latent_parameters

    @property
    def _default_dict_latent_parameters(self):
        return {
            "latent_mean": self.latent_mean,
            "latent_sqrt_variance": self.latent_sqrt_variance,
        }

    def _handle_optimizer(self, lr):
        self.optim = torch.optim.Rprop(
            self.list_of_parameters_needing_gradient, lr=lr, step_sizes=(1e-10, 50)
        )

    def _fitting_initialization(self, lr, maxiter):
        if not isinstance(maxiter, int):
            raise ValueError("The argument `maxiter` should be an `int`.")
        self._initialize_timing()
        if self._fitted is False:
            self._init_parameters()
            self._dict_list_mse = {
                name_model: [] for name_model in self.dict_model_parameters.keys()
            }
        self._set_requiring_grad_true()
        self._handle_optimizer(lr)

    @abstractmethod
    def compute_elbo(self):
        """Compute the elbo of the current parameters."""

    def _project_parameters(self):
        """Project some parameters such as probabilities."""

    def _update_closed_forms(self):
        """Update some parameters."""

    def _track_mse(self):
        for name_param, param in self.dict_model_parameters.items():
            mse_param = torch.mean(param**2).item() if param is not None else 0
            self._dict_list_mse[name_param].append(mse_param)

    def _print_stats(self, iterdone, maxiter, tol):
        """
        Print the training statistics.
        """
        print("-------UPDATE-------")
        print("Iteration ", iterdone, "out of ", maxiter, "iterations.")
        msg_criterion = "Current criterion: " + str(
            np.round(self._elbo_criterion_monitor.criterion, 8)
        )
        msg_criterion += ". Stop if lower than " + str(tol)
        print(msg_criterion)
        print("ELBO:", np.round(self._elbo_criterion_monitor.elbo_list[-1], 8))

    @property
    def n_samples(self):
        """Number of samples in the dataset."""
        return self._endog.shape[0]

    @property
    def dim(self):
        """Number of dimensions (i.e. variables) of the dataset."""
        return self._endog.shape[1]

    @property
    def endog(self):
        """
        Property representing the endogenous variables (counts).

        Returns
        -------
        torch.Tensor
            The endogenous variables.
        """
        return self._endog.cpu()

    @property
    def exog(self):
        """
        Property representing the exogenous variables (covariates).

        Returns
        -------
        torch.Tensor or None
            The exogenous variables or `None` if no covariates are given in the model.
        """
        if self._exog is None:
            return None
        return self._exog.cpu()

    @property
    def nb_cov(self) -> int:
        """
        The number of exogenous variables.
        """
        if self.exog is None:
            return 0
        return self.exog.shape[1]

    @property
    def offsets(self):
        """
        Property representing the offsets.

        Returns
        -------
        torch.Tensor
            The offsets.
        """
        return self._offsets.cpu()

    @property
    def latent_mean(self):
        """
        Property representing the latent mean conditionally on the observed counts, i.e. the
        conditional mean of the latent variable of each sample.

        Returns
        -------
        torch.Tensor
            The latent mean.
        """
        return self._latent_mean.detach().cpu()

    @property
    def latent_variance(self):
        """
        Property representing the latent variance conditionally on the observed counts, i.e.
        the conditional variance of the latent variable of each sample.
        """
        return self.latent_sqrt_variance**2

    @property
    def latent_sqrt_variance(self):
        """
        Property representing the latent square root variance conditionally on
        the observed counts, i.e. the square root variance of the latent variable of each sample.

        Returns
        -------
        torch.Tensor
            The square root of the latent variance.
        """
        return self._latent_sqrt_variance.detach().cpu()

    @property
    def coef(self):
        """
        Property representing the regression coefficients of size (`nb_cov`, `dim`).
        If no exogenous (`exog`) is available, returns `None`.

        Returns
        -------
        torch.Tensor or None
            The coefficients or `None` if no coefficients are given in the model.
        """
        return self._coef.detach().cpu() if self._coef is not None else None

    @property
    def covariance(self):
        """
        Property representing the covariance of the model.

        Returns
        -------
        torch.Tensor
            The covariance.
        """
        return self._covariance.detach().cpu()

    @property
    def precision(self):
        """
        Property representing the precision of the model, that is the inverse covariance matrix.

        Returns
        -------
        torch.Tensor
            The precision matrix of size (dim, dim).
        """
        return self._precision.detach().cpu()

    @property
    def _precision(self):
        return torch.linalg.inv(self._covariance)

    @property
    def _marginal_mean(self):
        if self._exog is None:
            return 0
        return self._exog @ self._coef

    @property
    @_shouldbefitted
    def marginal_mean(self):
        """
        The marginal mean of the model, i.e. the mean of the gaussian latent variable.
        """
        if self._exog is None:
            return 0
        return self.exog @ self.coef

    def _pca_projected_latent_variables_with_covariances(
        self, rank=2, remove_exog_effect: bool = False
    ):
        """
        Perform PCA on latent variables and return the
        projected variables along with their covariances in the two dimensional space.

        Parameters
        ----------
        rank : int, optional
            The number of principal components to compute, by default 2.

        Returns
        -------
        tuple
            A tuple containing the projected variables and their covariances.
        """
        if remove_exog_effect is True:
            variables = self.latent_positions
        else:
            variables = self.latent_variables
        pca = PCA(n_components=rank)
        proj_variables = pca.fit_transform(variables)
        sk_components = pca.components_
        covariances = self._get_two_dim_latent_variances(sk_components)
        return proj_variables, covariances

    def projected_latent_variables(self, rank=2, remove_exog_effect: bool = False):
        """
        Perform PCA on latent variables and return the projected variables.

        Parameters
        ----------
        rank : int, optional
            The number of principal components to compute, by default 2.
        remove_exog_effect: bool, optional
            Whether to remove or not the effect of exogenous variables. Default to `False`.
        Returns
        -------
        numpy.ndarray
            The projected variables.
        """
        if remove_exog_effect is True:
            variables = self.latent_positions
        else:
            variables = self.latent_variables
        pca = PCA(n_components=rank)
        proj_variables = pca.fit_transform(variables)
        return torch.from_numpy(proj_variables)

    def transform(self, remove_exog_effect: bool = False):
        """
        Returns the latent variables. Can be seen as a normalization of the counts given.

        Parameters
        ----------
        remove_exog_effect: bool (optional)
            Whether to remove or not the mean induced by the exogenous variables.
            Default is `False`.

        Returns
        -------
        torch.Tensor
            The transformed endogenous variables (latent variables of the model).
        """
        if remove_exog_effect is True:
            return self.latent_positions
        return self.latent_variables

    @property
    @abstractmethod
    def latent_variables(self):
        """
        The (conditional) mean of the latent variables. This is
        the best approximation of latent variables. This variable
        is supposed to be more meaningful than the counts (`endog`).
        """

    @property
    @abstractmethod
    def latent_positions(self):
        """
        The (conditional) mean of the latent variables with the
        effect of covariates removed.
        """

    @abstractmethod
    def viz(
        self,
        *,
        ax=None,
        colors=None,
        show_cov: bool = False,
        remove_exog_effect: bool = False,
    ):
        """
        Visualize the latent variables. One can remove the effect of exogenous variables
        with the `remove_exog_effect` boolean variable.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            The axes on which to plot, by default `None`.
        colors : list, optional
            The labels to color the samples, of size `n_samples`.
        show_cov : bool, optional
            Whether to show covariances, by default False.
        remove_exog_effect: bool, optional
            Whether to remove or not the effect of exogenous variables. Default to `False`.

        """
        if show_cov is True:
            variables, covariances = (
                self._pca_projected_latent_variables_with_covariances(
                    remove_exog_effect=remove_exog_effect
                )
            )
        else:
            variables = self.projected_latent_variables(
                remove_exog_effect=remove_exog_effect
            )
            covariances = None
        _viz_variables(variables, ax=ax, colors=colors, covariances=covariances)

    @abstractmethod
    def _get_two_dim_latent_variances(self, sklearn_components):
        """
        Computes the covariance when the latent variables are
        embedded in a lower dimensional space (often 2) with `sklearn_components`.

        Parameters
        ----------
        components : np.ndarray
            The components of the PCA.
        """

    @_shouldbefitted
    def __repr__(self):
        """
        Generate the string representation of the model.

        Returns
        -------
        str
            The string representation of the object.

        Raises
        ------
        RuntimeError
            If the model is not fitted.
        """

        delimiter = "=" * 70
        add_attributes = self._additional_attributes_list
        add_attributes = ["None"] if len(add_attributes) == 0 else add_attributes
        add_methods = self._additional_methods_list
        add_methods = ["None"] if len(add_methods) == 0 else add_methods
        parts = [
            f"A multivariate {self._name} with {self._description}",
            delimiter,
            _nice_string_of_dict(self._dict_for_printing),
            delimiter,
            "* Useful attributes",
            f"    {' '.join(self._useful_attributes_list)}",
            "* Useful methods",
            f"    {' '.join(self._useful_methods_list)}",
            f"* Additional attributes for {self._name} are:",
            f"    {' '.join(add_attributes)}",
            f"* Additional methods for {self._name} are:",
            f"    {' '.join(add_methods)}",
        ]
        return "\n".join(parts)

    @property
    def _useful_methods_list(self):
        return [
            ".transform()",
            ".show()",
            ".predict()",
            ".sigma()",
            ".projected_latent_variables()",
            ".plot_correlation_circle()",
            ".biplot()",
            ".viz()",
            ".pca_pairplot()",
            ".plot_expected_vs_true()",
        ]

    @property
    def _useful_attributes_list(self):
        return [
            ".latent_variables",
            ".latent_positions",
            ".coef",
            ".covariance",
            ".precision",
            ".model_parameters",
            ".latent_parameters",
            ".optim_details",
        ]

    @property
    @abstractmethod
    def _additional_attributes_list(self):
        """The attributes that are specific to this model."""

    @property
    @abstractmethod
    def _additional_methods_list(self):
        """The methods that are specific to this model."""

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
            "BIC": np.round(self.BIC, 4),
            "AIC": np.round(self.AIC, 4),
            "ICL": np.round(self.ICL, 2),
        }

    @property
    @_shouldbefitted
    def elbo(self):
        """
        Returns the last elbo computed.
        """
        return (self._elbo_criterion_monitor.elbo_list[-1]).item()

    @property
    def loglike(self):
        """
        Alias for elbo.
        """
        return self.elbo

    @property
    def BIC(self):
        """
        Bayesian Information Criterion (BIC) of the model.
        """
        return -self.loglike + self.number_of_parameters / 2 * np.log(self.n_samples)

    @property
    def ICL(self):
        """
        Integrated Completed Likelihood criterion.
        """
        return self.BIC - self.entropy

    @property
    def AIC(self):
        """
        Akaike Information Criterion (AIC).
        """
        return -self.loglike + self.number_of_parameters

    @property
    @abstractmethod
    def number_of_parameters(self):
        """
        Returns the number of parameters of the model.
        """

    @property
    @abstractmethod
    def entropy(self):
        """
        Entropy of the latent variables.
        """

    @_array2tensor
    def predict(self, exog: Union[torch.Tensor, np.ndarray, pd.DataFrame] = None):
        """
        Method for making predictions.

        Parameters
        ----------
        exog : Union[torch.Tensor, np.ndarray, pd.DataFrame, pd.Series], optional
            The `exog`, by default `None`.

        Returns
        -------
        torch.Tensor or None
            The predicted values or `None`.

        Raises
        ------
        AttributeError
            If there is no `exog` in the model but some are provided.
        RuntimeError
            If the shape of the `exog` is incorrect.

        Notes
        -----
        - If `exog` is not provided and there are no exog in the model, `None` is returned.
            If there are `exog` in the model, then the mean `exog @ coef` is returned.
        - If `exog` is provided, it should have the shape `(_, nb_cov)`,
            where `nb_cov` is the number of exog.
        - The predicted values are obtained by multiplying the exog by the coefficients.
        """
        if exog is not None and self.nb_cov == 0:
            msg = "No exog in the model, can't predict with exog"
            raise AttributeError(msg)
        if exog is None:
            if self.exog is None:
                warning_string = (
                    "No exog in the model, can't predict without exog. Returning None."
                )
                warnings.warn(warning_string)
                return None
            raise AttributeError(
                "Exogenous variables are given in the model. Please provide `exog`."
            )
        if exog.shape[-1] != self.nb_cov:
            error_string = f"`exog` has th wrong shape ({exog.shape}). Should"
            error_string += f" be ({self.n_samples, self.nb_cov})."
            raise RuntimeError(error_string)
        return exog @ self._coef

    @property
    @_shouldbefitted
    def optim_details(self):
        """
        Property representing the optimization details.

        Returns
        -------
        dict
            The dictionary of optimization details.
        """
        return {
            "Number of iterations done": len(self._elbo_criterion_monitor.elbo_list),
            "Last criterion:": self._elbo_criterion_monitor.criterion,
        }

    def sigma(self):
        """
        Covariance of the model.
        """
        return self.covariance

    @abstractmethod
    def pca_pairplot(
        self,
        n_components: int = 3,
        colors: np.ndarray = None,
        remove_exog_effect: bool = False,
    ):
        """
        Generates a scatter matrix plot based on Principal
        Component Analysis (PCA) on the latent variables.

        Parameters
        ----------
        n_components (int, optional): The number of components to consider for plotting.
            Defaults to 3. It cannot be greater than 6.
        colors (np.ndarray, optional): An array with one label for each
            sample in the endog property of the object. If `None`, no colors are applied.
            Defaults to `None`.
        remove_exog_effect (bool, optional): Whether to remove the effect of exogenous
            variables. Defaults to `False`.

        Raises
        ------
        ValueError: If the number of components requested is greater
            than the number of variables in the dataset.
        """
        min_n_components = min(6, n_components)
        if remove_exog_effect is True:
            array = self.latent_positions.numpy()
        else:
            array = self.latent_variables.numpy()
        _pca_pairplot(array, min_n_components, colors)

    def plot_expected_vs_true(self, ax=None, colors=None):
        """
        Plot the predicted value of the `endog` against the `endog`.

        Parameters
        ----------
        ax : Optional[matplotlib.axes.Axes], optional
            The matplotlib axis to use. If `None`, the current axis is used, by default `None`.

        colors : Optional[Any], optional
            The labels to color the samples, of size `n_samples`.
            By default `None` (no colors).

        Returns
        -------
        matplotlib.axes.Axes
            The matplotlib axis.

        See also
        --------
        :func:`pyPLNmodels.Pln.pca_pairplot`
        :func:`pyPLNmodels.PlnPCA.pca_pairplot`
        :func:`pyPLNmodels.Pln.biplot`
        :func:`pyPLNmodels.PlnPCA.biplot`
        """
        endog_predictions = self._endog_predictions
        reconstruction_error = torch.mean(
            torch.nan_to_num((self.endog - endog_predictions) ** 2)
        )
        return _plot_expected_vs_true(
            self.endog, endog_predictions, reconstruction_error, ax=ax, colors=colors
        )

    @property
    @abstractmethod
    def _endog_predictions(self):
        """Abstract method the predict the endog variables."""

    @property
    def _latent_dim(self):
        return self.dim

    @latent_sqrt_variance.setter
    @_array2tensor
    def latent_sqrt_variance(
        self, latent_sqrt_variance: Union[torch.Tensor, np.ndarray, pd.DataFrame]
    ):
        """
        Setter for the latent_sqrt_variance.

        Parameters
        ----------
        latent_sqrt_variance : torch.Tensor
            The latent_sqrt_variance to set.

        Raises
        ------
        ValueError
            If the latent_sqrt_variance have an invalid shape ( different than (n_samples,rank)).
        """
        if latent_sqrt_variance.shape != (self.n_samples, self._latent_dim):
            raise ValueError(
                f"Wrong shape. Expected ({self.n_samples, self._latent_dim}), "
                f"got {latent_sqrt_variance.shape}"
            )
        self._latent_sqrt_variance = latent_sqrt_variance

    @latent_mean.setter
    @_array2tensor
    def latent_mean(self, latent_mean: Union[torch.Tensor, np.ndarray, pd.DataFrame]):
        """
        Setter for the latent_mean.

        Parameters
        ----------
        latent_mean : torch.Tensor
            The latent_mean to set.

        Raises
        ------
        ValueError
            If the latent_mean have an invalid shape ( different than (n_samples,rank)).
        """
        if latent_mean.shape != (self.n_samples, self._latent_dim):
            raise ValueError(
                f"Wrong shape. Expected ({self.n_samples, self._latent_dim}), "
                f"got {latent_mean.shape}"
            )
        self._latent_mean = latent_mean
