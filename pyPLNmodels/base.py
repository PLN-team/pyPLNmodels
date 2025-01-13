from abc import ABC, abstractmethod
from typing import Union, Optional

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.decomposition import PCA

from pyPLNmodels._data_handler import _handle_data, _extract_data_from_formula
from pyPLNmodels._criterion import _ElboCriterionMonitor
from pyPLNmodels._utils import _TimeRecorder, _nice_string_of_dict
from pyPLNmodels._viz import _viz_variables


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
            Overriden (useless) if `offsets` is not None.
        add_const: bool, optional(keyword-only)
            Whether to add a column of one in the `exog`. Defaults to `True`.
        """

        (
            self._endog,
            self._exog,
            self._offsets,
            self._column_names_endog,
            self._column_names_exog,
        ) = _handle_data(
            endog,
            exog,
            offsets,
            compute_offsets_method,
            add_const,
        )

        self._elbo_criterion_monitor = _ElboCriterionMonitor()
        self._fitted = False

    @classmethod
    def from_formula(
        cls,
        formula: str,
        data: dict[str : Union[torch.Tensor, np.ndarray, pd.DataFrame, pd.Series]],
        *,
        compute_offsets_method: {"zero", "logsum"} = "zero",
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
        compute_offsets_method : str, optional(keyword-only)
            Method to compute offsets if not provided. Options are:
                - "zero" that will set the offsets to zero.
                - "logsum" that will take the logarithm of the sum (per line) of the counts.
            Overriden (useless) if data["offsets"] is not None.
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
        tol: float = 1e-6,
        verbose: bool = False,
    ):
        """
        Fit the model using variational inference.
        The lower the `tol`(erance), the more accurate the model.

        Parameters
        ----------
        maxiter : int, optional
            The maximum number of iterations to be done. Defaults to 400.
        lr : float, optional(keyword-only)
            The learning rate. Defaults to 0.01.
        tol : float, optional(keyword-only)
            The tolerance for convergence. Defaults to 1e-6.
        verbose : bool, optional(keyword-only)
            Whether to print training progress.  Defaults to False.
        Raises
        ------
        ValueError
            If 'maxiter' is not an int.
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
            pbar.update(1)
            iterdone += 1
        self._print_end_of_fitting_message(stop_condition, tol)
        self._fitted = True

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
        if torch.sum(torch.isnan(elbo)):
            raise ValueError("The ELBO contains nan values.")
        (-elbo).backward()
        self.optim.step()
        self._project_parameters()
        return elbo.detach().cpu()

    def _initialize_timing(self):
        self._print_beginning_message()
        if self._fitted is True:
            time_to_remove_from_beginning = self.running_times[-1]
        else:
            time_to_remove_from_beginning = 0
        self._time_recorder = _TimeRecorder(time_to_remove_from_beginning)

    def _init_parameters(self):
        pass

    def _print_beginning_message(self):
        print(f"Fitting a {self._name} model with {self._description}")

    def _name(self):
        return type(self).__name__

    def _print_end_of_fitting_message(self, stop_condition: bool, tol: float):
        if stop_condition is True:
            print(
                f"Tolerance {tol} reached "
                f"in {self._elbo_criterion_monitor.iteration_number} iterations"
            )
        else:
            print(
                "Maximum number of iterations reached : ",
                self._elbo_criterion_monitor.iteration_number,
                ".\nLast criterion = ",
                np.round(self._elbo_criterion_monitor.criterion.item(), 8),
                f". Required tolerance = {tol}",
            )

    def _init_parameters(self):
        print("Intializing parameters ...")
        self._init_model_parameters()
        self._init_latent_parameters()
        print("Initialization finished.")

    @abstractmethod
    def _init_model_parameters(self):
        pass

    @abstractmethod
    def _init_latent_parameters(self):
        pass

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
    ):  # pylint: disable=missing-function-docstring
        pass

    @property
    @abstractmethod
    def dict_model_parameters(self):
        """The parameters of the model."""

    @property
    def _default_dict_model_parameters(self):
        return {"coef": self.coef, "covariance": self.covariance}

    @property
    @abstractmethod
    def dict_latent_parameters(self):
        """The latent parameters of the model."""

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
            The exogenous variables or None if no covariates are given in the model.
        """
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
        the observed counts, i.e. the square root variance of the
        latent variable of each sample.

        Returns
        -------
        torch.Tensor
            The square root of the latent variance.
        """
        return self._latent_sqrt_variance.detach().cpu()

    @property
    def coef(self):
        """
        Property representing the regression coefficients of size (nb_cov, dim).
        If no exogenous (`exog`) is available, returns None.

        Returns
        -------
        torch.Tensor or None
            The coefficients or None if no coefficients are given in the model.
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
    def _marginal_mean(self):
        if self._exog is None:
            return 0
        return self._exog @ self._coef

    @property
    def marginal_mean(self):
        """
        The marginal mean of the model, i.e. the mean of the gaussian latent variable.
        """
        if self._exog is None:
            return 0
        return self.exog @ self.coef

    def _pca_projected_latent_variables_with_covariances(self, rank=2):
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
        variables = self.transform()
        pca = PCA(n_components=rank)
        proj_variables = pca.fit_transform(variables)
        sk_components = pca.components_
        covariances = self._get_two_dim_covariances(sk_components)
        return proj_variables, covariances

    def _pca_projected_latent_variables(self, rank=2):
        """
        Perform PCA on latent variables and return the projected variables.

        Parameters
        ----------
        rank : int, optional
            The number of principal components to compute, by default 2.

        Returns
        -------
        numpy.ndarray
            The projected variables.
        """
        variables = self.transform()
        pca = PCA(n_components=rank)
        proj_variables = pca.fit_transform(variables)
        return proj_variables

    def transform(self):
        """
        Returns the latent variables. Can be seen as a
        normalization of the counts given.
        """
        return self.latent_variables

    @property
    @abstractmethod
    def latent_variables(self):
        """
        The (conditional) mean of the latent variables is
        the best approximation of latent variables.
        """

    def viz(self, *, ax=None, colors=None, show_cov: bool = False):
        """
        Visualize the latent variables.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            The axes on which to plot, by default None.
        colors : list, optional
            The colors to use for the plot, by default None.
        show_cov : bool, optional
            Whether to show covariances, by default False.
        """
        if show_cov is True:
            variables, covariances = (
                self._pca_projected_latent_variables_with_covariances()
            )
        else:
            variables = self.pca_projected_latent_variables()
            covariances = None
        _viz_variables(variables, ax=ax, colors=colors, covariances=covariances)

    @abstractmethod
    def _get_two_dim_covariances(self, sklearn_components):
        """
        Computes the covariance when the latent variables are
        embedded in a lower dimensional space (often 2) with the compoents.

        Parameters
        ----------
        components : np.ndarray
            The components of the PCA.
        """

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
        if not self._fitted:
            raise RuntimeError("Please fit the model before printing it.")

        delimiter = "=" * 70
        parts = [
            f"A multivariate Poisson Lognormal with {self._description}",
            delimiter,
            _nice_string_of_dict(self._dict_for_printing),
            delimiter,
            "* Useful properties",
            f"    {self._useful_properties_string}",
            "* Useful methods",
            f"    {self._useful_methods_strings}",
            f"* Additional properties for {self._name} are:",
            f"    {self._additional_properties_string}",
            f"* Additional methods for {self._name} are:",
            f"    {self._additional_methods_string}",
        ]
        return "\n".join(parts)

    @property
    def _useful_methods_strings(self):
        """
        Useful methods of the model.
        """
        return (
            ".show(), "
            ".transform(), "
            ".sigma(), "
            ".predict(), "
            ".pca_projected_latent_variables(), "
            ".plot_pca_correlation_circle(), "
            ".viz(), "
            ".pca_pairplot(), "
            ".plot_expected_vs_true()"
        )

    @property
    def _useful_properties_string(self):
        """
        Useful properties of the model.
        """
        return ".latent_variables, .model_parameters, .latent_parameters, .optim_parameters"

    @property
    @abstractmethod
    def _additional_properties_string(self):
        """The properties that are specific to this model."""

    @property
    @abstractmethod
    def _additional_methods_string(self):
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
        }

    @property
    def elbo(self):
        """
        Returns the last elbo computed.
        """
        return self._elbo_criterion_monitor.elbo_list[-1]

    @property
    def loglike(self):
        """
        Returns the log likelihood of the model, that is n_samples*elbo.
        """
        return self.n_samples * self.elbo

    @property
    def BIC(self):
        """
        Bayesian Information Criterion (BIC) of the model.
        """
        return -self.loglike + self.number_of_parameters / 2 * np.log(self.n_samples)

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
