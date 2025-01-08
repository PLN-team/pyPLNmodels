from abc import ABC, abstractmethod
from typing import Union, Optional


import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from pyPLNmodels._data_handler import _handle_data, _extract_data_from_formula
from pyPLNmodels._criterion import _ElboCriterionMonitor
from pyPLNmodels._utils import _TimeRecorder


class BaseModel(ABC):  # pylint: disable=too-many-instance-attributes
    """
    Abstract base class for all the PLN based models that will be derived.
    """

    _time_recorder: _TimeRecorder
    optim: torch.optim.Optimizer
    _dict_list_mse: dict

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
        pbar = tqdm(desc="Fitting time will not exceed:", total=maxiter)
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
        self._update_closed_forms()
        self._project_parameters()
        return elbo

    def _initialize_timing(self):
        self._print_beginning_message()
        if self._fitted is True:
            time_to_remove_from_beginning = self.running_times[-1]
        else:
            time_to_remove_from_beginning = 0
        self._time_recorder = _TimeRecorder(time_to_remove_from_beginning)

    def _init_parameters(self):
        pass

    def viz(self):
        """
        Visualizes the latent variables.
        """
        print("Visualizing...")

    def _print_beginning_message(self):
        print(f"Fitting a {type(self).__name__} model with {self._description}")

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

    @abstractmethod
    def list_of_parameters_needing_gradient(
        self,
    ):  # pylint: disable=missing-function-docstring
        pass

    def _handle_optimizer(self, lr):
        self.optim = torch.optim.Rprop(self._list_of_parameters_needing_gradient, lr=lr)

    def _fitting_initialization(self, lr, maxiter):
        if not isinstance(maxiter, int):
            raise ValueError("The argument `maxiter` should be an `int`.")
        self._initialize_timing()
        if self._fitted is False:
            self._init_parameters()
            self._dict_list_mse = {
                name_model: [] for name_model in self.model_parameters.keys()
            }
        self._set_requiring_grad_true()
        self._handle_optimizer(lr)

    @abstractmethod
    def compute_elbo(self):
        """Compute the elbo of the current parameters."""

    def _update_closed_forms(self):
        """Update the closed forms, such as covariance or coef, when closed forms are available."""
        self._update_closed_forms()

    def _project_parameters(self):
        """Project some parameters such as probabilities."""

    def _track_mse(self):
        for name_param, param in self.model_parameters.items():
            mse_param = torch.mean(param**2).detach().item()
            self._dict_list_mse[name_param].append(mse_param)

    def _print_stats(self, iterdone, maxiter, tol):
        """
        Print the training statistics.
        """
        print("-------UPDATE-------")
        print("Iteration ", iterdone, "out of ", maxiter, "iterations.")
        msg_criterion = "Current criterion: " + str(
            np.round(self._elbo_criterion_monitor.criterion, 5)
        )
        msg_criterion += ". Stop if lower than " + str(tol)
        print(msg_criterion)
        print("ELBO:", np.round(self._elbo_criterion_monitor.elbo_list[-1], 6))
