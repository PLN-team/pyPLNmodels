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
from statsmodels.api import OLS

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

    def viz_positions(self, *, ax=None, colors=None, show_cov: bool = False):
        variables = self.latent_positions
        return self._viz_variables(variables, ax=ax, colors=colors, show_cov=show_cov)

    @property
    def latent_positions(self):
        return self.transform() - self.mean_gaussian

    def summary(
        self,
        variable_number,
        yname: str = None,
        xname: list[str] = None,
        title: str = None,
        alpha: float = 0.05,
        slim: bool = False,
    ):
        """
        Summary from statsmodels on the latent variables.

        parameters
        ----------
        yname : str, Optional
            Name of endogenous (response) variable. The Default is y.
        xname : str, Optional
            Names for the exogenous variables. Default is var_## for ##
            in the number of regressors.
            Must match the number of parameters in the model.

        title : str, Optional
            Title for the top table. If not None, then this replaces the default title.
        alpha : float, optional
            The significance level for the confidence intervals.
        slim: bool, Optional
            Flag indicating to produce reduced set or diagnostic information. Default is False.
        """
        if self.exog is None:
            print("No exog in the model, can not perform a summary.")
        else:
            ols = self._fit_ols(variable_number)
            return ols.summary(
                yname=yname, xname=xname, title=title, alpha=alpha, slim=slim
            )

    def _fit_ols(self, variable_number):
        return OLS(
            self.latent_variables.numpy()[:, variable_number],
            self.exog.numpy(),
            hasconst=True,
        ).fit()

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
    def _directory_name(self):
        """
        Property representing the directory name.

        Returns
        -------
        str
            The directory name.
        """
        return f"{self._NAME}_nbcov_{self.nb_cov}_dim_{self.dim}"


class PlnPCAcollection:

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


class PlnPCA(_model):

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
    def ortho_components(self):
        """
        Orthogonal components of the model.
        """
        return torch.linalg.qr(self._components, "reduced")[0]


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

    def _fit_ols(self, variable_number):
        latent_variables, _ = self.latent_variables
        return OLS(
            latent_variables.numpy()[:, variable_number],
            self.exog.numpy(),
            hasconst=True,
        ).fit()
