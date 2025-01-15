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
        tol: float = 1e-6,
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
                torch.ones(self.n_samples, self.dim)
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
            If None, will display the plot.
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

    def _fit_ols(self, variable_number):
        latent_variables, _ = self.latent_variables
        return OLS(
            latent_variables.numpy()[:, variable_number],
            self.exog.numpy(),
            hasconst=True,
        ).fit()


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
