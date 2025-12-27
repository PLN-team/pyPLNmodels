from typing import Optional, Union
import warnings

import torch
import numpy as np
import pandas as pd


from pyPLNmodels.models.base import BaseModel, DEFAULT_TOL, DEVICE
from pyPLNmodels.calculations.elbos import elbo_pln
from pyPLNmodels.utils._utils import (
    _add_doc,
    _get_two_dim_latent_variances,
    _group_lasso_penalty,
    _lasso_penalty,
    _sparse_group_lasso_penalty,
)
from pyPLNmodels.utils._viz import _viz_network, NetworkModelViz, _build_graph
from pyPLNmodels.utils._data_handler import _extract_data_from_formula, _array2tensor
from pyPLNmodels.calculations._closed_forms import _closed_formula_coef
from pyPLNmodels.calculations.entropies import entropy_gaussian
from pyPLNmodels.calculations._initialization import (
    _init_components_prec,
    _init_latent_pln,
    _init_coef,
)


THRESHOLD = 1e-5


class PlnNetwork(
    BaseModel
):  # pylint:disable=too-many-public-methods,too-many-instance-attributes
    """
    Pln model with regularization on the number of parameters
    of the precision matrix (inverse covariance matrix) representing correlation
    between variables. A penalty can also be imposed on the coef
    to have a sparse regression matrix.

    For more details, see:
    J. Chiquet, S. Robin, M. Mariadassou: "Variational Inference for sparse network
    reconstruction from count data"

    The model is the following:

    .. math::

        \begin{align}
        Z_i &\sim \mathcal{N}(X_i^{\top} B, \Sigma), \quad \|\Sigma^{-1}\|_1 \leq C \\
        Y_{ij} \mid Z_{ij} &\sim \mathcal{P}(\exp(o_{ij} + Z_{ij})).
        \end{align}

    The hyperparameter $\lambda$ (:code:`penalty`) controls the sparsity level. A non-zero entry in $\Sigma^{-1}_{jk}$
    implies a direct dependency between variables $j$ and $k$ in the latent space.

    The model parameters are:

    - :math:`B \in \mathbb{R}^{d \times p}` :code:`coef`: matrix of regression coefficients
    - :math:`\Sigma  \in \mathcal{S}_{+}^{p}` :code:`covariance`: covariance matrix of the latent variables :math:`Z_i`

    Data provided is

    - :math:`Y \in \mathbb{R}^{n \times p}` :code:`endog`: matrix of endogenous variables (counts). Required.
    - :math:`X \in \mathbb{R}^{n \times d}` :code:`exog`: matrix of exogenous variables (covariates). Defaults to vector of 1's.
    - :math:`O  \in \mathbb{R}^{n \times p}` :code:`offsets`: offsets (in log space). Defaults to matrix of 0's.

    The number of covariates is denoted by :math:`d` (:code:`nb_cov`), while :math:`n` denotes the number of samples (:code:`n_samples`)
    and :math:`p` denotes the number of dimensions (:code:`dim`), i.e. features or number of variables.


    Examples
    --------
    >>> from pyPLNmodels import PlnNetwork, load_scrna
    >>> data = load_scrna()
    >>> net = PlnNetwork(data["endog"], penalty = 200)
    >>> net.fit()
    >>> print(net)
    >>> net.viz(colors=data["labels"])
    >>> net.viz_network()

    >>> from pyPLNmodels import PlnNetwork, load_scrna
    >>> data = load_scrna()
    >>> net = PlnNetwork.from_formula("endog ~ 1 + labels", data=data, penalty = 200, penalty_coef = 10) #pylint: disable=line-too-long
    >>> net.fit()
    >>> print(net)
    >>> net.viz(colors=data["labels"])
    >>> net.viz_network()
    """

    penalty: float
    penalty_coef: float
    _components_prec: torch.Tensor
    _mask: torch.Tensor

    _ModelViz = NetworkModelViz
    __coef: torch.Tensor

    @_add_doc(
        BaseModel,
        example="""
            >>> from pyPLNmodels import PlnNetwork, load_scrna
            >>> data = load_scrna()
            >>> net = PlnNetwork.from_formula("endog ~ 1", data, penalty = 200)
            >>> net.fit()
            >>> print(net)
        """,
        returns="""
            PlnNetwork
        """,
        see_also="""
        :func:`pyPLNmodels.PlnNetwork.from_formula`
        :class:`pyPLNmodels.PlnPCA`
        :class:`pyPLNmodels.PlnMixture`
        :class:`pyPLNmodels.Pln`
        """,
        params="""
        penalty: float
            - The penalty parameter for the precision matrix. The larger the penalty, the larger the
               sparsity of the precision matrix.
        penalty_coef: float
            - The penalty parameter for the coef matrix. The larger the penalty, the larger the
               sparsity of the coef matrix. Default is 0 (no penalty).
        penalty_coef_type: optional ("lasso", "group_lasso", "sparse_group_lasso")
            - The penalty type for the `coef`. Useless if `penalty_coef` is 0. Can be either:
                - "lasso": Enforces sparsity on each coefficient independently, encouraging
                   many coefficients to be exactly zero.
                - "group_lasso": Enforces group sparsity, encouraging entire groups of
                   coefficients (e.g., corresponding to a covariate) to be zero.
                - "sparse_group_lasso": Combines the effects of "lasso" and
                   "group_lasso", enforcing both individual and group sparsity.
        """,
    )
    def __init__(
        self,
        endog: Union[torch.Tensor, np.ndarray, pd.DataFrame],
        penalty: float,
        *,
        exog: Optional[Union[torch.Tensor, np.ndarray, pd.DataFrame, pd.Series]] = None,
        offsets: Optional[Union[torch.Tensor, np.ndarray, pd.DataFrame]] = None,
        compute_offsets_method: {"zero", "logsum"} = "zero",
        add_const: bool = True,
        penalty_coef: float = 0,
        penalty_coef_type: {"lasso", "group_lasso", "sparse_group_lasso"} = "lasso",
    ):  # pylint: disable=too-many-arguments
        self._set_penalty(penalty)
        self._set_penalty_coef(penalty_coef)
        self._set_penalty_coef_type(penalty_coef_type)
        super().__init__(
            endog,
            exog=exog,
            offsets=offsets,
            compute_offsets_method=compute_offsets_method,
            add_const=add_const,
        )
        self._mask = torch.ones((self.dim, self.dim)).to(
            self._endog.device
        ) - torch.eye(self.dim, device=self._endog.device)

    @classmethod
    @_add_doc(
        BaseModel,
        example="""
            >>> from pyPLNmodels import PlnNetwork, load_scrna
            >>> data = load_scrna()
            >>> net = PlnNetwork.from_formula("endog ~ 1", data=data, penalty=1)
        """,
        returns="""
            PlnNetwork
        """,
        see_also="""
        :class:`pyPLNmodels.PlnNetwork`
        :func:`pyPLNmodels.PlnNetwork.__init__`
        """,
        params="""
        penalty: float
            - The penalty parameter for the precision matrix. The larger the penalty, the larger the
               sparsity of the precision matrix.
        penalty_coef: float
            - The penalty parameter for the coef matrix. The larger the penalty, the larger the
               sparsity of the coef matrix. Default is 0 (no penalty).
        penalty_coef_type: optional ("lasso", "group_lasso", "sparse_group_lasso")
            - The penalty type for the `coef`. Useless if `penalty_coef` is 0. Can be either:
                - "lasso": Enforces sparsity on each coefficient independently, encouraging
                   many coefficients to be exactly zero.
                - "group_lasso": Enforces group sparsity, encouraging entire groups of
                   coefficients (e.g., corresponding to a covariate) to be zero.
                - "sparse_group_lasso": Combines the effects of "lasso" and
                   "group_lasso", enforcing both individual and group sparsity.
        """,
    )
    def from_formula(
        cls,
        formula: str,
        data: dict[str : Union[torch.Tensor, np.ndarray, pd.DataFrame, pd.Series]],
        *,
        penalty: float,
        compute_offsets_method: {"zero", "logsum"} = "zero",
        penalty_coef: float = 0,
        penalty_coef_type: {"lasso", "group_lasso", "sparse_group_lasso"} = "lasso",
    ):  # pylint: disable=arguments-differ,too-many-arguments
        endog, exog, offsets = _extract_data_from_formula(formula, data)
        return cls(
            endog,
            penalty=penalty,
            exog=exog,
            offsets=offsets,
            compute_offsets_method=compute_offsets_method,
            add_const=False,
            penalty_coef=penalty_coef,
            penalty_coef_type=penalty_coef_type,
        )

    @_add_doc(
        BaseModel,
        example="""
        >>> from pyPLNmodels import PlnNetwork, load_scrna
        >>> data = load_scrna()
        >>> net = PlnNetwork.from_formula("endog ~ 1", data, penalty = 200)
        >>> net.fit()
        >>> print(net)

        >>> from pyPLNmodels import PlnNetwork, load_scrna
        >>> data = load_scrna()
        >>> net = PlnNetwork.from_formula("endog ~ 1", data, penalty = 200)
        >>> net.fit()
        >>> print(net)
        >>> net.fit(penalty = 100)
        >>> print(net)
        """,
        params="""
        penalty: float
            - The penalty parameter for the precision matrix. The larger the penalty, the larger the
               sparsity of the precision matrix.
        penalty_coef: float
            - The penalty parameter for the coef matrix. The larger the penalty, the larger the
               sparsity of the coef matrix. Default is 0 (no penalty).
        penalty_coef_type: optional ("lasso", "group_lasso", "sparse_group_lasso")
            - The penalty type for the `coef`. Useless if `penalty_coef` is 0. Can be either:
                - "lasso": Enforces sparsity on each coefficient independently, encouraging
                   many coefficients to be exactly zero.
                - "group_lasso": Enforces group sparsity, encouraging entire groups of
                   coefficients (e.g., corresponding to a covariate) to be zero.
                - "sparse_group_lasso": Combines the effects of "lasso" and
                   "group_lasso", enforcing both individual and group sparsity.
        """,
        returns="""
        PlnNetwork object
        """,
    )
    def fit(
        self,
        *,
        maxiter: int = 400,
        lr: float = 0.01,
        tol: float = DEFAULT_TOL,
        verbose: bool = False,
        penalty: float = None,
        penalty_coef: float = None,
        penalty_coef_type: {"lasso", "group_lasso", "sparse_group_lasso"} = None,
    ):  # pylint: disable = too-many-arguments
        if penalty is not None:
            self._set_penalty(penalty)
            print(f"Changing `penalty` from {self.penalty} to : ", penalty, ".")
        if penalty_coef is not None:
            if self.penalty_coef == 0 and penalty_coef > 0:
                if self.nb_cov == 0:
                    self.__coef = None
                else:
                    self.__coef = (
                        _closed_formula_coef(self._exog, self._latent_mean)
                        .detach()
                        .requires_grad_(True)
                    )
            self._set_penalty_coef(penalty_coef)
            print(
                f"Changing `penalty_coef` from {self.penalty_coef} to : ",
                penalty_coef,
                ".",
            )
        if penalty_coef_type is not None:
            self._set_penalty_coef_type(penalty_coef_type)
            print(
                f"Changing `penalty_coef_type` from {self.penalty} to : ", penalty, "."
            )
        return super().fit(maxiter=maxiter, lr=lr, tol=tol, verbose=verbose)

    def _set_penalty(self, penalty):
        if not isinstance(penalty, (int, float)):
            raise ValueError(
                f"`penalty` must be a float, got {type(penalty).__name__}."
            )
        if penalty < 0:
            raise ValueError(f"`penalty` should be positive. Got {penalty}")
        self.penalty = penalty

    def _set_penalty_coef(self, penalty_coef):
        if not isinstance(penalty_coef, (int, float)):
            raise ValueError(
                f"`penalty_coef` must be a float, got {type(penalty_coef).__name__}."
            )
        if penalty_coef < 0:
            raise ValueError(f"`penalty_coef` should be positive. Got {penalty_coef}")
        self.penalty_coef = penalty_coef

    def _set_penalty_coef_type(self, penalty_coef_type):
        if penalty_coef_type not in ["lasso", "group_lasso", "sparse_group_lasso"]:
            msg = "`penalty_coef_type` should be either 'lasso',"
            msg += f"'group_lasso', or 'sparse_group_lasso', got {penalty_coef_type}"
            raise ValueError(msg)
        self.penalty_coef_type = penalty_coef_type
        if self.penalty_coef == 0 and penalty_coef_type in [
            "group_lasso",
            "sparse_group_lasso",
        ]:
            raise ValueError("`penalty_coef` is 0, no penalty can be imposed.")

    @_add_doc(
        BaseModel,
        example="""
        >>> from pyPLNmodels import PlnNetwork, load_scrna
        >>> data = load_scrna()
        >>> net = PlnNetwork.from_formula("endog ~ 1", data, penalty = 200)
        >>> net.fit()
        >>> elbo = net.compute_elbo()
        >>> print('ELBO:', elbo)
        """,
    )
    def compute_elbo(self):
        precision = self._precision
        elbo_no_penalty = elbo_pln(
            endog=self._endog,
            marginal_mean=self._marginal_mean,
            offsets=self._offsets,
            latent_mean=self._latent_mean,
            latent_sqrt_variance=self._latent_sqrt_variance,
            precision=precision,
        )
        elbo_penalty = elbo_no_penalty - self.penalty * self._l1_penalty_precision(
            precision
        )
        if self.nb_cov == 0:
            return elbo_penalty
        if self.penalty_coef > 0:
            penalty_coef_value = self._get_penalty_coef_value()
            elbo_penalty -= self.penalty_coef * penalty_coef_value
        return elbo_penalty

    def _l1_penalty_precision(self, precision):
        return torch.norm(precision * self._mask, p=1)

    def _get_penalty_coef_value(self):
        if self.penalty_coef_type == "lasso":
            return _lasso_penalty(self.__coef)
        if self.penalty_coef_type == "group_lasso":
            return _group_lasso_penalty(self.__coef)
        return _sparse_group_lasso_penalty(self.__coef)

    @property
    def _precision(self):
        return self._components_prec @ (self._components_prec.T)

    @property
    def _covariance(self):
        return torch.linalg.inv(self._precision)

    @property
    @_add_doc(BaseModel)
    def number_of_parameters(self):
        nb_param = (
            self.dim * (self.dim + 2 * self.nb_cov + 1) / 2
            - self.nb_zeros_precision.detach().item()
        )
        return nb_param

    @property
    def nb_zeros_precision(self):
        """Number of zeros in the precision matrix without (on the lower diagonal)."""
        return torch.sum((torch.abs(self.precision) < THRESHOLD).float()) / 2

    def _init_model_parameters(self):
        if not hasattr(self, "_components_prec"):
            self._components_prec = _init_components_prec(self._endog)
        if self.penalty_coef > 0:
            # if penalty coef is positive, then no closed forms for the coef
            if not hasattr(self, "__coef"):
                coef = _init_coef(
                    endog=self._endog, exog=self._exog, offsets=self._offsets
                )
                if coef is not None:
                    self.__coef = coef.detach().to(DEVICE)
                else:
                    self.__coef = None
                    warnings.warn(
                        "No covariates in the model, `penalty_coef` is useless."
                    )

    @property
    @_add_doc(BaseModel)
    def list_of_parameters_needing_gradient(self):
        list_params = [
            self._latent_mean,
            self._latent_sqrt_variance,
            self._components_prec,
        ]
        if self.penalty_coef > 0 and self.__coef is not None:
            list_params.append(self.__coef)
        return list_params

    def viz_network(self, ax=None):
        """
        Visualize the network infered by the model, i.e. the correlation between variables.
        The network is created by establishing edges between variables that have
        a non-zero correlation.

        >>> from pyPLNmodels import PlnNetwork, load_scrna
        >>> data = load_scrna()
        >>> net = PlnNetwork(data["endog"], penalty = 200)
        >>> net.fit()
        >>> print(net)
        >>> net.viz(colors=data["labels"])
        >>> net.viz_network()
        """
        _viz_network(
            self.precision,
            ax=ax,
            node_labels=self.column_names_endog,
        )

    @property
    def precision(self):
        """
        Property representing the precision of the model, that is the inverse covariance matrix.

        Returns
        -------
        torch.Tensor
            The precision matrix of size (dim, dim).
        """
        precision = self._precision
        return (precision * (torch.abs(precision) > THRESHOLD)).detach().cpu()

    @property
    def network(self):
        """
        Return the network infered by the model, i.e. variables with non zero correlations.

        Examples

        >>> from pyPLNmodels import PlnNetwork, load_scrna
        >>> data = load_scrna()
        >>> net = PlnNetwork(data["endog"], penalty = 200)
        >>> net.fit()
        >>> print(net)
        >>> print(net.network)
        """
        _, connections = _build_graph(
            self.precision,
            node_labels=self.column_names_endog,
        )
        return connections

    @property
    @_add_doc(BaseModel)
    def dict_model_parameters(self):
        return self._default_dict_model_parameters

    @property
    @_add_doc(BaseModel)
    def dict_latent_parameters(self):
        return self._default_dict_latent_parameters

    @property
    def _coef(self):
        if self.penalty_coef > 0:
            return self.__coef
        return _closed_formula_coef(self._exog, self._latent_mean)

    @property
    def _additional_methods_list(self):
        return [".viz_network()"]

    @property
    def _additional_attributes_list(self):
        return [".network", ".nb_zeros_precision"]

    def _get_two_dim_latent_variances(self, sklearn_components):
        return _get_two_dim_latent_variances(
            sklearn_components, self.latent_sqrt_variance
        )

    @property
    @_add_doc(
        BaseModel,
        example="""
        >>> from pyPLNmodels import PlnNetwork, load_scrna
        >>> data = load_scrna()
        >>> net = PlnNetwork.from_formula("endog ~ 1", data, penalty = 200)
        >>> net.fit()
        >>> print(net.latent_variables.shape)
        >>> net.viz() # Visualize the latent variables
        """,
        see_also="""
        :func:`pyPLNmodels.PlnNetwork.latent_positions`
        """,
    )
    def latent_variables(self):
        return self.latent_mean

    @property
    @_add_doc(
        BaseModel,
        example="""
        >>> from pyPLNmodels import PlnNetwork, load_scrna
        >>> data = load_scrna()
        >>> net = PlnNetwork.from_formula("endog ~ 1", data, penalty = 200)
        >>> net.fit()
        >>> print(net.latent_positions.shape)
        >>> net.viz(remove_exog_effect=True) # Visualize the latent positions
        """,
        see_also="""
        :func:`pyPLNmodels.PlnNetwork.latent_variables`
        """,
    )
    def latent_positions(self):
        return self.latent_mean - self.marginal_mean

    @_add_doc(
        BaseModel,
        example="""
        >>> from pyPLNmodels import PlnNetwork, load_scrna
        >>> data = load_scrna()
        >>> net = PlnNetwork.from_formula("endog ~ 1", data=data, penalty = 200)
        >>> net.fit()
        >>> net.plot_correlation_circle(column_names=["MALAT1", "ACTB"])
        >>> net.plot_correlation_circle(column_names=["A", "B"], column_index=[0, 4])
        """,
    )
    def plot_correlation_circle(self, column_names, column_index=None, title: str = ""):
        super().plot_correlation_circle(
            column_names=column_names,
            column_index=column_index,
            title=title,
        )

    @_add_doc(
        BaseModel,
        example="""
        >>> from pyPLNmodels import PlnNetwork, load_scrna
        >>> data = load_scrna()
        >>> net = PlnNetwork.from_formula("endog ~ 1", data=data, penalty=1)
        >>> net.fit()
        >>> net.biplot(column_names=["MALAT1", "ACTB"])
        >>> net.biplot(column_names=["A", "B"], column_index=[0, 4], colors=data["labels"])
        """,
    )
    def biplot(
        self,
        column_names,
        *,
        column_index: np.ndarray = None,
        colors: np.ndarray = None,
        title: str = "",
        remove_exog_effect: bool = False,
    ):  # pylint:disable=too-many-arguments
        super().biplot(
            column_names=column_names,
            column_index=column_index,
            colors=colors,
            title=title,
            remove_exog_effect=remove_exog_effect,
        )

    @_add_doc(
        BaseModel,
        example="""
        >>> from pyPLNmodels import PlnNetwork, load_scrna
        >>> data = load_scrna()
        >>> net = PlnNetwork.from_formula("endog ~ 1", data=data, penalty = 200)
        >>> net.fit()
        >>> net.pca_pairplot(n_components=5)
        >>> net.pca_pairplot(n_components=5, colors=data["labels"])
        """,
    )
    def pca_pairplot(
        self, n_components: bool = 3, colors=None, remove_exog_effect: bool = False
    ):
        super().pca_pairplot(
            n_components=n_components,
            colors=colors,
            remove_exog_effect=remove_exog_effect,
        )

    @_add_doc(
        BaseModel,
        example="""
              >>> from pyPLNmodels import PlnNetwork, load_scrna
              >>> data = load_scrna()
              >>> net = PlnNetwork.from_formula("endog ~ 1", data=data, penalty = 200)
              >>> net.fit()
              >>> transformed_endog = net.transform()
              >>> print(transformed_endog.shape)
              >>> net.viz()
              >>> transformed_no_exog = net.transform(remove_exog_effect=True)
              """,
    )
    def transform(self, remove_exog_effect: bool = False):
        return super().transform(remove_exog_effect=remove_exog_effect)

    @property
    def _endog_predictions(self):
        return torch.exp(
            self.offsets + self.latent_mean + 1 / 2 * self.latent_sqrt_variance**2
        )

    @_add_doc(
        BaseModel,
        example="""
            >>> from pyPLNmodels import PlnNetwork, load_scrna
            >>> data = load_scrna()
            >>> net = PlnNetwork(data["endog"], penalty=1)
            >>> net.fit()
            >>> net.plot_expected_vs_true()
            >>> net.plot_expected_vs_true(colors=data["labels"])
            """,
    )
    def plot_expected_vs_true(self, ax=None, colors=None):
        super().plot_expected_vs_true(ax=ax, colors=colors)

    @_add_doc(
        BaseModel,
        example="""
            >>> from pyPLNmodels import PlnNetwork, load_scrna
            >>> data = load_scrna()
            >>> net = PlnNetwork.from_formula("endog ~ 1 + labels", data=data, penalty=1)
            >>> net.fit()
            >>> net.viz()
            >>> net.viz(colors=data["labels"])
            >>> net.viz(show_cov=True)
            >>> net.viz(remove_exog_effect=True, colors=data["labels"])
            """,
    )
    def viz(
        self,
        *,
        ax=None,
        colors=None,
        show_cov: bool = False,
        remove_exog_effect: bool = False,
    ):
        super().viz(
            ax=ax,
            colors=colors,
            show_cov=show_cov,
            remove_exog_effect=remove_exog_effect,
        )

    @property
    def _dict_for_printing(self):
        orig_dict = super()._dict_for_printing
        orig_dict["Nb edges"] = int(self.nb_links)
        return orig_dict

    @property
    def nb_links(self):
        """Returns the number of links in the graph."""
        return self.dim * (self.dim - 1) / 2 - self.nb_zeros_precision

    @property
    def _description(self):
        descr = f"penalty {self.penalty} on the precision matrix and {self.penalty_coef_type}"
        descr += f"  penalty {self.penalty_coef} on the regression coefficients."
        return descr

    def _init_latent_parameters(self):
        if not hasattr(self, "_latent_mean") or not hasattr(
            self, "_latent_sqrt_variance"
        ):
            self._latent_mean, self._latent_sqrt_variance = _init_latent_pln(
                self._endog
            )

    @property
    def components_prec(self):
        """
        Returns an unorthogonal square root of the precision matrix.

        Returns
        -------
        torch.Tensor
            The components of the precision with size (dim, dim)
        """
        return self._components_prec.detach().cpu()

    @components_prec.setter
    @_array2tensor
    def components_prec(
        self, components_prec: Union[torch.Tensor, np.ndarray, pd.DataFrame]
    ):
        """
        Setter for the components_prec, that is an unorthogonal square root of the precision matrix.

        Parameters
        ----------
        components_prec : torch.Tensor
            The components_prec to set.

        Raises
        ------
        ValueError
            If the components_prec have an invalid shape.
        """
        if components_prec.shape != (self.dim, self.dim):
            raise ValueError(
                f"Wrong shape. Expected ({self.dim, self.dim}), got {components_prec.shape}"
            )
        self._components_prec = components_prec

    @property
    @_add_doc(BaseModel)
    def entropy(self):
        return entropy_gaussian(self._latent_sqrt_variance**2).detach().cpu().item()

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
        if self._coef is not None:
            coef_thresholded = (torch.abs(self._coef) > THRESHOLD) * self._coef
            return coef_thresholded.detach().cpu()
        return None
