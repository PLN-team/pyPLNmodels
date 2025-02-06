from typing import Optional, Union

import torch
import numpy as np
import pandas as pd


from pyPLNmodels.base import BaseModel, DEFAULT_TOL
from pyPLNmodels.elbos import elbo_pln
from pyPLNmodels._utils import _add_doc, _two_dim_covariances
from pyPLNmodels._closed_forms import _closed_formula_coef
from pyPLNmodels._initialization import _init_components_prec
from pyPLNmodels._viz import _viz_network
from pyPLNmodels._data_handler import _extract_data_from_formula


THRESHOLD = 1e-5


class PlnNetwork(BaseModel):
    """Pln model with regularization on the number of parameters
    of the precision matrix (inverse covariance matrix) representing correlation
    between variables.

    Examples
    --------
    >>> from pyPLNmodels import PlnNetwork, load_scrna
    >>> data = load_scrna()
    >>> net = PlnNetwork(data["endog"], penalty = 1)
    >>> net.fit()
    >>> print(net)
    >>> net.viz(colors=data["labels"])
    >>> net.viz_network()

    >>> from pyPLNmodels import PlnNetwork, load_scrna
    >>> data = load_scrna()
    >>> net = PlnNetwork.from_formula("endog ~ 1 + labels", data=data, penalty = 1)
    >>> net.fit()
    >>> print(net)
    >>> net.viz(colors=data["labels"])
    >>> net.viz_network()
    """

    penalty: float
    _components_prec: torch.Tensor
    mask: torch.Tensor

    @_add_doc(
        BaseModel,
        example="""
            >>> from pyPLNmodels import PlnNetwork, load_scrna
            >>> data = load_scrna()
            >>> net = PlnNetwork.from_formula("endog ~ 1", data)
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
    ):  # pylint: disable=too-many-arguments
        self.penalty = penalty
        super().__init__(
            endog,
            exog=exog,
            offsets=offsets,
            compute_offsets_method=compute_offsets_method,
            add_const=add_const,
        )
        self.mask = torch.ones((self.dim, self.dim)).to(self._endog.device) - torch.eye(
            self.dim, device=self._endog.device
        )

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
    )
    def from_formula(
        cls,
        formula: str,
        data: dict[str : Union[torch.Tensor, np.ndarray, pd.DataFrame, pd.Series]],
        *,
        penalty: float,
        compute_offsets_method: {"zero", "logsum"} = "zero",
    ):  # pylint: disable=arguments-differ
        endog, exog, offsets = _extract_data_from_formula(formula, data)
        return cls(
            endog,
            penalty=penalty,
            exog=exog,
            offsets=offsets,
            compute_offsets_method=compute_offsets_method,
            add_const=False,
        )

    @_add_doc(
        BaseModel,
        example="""
        >>> from pyPLNmodels import PlnNetwork, load_scrna
        >>> data = load_scrna()
        >>> net = PlnNetwork.from_formula("endog ~ 1", data, penalty = 1)
        >>> net.fit()
        >>> print(net)

        >>> from pyPLNmodels import PlnNetwork, load_scrna
        >>> data = load_scrna()
        >>> net = PlnNetwork.from_formula("endog ~ 1", data, penalty = 1)
        >>> net.fit()
        >>> print(net)
        >>> net.fit(penalty = 10)
        >>> print(net)
        """,
        params="""
        penalty: float
            - The penalty parameter. The larger the penalty, the larger the
               sparsity of the precision matrix.
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
    ):  # pylint: disable = too-many-arguments
        if penalty is not None:
            if not isinstance(penalty, (int, float)):
                raise ValueError("penalty must be a float.")
            print(f"Changing penalty from {self.penalty} to : ", penalty, ".")
            self.penalty = penalty
        super().fit(maxiter=maxiter, lr=lr, tol=tol, verbose=verbose)

    @_add_doc(
        BaseModel,
        example="""
        >>> from pyPLNmodels import PlnNetwork, load_scrna
        >>> data = load_scrna()
        >>> net = PlnNetwork.from_formula("endog ~ 1", data, penalty = 1)
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
        return elbo_no_penalty - self.penalty * self._l1_penalty(precision)

    def _l1_penalty(self, precision):
        return torch.norm(precision * self.mask, p=1)

    @property
    def _precision(self):
        return self._components_prec @ (self._components_prec.T)

    @property
    def precision(self):
        """
        Precision matrix of the model (i.e. inverse covariance matrix).
        """
        return self._precision.detach().cpu()

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
        return torch.sum((torch.abs(self._precision) < THRESHOLD).float()) / 2

    def _init_model_parameters(self):
        self._components_prec = _init_components_prec(self._endog)

    @property
    def list_of_parameters_needing_gradient(self):
        list_params = [
            self._latent_mean,
            self._latent_sqrt_variance,
            self._components_prec,
        ]
        return list_params

    def viz_network(self, ax=None):
        """
        Visualize the network infered by the model, i.e. the correlation between variables.
        The network is created by establishing edges between variables that have
        a non-zero correlation.

        >>> from pyPLNmodels import PlnNetwork, load_scrna
        >>> data = load_scrna()
        >>> net = PlnNetwork(data["endog"], penalty = 1)
        >>> net.fit()
        >>> print(net)
        >>> net.viz(colors=data["labels"])
        >>> net.viz_network()
        """
        _viz_network(
            self.precision * (torch.abs(self.precision) > THRESHOLD),
            ax=ax,
            node_labels=self.column_names_endog,
        )

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
        return _closed_formula_coef(self._exog, self._latent_mean)

    @property
    def _additional_methods_list(self):
        return [".viz_network()"]

    @property
    def _additional_attributes_list(self):
        return [".nb_zeros_precision"]

    def _get_two_dim_covariances(self, sklearn_components):
        _two_dim_covariances(sklearn_components, self.latent_sqrt_variance)

    @property
    @_add_doc(
        BaseModel,
        example="""
        >>> from pyPLNmodels import PlnNetwork, load_scrna
        >>> data = load_scrna()
        >>> net = PlnNetwork.from_formula("endog ~ 1", data, penalty = 1)
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
        >>> net = PlnNetwork.from_formula("endog ~ 1", data, penalty = 1)
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
        >>> net = PlnNetwork.from_formula("endog ~ 1", data=data, penalty = 1)
        >>> net.fit()
        >>> net.plot_correlation_circle(variables_names=["MALAT1", "ACTB"])
        >>> net.plot_correlation_circle(variables_names=["A", "B"], indices_of_variables=[0, 4])
        """,
    )
    def plot_correlation_circle(
        self, variables_names, indices_of_variables=None, title: str = ""
    ):
        super().plot_correlation_circle(
            variables_names=variables_names,
            indices_of_variables=indices_of_variables,
            title=title,
        )

    @_add_doc(
        BaseModel,
        example="""
        >>> from pyPLNmodels import PlnNetwork, load_scrna
        >>> data = load_scrna()
        >>> net = PlnNetwork.from_formula("endog ~ 1", data=data)
        >>> net.fit()
        >>> net.biplot(variables_names=["MALAT1", "ACTB"])
        >>> net.biplot(variables_names=["A", "B"], indices_of_variables=[0, 4], colors=data["labels"])
        """,
    )
    def biplot(
        self,
        variables_names,
        *,
        indices_of_variables: np.ndarray = None,
        colors: np.ndarray = None,
        title: str = "",
    ):
        super().biplot(
            variables_names=variables_names,
            indices_of_variables=indices_of_variables,
            colors=colors,
            title=title,
        )

    @_add_doc(
        BaseModel,
        example="""
        >>> from pyPLNmodels import PlnNetwork, load_scrna
        >>> data = load_scrna()
        >>> net = PlnNetwork.from_formula("endog ~ 1", data=data)
        >>> net.fit()
        >>> net.pca_pairplot(n_components=5)
        >>> net.pca_pairplot(n_components=5, colors=data["labels"])
        """,
    )
    def pca_pairplot(self, n_components: bool = 3, colors=None):
        super().pca_pairplot(n_components=n_components, colors=colors)

    @_add_doc(
        BaseModel,
        returns="""
        torch.Tensor
            The transformed endogenous variables (latent variables of the model).
        """,
        example="""
              >>> from pyPLNmodels import PlnNetwork, load_scrna
              >>> data = load_scrna()
              >>> net = PlnNetwork.from_formula("endog ~ 1", data=data)
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
            >>> import matplotlib.pyplot as plt
            >>> from pyPLNmodels import PlnNetwork, load_scrna
            >>> data = load_scrna()
            >>> net = PlnNetwork(data["endog"])
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
            >>> import matplotlib.pyplot as plt
            >>> from pyPLNmodels import PlnNetwork, load_scrna
            >>> data = load_scrna()
            >>> net = PlnNetwork.from_formula("endog ~ 1 + labels", data=data)
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
