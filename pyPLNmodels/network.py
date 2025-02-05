from typing import Optional, Union

import torch
import numpy as np
import pandas as pd
from sklearn.covariance import graphical_lasso as GL


from pyPLNmodels.base import BaseModel, DEFAULT_TOL
from pyPLNmodels.pln import Pln
from pyPLNmodels.elbos import elbo_pln
from pyPLNmodels._utils import _add_doc
from pyPLNmodels._closed_forms import _closed_formula_covariance
from pyPLNmodels._initialization import _init_components_prec

THRESHOLD = 1e-5


class PlnNetwork(Pln):
    """Pln model with regularization on the number of parameters
    of the covariance matrix.

    Examples
    --------
    >>> from pyPLNmodels import PlnNetwork, load_scrna
    >>> data = load_scrna()
    >>> net = PlnNetwork(data["endog"])
    >>> net.fit()
    >>> print(net)
    >>> net.viz(colors=data["labels"])
    >>> net.viz_network()

    >>> from pyPLNmodels import PlnNetwork, load_scrna
    >>> data = load_scrna()
    >>> net = PlnNetwork.from_formula("endog ~ 1 + labels", data=data)
    >>> net.fit()
    >>> print(net)
    >>> net.viz(colors=data["labels"])
    >>> net.viz_network()
    """

    penalty: float
    _components_prec: torch.Tensor
    mask: torch.Tensor
    use_gl: bool

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
        *,
        exog: Optional[Union[torch.Tensor, np.ndarray, pd.DataFrame, pd.Series]] = None,
        offsets: Optional[Union[torch.Tensor, np.ndarray, pd.DataFrame]] = None,
        compute_offsets_method: {"zero", "logsum"} = "zero",
        add_const: bool = True,
    ):  # pylint: disable=too-many-arguments
        super().__init__(
            endog,
            exog=exog,
            offsets=offsets,
            compute_offsets_method=compute_offsets_method,
            add_const=add_const,
        )

    @_add_doc(
        BaseModel,
        example="""
        >>> from pyPLNmodels import PlnNetwork, load_scrna
        >>> data = load_scrna()
        >>> net = PlnNetwork.from_formula("endog ~ 1", data)
        >>> net.fit()
        >>> print(net)

        >>> from pyPLNmodels import PlnNetwork, load_scrna
        >>> data = load_scrna()
        >>> net = PlnNetwork.from_formula("endog ~ 1", data)
        >>> net.fit(criterion_or_penalty = "BIC")
        >>> print(net)
        """,
        params="""
        criterion_or_penalty: Union[Literal["AIC"], Literal["BIC"], float]
            - If "AIC" or "BIC": the penalty parameter will be derived
                by maximizing the given criterion.
            - If `float`, optimize the ELBO with the given penalty parameter.
        """,
    )
    def fit(
        self,
        *,
        maxiter: int = 400,
        lr: float = 0.01,
        tol: float = DEFAULT_TOL,
        verbose: bool = False,
        penalty=1,
        use_gl: bool = False,
    ):  # pylint: disable = too-many-arguments
        self.penalty = penalty
        self.use_gl = use_gl
        self.mask = torch.ones((self.dim, self.dim)).to(self._endog.device) - torch.eye(
            self.dim, device=self._endog.device
        )
        super().fit(maxiter=maxiter, lr=lr, tol=tol, verbose=verbose)

    @_add_doc(
        BaseModel,
        example="""
        >>> from pyPLNmodels import PlnNetwork, load_scrna
        >>> data = load_scrna()
        >>> net = PlnNetwork.from_formula("endog ~ 1", data)
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
        return elbo_no_penalty - self.penalty * self._l0_penalty(precision)

    def _l0_penalty(self, precision):
        return torch.norm(precision * self.mask, p=1)

    @property
    def _precision(self):
        if self.use_gl is True:
            covariance = _closed_formula_covariance(
                self._marginal_mean,
                self._latent_mean,
                self._latent_sqrt_variance,
                self.n_samples,
            )
            _, precision = GL(covariance.detach().cpu().numpy(), alpha=self.penalty)
            return torch.from_numpy(precision).to(self._endog.device)
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
            self.dim * (self.dim + 2 * self.nb_cov + 1) / 2 - self._nb_zeros_precision
        )
        return nb_param

    @property
    def _nb_zeros_precision(self):
        return torch.sum((torch.abs(self.precision) < THRESHOLD).float()) / 2

    def _init_model_parameters(self):
        if self.use_gl is False:
            self._components_prec = _init_components_prec(self._endog)

    @property
    def list_of_parameters_needing_gradient(self):
        list_params = [self._latent_mean, self._latent_sqrt_variance]
        if self.use_gl is False:
            list_params.append(self._components_prec)
        return list_params
