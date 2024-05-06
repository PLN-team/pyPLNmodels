from typing import Tuple, Union
import warnings
import pkg_resources

import torch
import numpy as np
import pandas as pd


from pyPLNmodels._utils import (
    _check_two_tensors_dimensions_are_equal,
    _format_data,
    _sigmoid,
)


def _get_simulation_components(dim: int, rank: int) -> torch.Tensor:
    """
    Get the components for simulation. The resulting covariance matrix
    will be a matrix per blocks plus a little noise.

    Parameters
    ----------
    dim : int
        Dimension of the data.
    rank : int
        Rank of the resulting covariance matrix (i.e. number of components).

    Returns
    -------
    torch.Tensor
        Components.
    """
    block_size = dim // rank
    prev_state = torch.random.get_rng_state()
    torch.random.manual_seed(0)
    components = torch.zeros(dim, rank)
    for column_number in range(rank):
        components[
            column_number * block_size : (column_number + 1) * block_size, column_number
        ] = 1
    # components += torch.randn(dim, rank) / 8
    torch.random.set_rng_state(prev_state)
    return components


def _get_simulation_coef_cov_offsets_coefzi(
    n_samples: int,
    nb_cov: int,
    nb_cov_inflation: int,
    dim: int,
    add_const: bool,
    add_const_inflation: bool,
    zero_inflation_formula: {None, "global", "column-wise", "row-wise"},
    mean_infla: float,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Get offsets, covariance coefficients with right shapes.

    Parameters
    ----------
    n_samples : int
        Number of samples.
    nb_cov : int
        Number of exog. If 0, exog will be None,
        unless add_const is True.
        If add_const is True, then there will be nb_cov+1
        exog as the intercept can be seen as a exog.
    nb_cov_inflation : int
        Number of exog for the inflation part.
        If zero, will not add zero inflation except if
        zero_inflation_formula is "global"
    dim : int
        Dimension required of the data.
    add_const : bool
        If True, will add a vector of ones in the exog.
    add_const_inflation : bool
        If True, will add a vector of ones in the exog_inflation
    zero_inflation_formula : {None, "global", "column-wise","row-wise"}
        If None, coef_inflation will be None.
        If "global", will return one global coefficient.
        If "column-wise", will return a (n_samples, nb_cov_inflation) torch.Tensor
        If "row-wise", will return a (nb_cov_inflation, dim) torch.Tensor
    seed: int
        The seed for simulation.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Tuple containing offsets, exog, and coefficients.
    """
    prev_state = torch.random.get_rng_state()
    torch.random.manual_seed(seed)
    _check_all_integers_or_none([n_samples, dim, nb_cov, nb_cov_inflation])
    if nb_cov_inflation == 0:
        if zero_inflation_formula == "global":
            if add_const_inflation is True:
                msg = "If the zero_inflation_formula is global, you can not have add_const_inflation=True."
                raise ValueError(msg)
            else:
                exog_inflation = None
        elif zero_inflation_formula in ["row-wise", "column-wise"]:
            if add_const_inflation is False:
                raise ValueError(
                    "nb_cov_inflation=0 and requesting a zero inflation formula"
                )
            if add_const_inflation is True:
                if zero_inflation_formula == "column-wise":
                    exog_inflation = torch.ones(n_samples, 1)
                else:
                    exog_inflation = torch.ones(1, dim)

        else:
            if zero_inflation_formula is None:
                exog_inflation = None
            else:
                raise ValueError("Wrong argument for zero_inflation_formula.")
    else:
        if zero_inflation_formula == "global":
            raise ValueError(
                "nb_cov_inflation should be 0 if the zero_inflation_formula is global."
            )
        elif zero_inflation_formula == "column-wise":
            exog_inflation = torch.randint(
                low=0,
                high=2,
                size=(n_samples, nb_cov_inflation),
                dtype=torch.float64,
            )
            if add_const_inflation is True:
                exog_inflation = torch.cat(
                    (exog_inflation, torch.ones(n_samples, 1)), axis=1
                )

        elif zero_inflation_formula == "row-wise":
            exog_inflation = torch.randint(
                low=1,
                high=3,
                size=(nb_cov_inflation, dim),
                dtype=torch.float64,
            )
            if add_const_inflation is True:
                exog_inflation = torch.cat((exog_inflation, torch.ones(1, dim)), axis=0)
        else:
            msg = f"Wrong zero_inflation formula. Got {zero_inflation_formula},"
            msg += ' expected "column-wise" or "row-wise".'
            raise ValueError(msg)
    if zero_inflation_formula is not None:
        if zero_inflation_formula in ["column-wise", "row-wise"]:
            if nb_cov_inflation > 0:
                exog_inflation = torch.randint(
                    low=0,
                    high=2,
                    size=(n_samples, nb_cov_inflation),
                    dtype=torch.float64,
                )
                exog_inflation -= (exog_inflation == 0) * torch.ones(
                    exog_inflation.shape
                )
                if add_const_inflation is True:
                    exog_inflation = torch.cat(
                        (exog_inflation, torch.ones(n_samples, 1)), axis=1
                    )
            else:
                exog_inflation = torch.ones(n_samples, 1)

            coef_inflation = torch.randn(exog_inflation.shape[1], dim) / np.sqrt(
                nb_cov_inflation + 1
            )
            coef_inflation += -torch.mean(coef_inflation) + torch.logit(
                torch.tensor([mean_infla])
            )
            if zero_inflation_formula == "row-wise":
                coef_inflation, exog_inflation = exog_inflation, coef_inflation
        else:
            coef_inflation = torch.logit(torch.Tensor([mean_infla]))
    else:
        coef_inflation = None

    if nb_cov == 0:
        if add_const is True:
            exog = torch.ones(n_samples, 1)
        else:
            exog = None
    else:
        exog = torch.randint(
            low=0,
            high=2,
            size=(n_samples, nb_cov),
            dtype=torch.float64,
        )
        exog -= (exog == 0) * torch.ones(exog.shape)
        if add_const is True:
            exog = torch.cat((exog, torch.ones(n_samples, 1)), axis=1)
    if exog is None:
        coef = None
    else:
        coef = torch.randn(exog.shape[1], dim) / np.sqrt(nb_cov + 1)
    offsets = torch.randint(low=0, high=2, size=(n_samples, dim), dtype=torch.float64)
    torch.random.set_rng_state(prev_state)
    return coef, exog, exog_inflation, offsets, coef_inflation


class PlnParameters:
    def __init__(
        self,
        *,
        components: Union[torch.Tensor, np.ndarray, pd.DataFrame],
        coef: Union[torch.Tensor, np.ndarray, pd.DataFrame],
        exog: Union[torch.Tensor, np.ndarray, pd.DataFrame],
        offsets: Union[torch.Tensor, np.ndarray, pd.DataFrame],
        n_samples: int,
    ):
        """
        Instantiate all the needed parameters to sample from the PLN model.

        Parameters
        ----------
        components : Union[torch.Tensor, np.ndarray, pd.DataFrame](keyword-only)
            Components of size (p, rank)
        coef : Union[torch.Tensor, np.ndarray, pd.DataFrame](keyword-only)
            Coefficient of size (d, p)
        exog : Union[torch.Tensor, np.ndarray, pd.DataFrame] or None(keyword-only)
            Covariates, size (n, d) or None
        offsets : Union[torch.Tensor, np.ndarray, pd.DataFrame](keyword-only)
            Offset, size (n, p)
        n_samples : int
            The number of samples that will be produced.
        Raises

        """
        self._components = _format_data(components)
        self._coef = _format_data(coef)
        self._exog = _format_data(exog)
        self._offsets = _format_data(offsets)
        self._n_samples = n_samples
        if self._coef is not None:
            _check_two_tensors_dimensions_are_equal(
                "components",
                "coef",
                self._components.shape[0],
                self._coef.shape[1],
                0,
                1,
            )
        if self._offsets is not None:
            _check_two_tensors_dimensions_are_equal(
                "components",
                "offsets",
                self._components.shape[0],
                self._offsets.shape[1],
                0,
                1,
            )
            _check_one_dimension(self._offsets, "offsets", 0, self._n_samples)
        if self._exog is not None:
            _check_two_tensors_dimensions_are_equal(
                "offsets",
                "exog",
                self._offsets.shape[0],
                self._exog.shape[0],
                0,
                0,
            )
            _check_one_dimension(self._exog, "offsets", 0, self._n_samples)
            _check_two_tensors_dimensions_are_equal(
                "exog",
                "coef",
                self._exog.shape[1],
                self._coef.shape[0],
                1,
                0,
            )
        for array in [self._components, self._coef, self._exog, self._offsets]:
            if array is not None:
                if len(array.shape) != 2:
                    raise RuntimeError(
                        f"Expected all arrays to be 2-dimensional, got {len(array.shape)}"
                    )

    @property
    def dim(self):
        """The dimension (number of variables) of the model."""
        return self.components.shape[0]

    @property
    def nb_cov(self):
        """The number of covariates of the model."""
        if self._exog is None:
            return 0
        return self._exog.shape[1]

    @property
    def gaussian_mean(self):
        """return the mean of the gaussian"""
        if self.exog is None:
            return None
        return self.exog @ self.coef

    @property
    def n_samples(self):
        """Number of samples."""
        return self._n_samples

    @property
    def covariance(self):
        """
        Covariance of the model.
        """
        return self._components @ self._components.T

    @property
    def components(self):
        """
        Components of the model.
        """
        return self._components

    @property
    def offsets(self):
        """
        Data offsets.
        """
        return self._offsets

    @property
    def coef(self):
        """
        Coef of the model.
        """
        if self._coef is None:
            return None
        return self._coef

    @property
    def exog(self):
        """
        Data exog.
        """
        if self._exog is None:
            return None
        return self._exog

    def _set_gaussian_mean(self, mean_gaussian: float):
        self._coef += -torch.mean(self._coef) + mean_gaussian

    def _set_mean_proba(self, mean_proba: float):
        if mean_proba > 1 or mean_proba < 0:
            raise ValueError("The mean should be a probability (0<p<1).")
        if self._zero_inflation_formula == "column-wise":
            self._coef_inflation += -torch.mean(self._coef_inflation) + torch.logit(
                torch.tensor([mean_proba])
            )
        elif self._zero_inflation_formula == "row-wise":
            self._exog_inflation += -torch.mean(self._exog_inflation) + torch.logit(
                torch.tensor([mean_proba])
            )
        else:
            self._coef_inflation = torch.logit(torch.tensor([mean_proba]))


def _check_one_dimension(
    array: torch.Tensor, array_name: str, dim: int, expected_dim: int
):
    """Raises a value error if the tensor has not the right shape on the given dimension."""
    if array.shape[dim] != expected_dim:
        msg = f"{array_name} should have size {expected_dim} for dimension {dim}."
        raise ValueError(msg)


class ZIPlnParameters(PlnParameters):
    def __init__(
        self,
        *,
        components: Union[torch.Tensor, np.ndarray, pd.DataFrame],
        coef: Union[torch.Tensor, np.ndarray, pd.DataFrame],
        coef_inflation: Union[torch.Tensor, np.ndarray, pd.DataFrame, float],
        exog: Union[torch.Tensor, np.ndarray, pd.DataFrame],
        exog_inflation: Union[torch.Tensor, np.ndarray, pd.DataFrame, None],
        offsets: Union[torch.Tensor, np.ndarray, pd.DataFrame],
        zero_inflation_formula="column-wise",
        n_samples: int,
    ):
        """
        Instantiate all the needed parameters to sample from the PLN model.

        Parameters
        ----------
        components : Union[torch.Tensor, np.ndarray, pd.DataFrame](keyword-only)
            Components of size (p, rank)
        coef : Union[torch.Tensor, np.ndarray, pd.DataFrame](keyword-only)
            Coefficient of size (d, p)
        coef_inflation : Union[torch.Tensor, np.ndarray, pd.DataFrame, float], optional(keyword-only)
            Coefficient for zero-inflation model, size (d, p) or (n,d).
        exog : Union[torch.Tensor, np.ndarray, pd.DataFrame] or None(keyword-only)
            Covariates, size (n, d) or None
        exog_inflation : Union[torch.Tensor, np.ndarray, pd.DataFrame] or None(keyword-only)
            Covariates, size (n, d) or None
        offsets : Union[torch.Tensor, np.ndarray, pd.DataFrame](keyword-only)
            Offset, size (n, p)
        zero_inflation_formula : {"column-wise", "row-wise","global"}
            Which formula should the inflation be.
        n_samples : int
            The number of samples that will be produced.
        """
        super().__init__(
            components=components,
            coef=coef,
            exog=exog,
            offsets=offsets,
            n_samples=n_samples,
        )
        if isinstance(coef_inflation, float):
            if (coef_inflation < 1 and coef_inflation > 0) is False:
                raise ValueError("coef_inflation should be between 0 and 1")
            else:
                self._coef_inflation = coef_inflation
        else:
            self._coef_inflation = _format_data(coef_inflation)
        self._check_formula(zero_inflation_formula)
        self._zero_inflation_formula = zero_inflation_formula
        self._exog_inflation = _format_data(exog_inflation)
        if zero_inflation_formula == "column-wise":
            _check_one_dimension(
                self.exog_inflation, "exog_inflation", 0, self._n_samples
            )
            _check_two_tensors_dimensions_are_equal(
                "exog_inflation",
                "coef_inflation",
                self._exog_inflation.shape[1],
                self.coef_inflation.shape[0],
                1,
                0,
            )
        if zero_inflation_formula == "row-wise":
            _check_one_dimension(
                self.coef_inflation, "coef_inflation", 0, self.n_samples
            )
            _check_two_tensors_dimensions_are_equal(
                "coef_inflation",
                "exog_inflation",
                self.coef_inflation.shape[1],
                self._exog_inflation.shape[0],
                1,
                0,
            )
        if zero_inflation_formula == "global":
            if coef_inflation.shape != (1,):
                msg = "If the zero_inflation_formula is global, the "
                msg += "zero_inflation_formula should be a tensor of size 1."
                raise ValueError(msg)

    @property
    def proba_inflation(self):
        if self._zero_inflation_formula == "column-wise":
            return _sigmoid(self._exog_inflation @ self._coef_inflation)
        elif self._zero_inflation_formula == "row-wise":
            return _sigmoid(self._coef_inflation @ self._exog_inflation)
        return torch.sigmoid(self._coef_inflation)

    @property
    def coef_inflation(self):
        """
        Inflation coefficient of the model.
        """
        return self._coef_inflation

    @property
    def exog_inflation(self):
        """
        Inflation coefficient of the model.
        """
        return self._exog_inflation

    def _check_formula(self, zero_inflation_formula):
        list_available = ["column-wise", "row-wise", "global"]
        if zero_inflation_formula not in list_available:
            msg = f"Wrong inflation formula, got {zero_inflation_formula}, expected one of {list_available}"
            raise ValueError(msg)

    def nb_cov_inflation(self):
        if self._zero_inflation_formula == "column-wise":
            return self.coef_inflation.shape[1]
        if self._zero_inflation_formula == "row-wise":
            return self.coef_inflation.shape[0]
        if self._zero_inflation_formula == "global":
            return None


def sample_pln(pln_param, *, seed: int = None, return_latent=False) -> torch.Tensor:
    """
    Sample from the Poisson Log-Normal (Pln) model.

    Parameters
    ----------
    pln_param : PlnParameters object
        parameters of the model, containing the coeficient, the exog,
        the components and the offsets.
    seed : int or None, optional(keyword-only)
        Random seed for reproducibility. Default is None.
    return_latent : bool, optional(keyword-only)
        If True will return also the latent variables. Default is False.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor] if return_latent is True
        Tuple containing endog (torch.Tensor) and gaussian (torch.Tensor)
    torch.Tensor if return_latent is False

    See also :func:`~pyPLNmodels.PlnParameters`
    See also :func:`-pyPLNmodels.ZIPlnParameters`
    See also :func:`-pyPLNmodels.sample_zipln`
    """
    prev_state = torch.random.get_rng_state()
    if seed is not None:
        torch.random.manual_seed(seed)

    n_samples = pln_param.offsets.shape[0]
    rank = pln_param.components.shape[1]

    if pln_param.exog is None:
        XB = 0
    else:
        XB = torch.matmul(pln_param.exog, pln_param.coef)

    gaussian = torch.mm(torch.randn(n_samples, rank), pln_param.components.T) + XB
    parameter = torch.exp(pln_param.offsets + gaussian)
    endog = torch.poisson(parameter)

    torch.random.set_rng_state(prev_state)
    if return_latent is True:
        return endog, gaussian
    return endog


def sample_zipln(
    zipln_param, *, seed: int = None, return_latent=False, return_pln=False
) -> torch.Tensor:
    """
    Sample from the Zero Inflated Poisson Log-Normal (ZIPln) model.

    Parameters
    ----------
    zipln_param : ZIPlnParameters object
        parameters of the model, containing the coeficient, the exog,
        the components, the offsets, the exog_inflation and the
        coef_inflation.
    seed : int or None, optional(keyword-only)
        Random seed for reproducibility. Default is None.
    return_latent : bool, optional(keyword-only)
        If True will return also the latent variables. Default is False.
    return_pln : bool, optional(keyword-only)
        If True will also return the sampling from a pln model.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor] if return_latent is True
        Tuple containing endog (torch.Tensor), gaussian (torch.Tensor), and ksi (torch.Tensor)
        An additional tensor is added if return_pln is True
    torch.Tensor if return_latent is False. An additional tensor is added if return_pln is True

    See also :func:`~pyPLNmodels.PlnParameters`
    See also :func:`-pyPLNmodels.ZIPlnParameters`
    See also :func:`-pyPLNmodels.sample_pln`
    """
    print("ZIPln is sampled")
    prev_state = torch.random.get_rng_state()
    if seed is not None:
        torch.random.manual_seed(seed)
    proba_inflation = zipln_param.proba_inflation
    if zipln_param._zero_inflation_formula == "global":
        ksi = torch.bernoulli(
            torch.ones(zipln_param.n_samples, zipln_param.dim) * proba_inflation
        )
    else:
        ksi = torch.bernoulli(proba_inflation)
    pln_endog, gaussian = sample_pln(zipln_param, seed=seed, return_latent=True)
    endog = (1 - ksi) * pln_endog
    torch.random.set_rng_state(prev_state)
    if return_latent is True:
        if return_pln is True:
            return endog, gaussian, ksi, pln_endog
        return endog, gaussian, ksi
    if return_pln is True:
        return endog, pln_endog
    return endog


def get_pln_simulated_count_data(
    *,
    n_samples: int = 50,
    dim: int = 25,
    rank: int = 5,
    nb_cov: int = 1,
    return_true_param: bool = False,
    add_const: bool = True,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Get simulated count data from the PlnPCA model.

    Parameters
    ----------
    n_samples : int, optional(keyword-only)
        Number of samples, by default 50.
    dim : int, optional(keyword-only)
        Dimension, by default 25.
    rank : int, optional(keyword-only)
        Rank of the covariance matrix, by default 5.
    add_const : bool, optional(keyword-only)
        If True, will add a vector of ones in the exog. Default is True
    nb_cov : int, optional(keyword-only)
        Number of exog, by default 1.
    return_true_param : bool, optional(keyword-only)
        Whether to return the true parameters of the model, by default False.
    seed : int, optional(keyword-only)
        Seed value for random number generation, by default 0.
    Returns
    -------
    if return_true_param is False:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Tuple containing endog, exog, and offsets.
    else:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            Tuple containing endog, exog, offsets, covariance, coef.

    """
    param = get_simulation_parameters(
        n_samples=n_samples,
        dim=dim,
        nb_cov=nb_cov,
        nb_cov_inflation=0,
        rank=rank,
        add_const=add_const,
        add_const_inflation=False,
        zero_inflation_formula=None,
    )
    endog = sample_pln(param, seed=seed, return_latent=False)
    if return_true_param:
        return (
            endog,
            param.exog,
            param.offsets,
            param.covariance,
            param.coef,
        )
    return endog, param.exog, param.offsets


def get_zipln_simulated_count_data(
    *,
    n_samples: int = 100,
    dim: int = 25,
    rank: int = 5,
    nb_cov: int = 1,
    nb_cov_inflation: int = 0,
    return_true_param: bool = False,
    add_const: bool = True,
    add_const_inflation: bool = True,
    zero_inflation_formula: {"global", "column-wise", "row-wise"} = "column-wise",
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Get simulated count data from the PlnPCA model.

    Parameters
    ----------
    n_samples : int, optional(keyword-only)
        Number of samples, by default 100.
    dim : int, optional(keyword-only)
        Dimension, by default 25.
    rank : int, optional(keyword-only)
        Rank of the covariance matrix, by default 5.
    add_const : bool, optional(keyword-only)
        If True, will add a vector of ones to the exog. Default is True.
    add_const_inflation : bool, optional(keyword-only)
        If True, will add a vector of ones to the exog_inflation. Default is True.
    nb_cov : int, optional(keyword-only)
        Number of exog, by default 1.
    nb_cov_inflation : int, optional(keyword-only)
        Number of exog, by default 1.
    return_true_param : bool, optional(keyword-only)
        Whether to return the true parameters of the model, by default False.
    zero_inflation_formula : {"column-wise", "global","row-wise"}
        If "column-wise", will return a (n_samples, nb_cov_inflation) torch.Tensor
        If "global", will return one global coefficient.
        If "row-wise", will return a (nb_cov_inflation, dim) torch.Tensor
        Default is "column-wise".
    seed : int, optional(keyword-only)
        Seed value for random number generation, by default 0.

    Returns
    -------
    if return_true_param is False:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            Tuple containing endog, exog,exog_infla, offsets.
    else:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            Tuple containing endog, exog, exog_infla, offsets, covariance, coef, coef_inflation .

    """
    param = get_simulation_parameters(
        n_samples=n_samples,
        dim=dim,
        nb_cov=nb_cov,
        nb_cov_inflation=nb_cov_inflation,
        rank=rank,
        add_const=add_const,
        add_const_inflation=add_const_inflation,
        zero_inflation_formula=zero_inflation_formula,
    )
    endog = sample_zipln(param)
    if return_true_param is True:
        return (
            endog,
            param.exog,
            param.exog_inflation,
            param.offsets,
            param.covariance,
            param.coef,
            param.coef_inflation,
        )
    return endog, param.exog, param.exog_inflation, param.offsets


def get_simulation_parameters(
    *,
    n_samples: int = 50,
    dim: int = 25,
    nb_cov: int = 1,
    nb_cov_inflation: int = 0,
    rank: int = 5,
    add_const: bool = True,
    add_const_inflation: bool = False,
    zero_inflation_formula: {None, "global", "column-wise", "row-wise"} = None,
    mean_infla=0.2,
    seed=0,
) -> Union[PlnParameters, ZIPlnParameters]:
    """
    Generate simulation parameters for a Poisson-lognormal model.

    Parameters
    ----------
        n_samples : int, optional(keyword-only)
            The number of samples, by default 100.
        dim : int, optional(keyword-only)
            The dimension of the data, by default 25.
        nb_cov : int, optional(keyword-only)
            The number of exog, by default 1. If add_const is True,
            then there will be nb_cov+1 exog as the intercept can be seen
            as a exog.
        nb_cov_inflation : int, optional(keyword-only)
            The number of exog for the inflation part.
            If 0, will not add zero-inflation. Default is zero.
        rank : int, optional(keyword-only)
            The rank of the data components, by default 5.
        add_const : bool, optional(keyword-only)
            If True, will add a vector of ones in the exog.
        add_const_inflation : bool, optional(keyword-only)
            If True, will add a vector of ones in the exog_inflation.
        zero_inflation_formula : {None, "global", "column-wise","row-wise"}
            If None, coef_inflation will be None.
            If "global", will return one global coefficient.
            If "column-wise", will return a (n_samples, nb_cov_inflation) torch.Tensor
            If "row-wise", will return a (nb_cov_inflation, dim) torch.Tensor
        seed

    Returns
    -------
        PlnParameters if zero_inflation_formula is None
            The generated simulation parameters.
        ZIPlnParamter if zero_inflation_formula is not None
            The generated simulation parameters for a zero inflation context.
    """
    (
        coef,
        exog,
        exog_inflation,
        offsets,
        coef_inflation,
    ) = _get_simulation_coef_cov_offsets_coefzi(
        n_samples,
        nb_cov,
        nb_cov_inflation,
        dim,
        add_const,
        add_const_inflation,
        zero_inflation_formula,
        mean_infla,
        seed,
    )
    if add_const_inflation is True and zero_inflation_formula is None:
        warnings.warn(
            "add const_inflation set to True but no zero inflation is sampled."
        )
    components = _get_simulation_components(dim, rank)
    sigma = components @ (components.T)
    sigma += torch.eye(components.shape[0])
    # omega = torch.inverse(sigma)
    # omega = sigma
    components = torch.linalg.cholesky(sigma)
    if coef_inflation is None:
        print("Pln model will be sampled.")
        return PlnParameters(
            components=components,
            coef=coef,
            exog=exog,
            offsets=offsets,
            n_samples=n_samples,
        )
    print("ZIPln model will be sampled.")
    return ZIPlnParameters(
        components=components,
        coef=coef,
        coef_inflation=coef_inflation,
        exog=exog,
        exog_inflation=exog_inflation,
        offsets=offsets,
        zero_inflation_formula=zero_inflation_formula,
        n_samples=n_samples,
    )


def _check_all_integers_or_none(l):
    for element in l:
        if element is not None:
            if not isinstance(element, int):
                raise ValueError(f"Got,{type(element)}, expected an int")
