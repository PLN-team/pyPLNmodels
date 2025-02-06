from pyPLNmodels import Pln, PlnPCA, ZIPln

PENALTY = 1


def _Pln_init(init_method, **kwargs):
    if init_method == "explicit":
        endog = kwargs.get("endog", None)
        exog = kwargs.get("exog", None)
        offsets = kwargs.get("offsets", None)
        add_const = kwargs.get("add_const", False)
        return Pln(
            endog=endog,
            exog=exog,
            offsets=offsets,
            add_const=add_const,
        )
    if init_method == "formula":
        data = kwargs.get("data", None)
        formula = kwargs.get("formula", None)
        return Pln.from_formula(formula, data=data)
    raise ValueError('init_method must be "explicit" or "formula"')


def _PlnDiag_init(init_method, **kwargs):
    return _Pln_init(init_method, **kwargs)


def _PlnNetwork_init(init_method, **kwargs):
    kwargs["penalty"] = PENALTY
    return _Pln_init(init_method, **kwargs)


def _PlnPCA_init(init_method, **kwargs):
    rank = kwargs.get("rank", None)
    if init_method == "explicit":
        endog = kwargs.get("endog", None)
        exog = kwargs.get("exog", None)
        offsets = kwargs.get("offsets", None)
        add_const = kwargs.get("add_const", False)
        return PlnPCA(
            endog=endog, exog=exog, offsets=offsets, add_const=add_const, rank=rank
        )
    if init_method == "formula":
        data = kwargs.get("data", None)
        formula = kwargs.get("formula", None)
        return PlnPCA.from_formula(formula, data=data, rank=rank)
    raise ValueError('init_method must be "explicit" or "formula"')


def _ZIPln_init(init_method, **kwargs):
    if init_method == "explicit":
        endog = kwargs.get("endog", None)
        exog = kwargs.get("exog", None)
        exog_inflation = kwargs.get("exog_inflation", None)
        offsets = kwargs.get("offsets", None)
        add_const = kwargs.get("add_const", False)
        add_const_inflation = kwargs.get("add_const_inflation", False)
        return ZIPln(
            endog=endog,
            exog=exog,
            offsets=offsets,
            add_const=add_const,
            exog_inflation=exog_inflation,
            add_const_inflation=add_const_inflation,
        )
    if init_method == "formula":
        data = kwargs.get("data", None)
        formula = kwargs.get("formula", None)
        return ZIPln.from_formula(formula, data=data)
    raise ValueError('init_method must be "explicit" or "formula"')
