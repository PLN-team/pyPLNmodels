from pyPLNmodels import Pln, PlnPCA, ZIPln, PlnDiag, PlnNetwork

PENALTY = 1


def _basic_init(model):
    def _init_function(init_method, **kwargs):
        if init_method == "explicit":
            endog = kwargs.get("endog", None)
            exog = kwargs.get("exog", None)
            offsets = kwargs.get("offsets", None)
            add_const = kwargs.get("add_const", False)
            return model(
                endog=endog,
                exog=exog,
                offsets=offsets,
                add_const=add_const,
            )
        if init_method == "formula":
            data = kwargs.get("data", None)
            formula = kwargs.get("formula", None)
            return model.from_formula(formula, data=data)
        raise ValueError('init_method must be "explicit" or "formula"')

    return _init_function


def _Pln_init(init_method, **kwargs):
    _init = _basic_init(Pln)
    return _init(init_method, **kwargs)


def _PlnDiag_init(init_method, **kwargs):
    _init = _basic_init(PlnDiag)
    return _init(init_method, **kwargs)


def _PlnNetwork_init(init_method, **kwargs):
    if init_method == "explicit":
        endog = kwargs.get("endog", None)
        exog = kwargs.get("exog", None)
        offsets = kwargs.get("offsets", None)
        add_const = kwargs.get("add_const", False)
        return PlnNetwork(
            endog=endog,
            exog=exog,
            offsets=offsets,
            add_const=add_const,
            penalty=PENALTY,
        )
    if init_method == "formula":
        data = kwargs.get("data", None)
        formula = kwargs.get("formula", None)
        return PlnNetwork.from_formula(formula, data=data, penalty=PENALTY)
    raise ValueError('init_method must be "explicit" or "formula"')


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
