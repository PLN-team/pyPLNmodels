# pylint: skip-file
import pytest
from pyPLNmodels import PlnNetwork, load_scrna, PlnNetworkCollection


data = load_scrna()


def test_penalty_coef_network():
    _ = PlnNetwork.from_formula(
        "endog ~ 1 + labels", data=data, penalty=10, penalty_coef=1
    ).fit(maxiter=10)
    _ = PlnNetwork.from_formula(
        "endog ~ 1 + labels", data=data, penalty=10, penalty_coef=0
    ).fit(maxiter=10)
    with pytest.raises(ValueError):
        _ = PlnNetwork.from_formula(
            "endog ~ 1 + labels",
            data=data,
            penalty=10,
            penalty_coef=0,
            penalty_coef_type="group_lasso",
        )
    with pytest.raises(ValueError):
        _ = PlnNetwork.from_formula(
            "endog ~ 1 + labels",
            data=data,
            penalty=10,
            penalty_coef=-1,
            penalty_coef_type="group_lasso",
        )
    net = PlnNetwork.from_formula(
        "endog ~ 1 + labels",
        data=data,
        penalty=10,
        penalty_coef=1,
        penalty_coef_type="sparse_group_lasso",
    )
    net.fit(penalty_coef=10)
    net.fit(penalty_coef=0, penalty_coef_type="lasso")
    net.fit(penalty_coef_type="group_lasso", maxiter=10, penalty_coef=10)
    net.fit(penalty_coef_type="sparse_group_lasso", maxiter=10, penalty_coef=5)
    with pytest.raises(ValueError):
        net.fit(penalty_coef=-10)
    with pytest.raises(ValueError):
        _ = PlnNetwork.from_formula(
            "endog ~ 1 + labels",
            data=data,
            penalty=10,
            penalty_coef=1,
            penalty_coef_type="dumb",
        )

    _ = PlnNetwork.from_formula("endog ~ 0", data=data, penalty=10, penalty_coef=1).fit(
        maxiter=10, penalty_coef=5
    )
    net = PlnNetwork.from_formula(
        "endog ~ 1 + labels",
        data=data,
        penalty=1,
        penalty_coef=1,
        penalty_coef_type="sparse_group_lasso",
    ).fit(maxiter=10)
    with pytest.raises(ValueError):
        net.components_prec = net._components_prec[:3]


def test_penalty_coef_network_collection():
    _ = PlnNetworkCollection.from_formula(
        "endog ~ 1 + labels", data=data, penalties=[1, 10], penalty_coef=1
    ).fit(maxiter=10)
    _ = PlnNetworkCollection.from_formula(
        "endog ~ 1 + labels", data=data, penalties=[1, 10], penalty_coef=0
    ).fit(maxiter=10)
    with pytest.raises(ValueError):
        _ = PlnNetworkCollection.from_formula(
            "endog ~ 1 + labels",
            data=data,
            penalties=[1, 10],
            penalty_coef=0,
            penalty_coef_type="group_lasso",
        )
    with pytest.raises(ValueError):
        _ = PlnNetworkCollection.from_formula(
            "endog ~ 1 + labels",
            data=data,
            penalties=[1, 10],
            penalty_coef=-1,
            penalty_coef_type="group_lasso",
        )
