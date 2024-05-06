from pyPLNmodels import load_microcosm


def test_microcosm():
    data = load_microcosm(min_perc=0, remove_useless=False)
    assert data["endog"].shape == (300, 200)
    data = load_microcosm(min_perc=0, remove_useless=True)

    data = load_microcosm(n_samples=50, dim=12, min_perc=0, remove_useless=False)
    assert data["endog"].shape == (50, 12)
    data = load_microcosm(get_affil=True)
    assert data["affiliations"] is not None
