from pyPLNmodels import load_microcosm


def test_microcosm():
    endog, exog = load_microcosm()
    assert endog.shape == (300, 200)
    assert exog.shape == (300, 4)

    endog, exog = load_microcosm(n_samples=50, dim=12)
    assert endog.shape == (50, 12)
    assert exog.shape == (50, 3)


def test_microcosm_with_affil():
    endog, exog, affil = load_microcosm(get_affil=True)


def test_microcosm_formula():
    data = load_microcosm(for_formula=True)
