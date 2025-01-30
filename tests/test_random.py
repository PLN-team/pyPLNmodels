# pylint: skip-file
from pyPLNmodels import load_scrna


def test_threshold():
    data = load_scrna(n_samples=2000, dim=2000)
