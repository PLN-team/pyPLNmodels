import pandas as pd
from pyPLNmodels import PLN, PLNPCA
import pytest
from pytest_lazyfixture import lazy_fixture
import numpy as np

Y = pd.read_csv("../example_data/test_data/Y_test.csv")
covariates = pd.read_csv("../example_data/test_data/cov_test.csv")
O = pd.read_csv("../example_data/test_data/O_test.csv")
