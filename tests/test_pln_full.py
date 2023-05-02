import torch

from import_fixtures_and_data import get_dict_fixtures
from pyPLNmodels import PLN


df = get_dict_fixtures(PLN)
for key, fixture in df.items():
    print(len(fixture))
