import pandas as pd 
import torch 
from IMPSPLNmodel import IMPSPLN
import matplotlib.pyplot as plt 
from utils import lissage 



Y = pd.read_csv("../example_data/Y_test")
O = np.exp(pd.read_csv("../example_data/O_test"))
covariates = pd.read_csv("../example_data/cov_test")
true_Sigma = torch.from_numpy(pd.read_csv("../example_data/true_Sigma_test").values)
true_beta = torch.from_numpy(pd.read_csv("../example_data/true_beta_test").values)
n = 20


