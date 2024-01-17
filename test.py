from pyPLNmodels import get_real_count_data, ZIPln, Pln
import matplotlib.pyplot as plt

data = get_real_count_data()
zi = ZIPln(data)
zi.fit()
print(zi)
zi.plot_expected_vs_true()
