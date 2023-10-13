from pyPLNmodels import ZIPln, get_real_count_data


endog = get_real_count_data()
zi = ZIPln(endog, add_const = True)
zi.fit(nb_max_iteration = 10)
zi.show()


