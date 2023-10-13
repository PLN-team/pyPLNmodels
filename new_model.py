from pyPLNmodels import PlnPCA, get_real_count_data, Pln


endog = get_real_count_data()
zi = Pln(endog, add_const=True)
zi.fit(nb_max_iteration=10)
