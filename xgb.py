import scanpy
from pyPLNmodels.models import Pln, ZIPln
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt

from xgboost import XGBClassifier


from sklearn.model_selection import cross_val_score

fig, axes = plt.subplots(3,1, figsize = (20,20))

def get_sc_mark_data(max_class=28, max_n=200, dim=100, to_begin_with = None):
    data = scanpy.read_h5ad("2k_cell_per_study_10studies.h5ad")
    Y = data.X.toarray()
    GT_name = data.obs["standard_true_celltype_v5"]
    le = LabelEncoder()
    GT = le.fit_transform(GT_name)
    filter = GT < max_class
    unique, index = np.unique(GT, return_counts=True)
    enough_elem = index > 15
    classes_with_enough_elem = unique[enough_elem]
    filter_bis = np.isin(GT, classes_with_enough_elem)
    mask = filter * filter_bis
    GT = GT[mask]
    GT_name = GT_name[mask]
    Y = Y[mask]
    GT = le.fit_transform(GT)
    not_only_zeros = np.sum(Y, axis=0) > 0
    Y = Y[:, not_only_zeros]
    Y = Y[:, :dim]
    nb_non_zero = np.sum(Y>0, axis=1)
    if to_begin_with is not None:
        # most_non_zeros = np.argsort(nb_non_zero)[11200:(11200+dim)]
        most_non_zeros = np.argsort(nb_non_zero)[to_begin_with:(to_begin_with + max_n)]
    else:
        most_non_zeros = np.argsort(nb_non_zero)[-max_n:]
    Y = Y[most_non_zeros]
    GT = GT[most_non_zeros]
    GT_name = GT_name[most_non_zeros]
    le = LabelEncoder()
    other = le.fit_transform(GT_name)
    GT_bis = le.fit_transform(GT)
    print('gt_bis unique', np.unique(GT_bis))
    print('nb_non_zero', np.sum(Y>0, axis = 1))
    return Y, GT_bis, list(GT_name.values.__array__())

def launch_test():
    pass

zi_closed_multi_list = []
zi_closed_single_list = []
zi_free_multi_list = []
zi_free_single_list = []
zi_closed_n_list = []
zi_free_n_list = []
zi_closed_np_list = []
zi_free_np_list = []
pln_list = []



cv = 2
nb_iter = 150
p = 200
# ps = [50, 75]#,100,125,150,200,300, 400, 500, 600, 700, 800, 900, 1000]
to_begins_with = [None, 19000]
n = 1000
def get_all_scores(n,p, to_begin_with):
    Y, GT_, _ = get_sc_mark_data(max_n=n, dim=p, to_begin_with = to_begin_with)
    print("percentage of zeros", np.sum(Y == 0) / np.prod(Y.shape))
    print(Y.shape)

    tol = 0.00

    def get_and_append_score(model, is_inflated, list_score, ground_truth):
        model.fit(tol = tol, nb_max_iteration = nb_iter)
        if is_inflated:
            Z, _ = model.transform()
        else:
            Z = model.transform()
        xgb = XGBClassifier()
        le = LabelEncoder()
        ground_truth = le.fit_transform(ground_truth)
        print('unique ground truth', np.unique(ground_truth))
        score = cross_val_score(xgb, Z, ground_truth, cv=cv, scoring="balanced_accuracy")
        list_score.append(np.mean(score))

    zi_closed_single = ZIPln(Y, use_closed_form_prob=True, do_single_inflation = True)
    zi_closed_multi = ZIPln(Y, use_closed_form_prob=True)
    zi_free_single = ZIPln(Y, use_closed_form_prob=False, do_single_inflation = True)
    zi_free_multi = ZIPln(Y, use_closed_form_prob=False)
    zi_closed_n = ZIPln(Y, use_closed_form_prob = True, do_n_inflation = True)
    zi_free_n = ZIPln(Y, use_closed_form_prob = False, do_n_inflation = True)
    zi_closed_np = ZIPln(Y, use_closed_form_prob = True, do_np_inflation = True)
    zi_free_np = ZIPln(Y, use_closed_form_prob = False, do_np_inflation = True)
    pln = Pln(Y)

    get_and_append_score(zi_closed_single, True, zi_closed_single_list, GT_)
    get_and_append_score(zi_closed_multi, True, zi_closed_multi_list)
    get_and_append_score(zi_free_multi, True, zi_free_multi_list)
    get_and_append_score(zi_free_single, True, zi_free_single_list)
    get_and_append_score(pln, False, pln_list)
    get_and_append_score(zi_closed_n, True, zi_closed_n_list)
    get_and_append_score(zi_free_n, True, zi_free_n_list)
    get_and_append_score(zi_closed_np, True, zi_closed_np_list)
    get_and_append_score(zi_free_np, True, zi_free_np_list)


for to_begin_with in to_begins_with:
    get_all_scores(n,p, to_begin_with)

fig = plt.figure(figsize = (20,20))
linewidth = 2
plt.title(f"Comparison of scores on scMark dataset with n={n}, d=1 with few zeros")

plt.plot(ps, zi_closed_multi_list, label = "ZI Closed multi", color = "red", linewidth = linewidth)
plt.plot(ps, zi_free_multi_list, label = "ZI free multi", color = "red", linestyle = '--', linewidth = linewidth)

plt.plot(ps, zi_closed_single_list, label = "ZI Closed single", color = "green", linestyle = "solid", linewidth = linewidth)
plt.plot(ps, zi_free_single_list, label = "ZI free single", color = "green", linestyle = '--', linewidth = linewidth)

plt.plot(ps, zi_closed_n_list, label = "ZI Closed n", color = "black", linestyle = "solid", linewidth = linewidth)
plt.plot(ps, zi_free_n_list, label = "ZI free n", color = "black", linestyle = "--", linewidth = linewidth)

plt.plot(ps, zi_closed_np_list, label = "ZI Closed np", color = "orange", linestyle = "solid", linewidth = linewidth)
plt.plot(ps, zi_free_np_list, label = "ZI free np", color = "orange", linestyle = "--", linewidth = linewidth)


plt.plot(ps, pln_list, label = "pln", color = "blue", linestyle = "dotted", linewidth = linewidth)



plt.legend()
plt.xlabel("Number of dimensions", fontsize = 20)
plt.ylabel(f"Xgb score with {cv} crossval", fontsize = 20)
plt.savefig("xgb_pln_very_few_zeros.pdf", format = "pdf")
plt.show()



