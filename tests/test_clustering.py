# pylint: skip-file

import matplotlib.pyplot as plt

from pyPLNmodels import load_scrna, PlnLDA, plot_confusion_matrix


def test_confusion_matrix():
    data = load_scrna()
    lda = PlnLDA(data["endog"], clusters=data["labels"])
    lda.fit()
    pred = lda.predict_clusters(data["endog"])
    _, ax = plt.subplots()
    plot_confusion_matrix(pred, data["labels"])
    plot_confusion_matrix(pred, data["labels"], ax=ax)
