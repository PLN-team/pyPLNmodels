# pylint: skip-file
import torch
from pyPLNmodels import (
    PlnMixture,
    PlnMixtureSampler,
    get_label_mapping,
    get_confusion_matrix,
    plot_confusion_matrix,
)


def test_right_prediction_and_confusion_matrix():
    ntrain = 1000
    ntest = 200
    for nb_cov in [0, 1]:
        sampler = PlnMixtureSampler(
            nb_cov=nb_cov, n_samples=ntrain + ntest, dim=100, n_clusters=3
        )

        endog = sampler.sample()
        endog_train = endog[:ntrain]
        endog_test = endog[ntrain:]

        if nb_cov > 0:
            exog_train = sampler.exog[:ntrain]
            exog_test = sampler.exog[ntrain:]
        else:
            exog_train = None
            exog_test = None

        mixt = PlnMixture(endog_train, exog=exog_train, n_clusters=sampler.n_clusters)
        mixt.fit()

        clusters_pred = mixt.predict_clusters(endog_test, exog=exog_test)
        true_clusters = sampler.clusters[ntrain:]

        label_mapping = get_label_mapping(clusters_pred, true_clusters)

        for i in enumerate(clusters_pred):
            clusters_pred[i] = label_mapping[clusters_pred[i]]
        confusion_matrix = get_confusion_matrix(clusters_pred, true_clusters)
        plot_confusion_matrix(confusion_matrix)
        assert (
            torch.mean(
                (torch.tensor(true_clusters) == torch.tensor(clusters_pred)).float()
            )
            > 0.9
        )
