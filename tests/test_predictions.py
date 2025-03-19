# pylint: skip-file
import torch
from pyPLNmodels import (
    PlnMixture,
    PlnMixtureSampler,
    get_label_mapping,
    plot_confusion_matrix,
    PlnLDA,
    PlnLDASampler,
)


def test_right_prediction_and_confusion_matrix():
    ntrain = 1000
    ntest = 200
    for nb_cov in [0, 1]:
        sampler = PlnMixtureSampler(
            nb_cov=nb_cov, n_samples=ntrain + ntest, dim=100, n_cluster=3
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

        mixt = PlnMixture(endog_train, exog=exog_train, n_cluster=sampler.n_cluster)
        mixt.fit()

        clusters_pred = mixt.predict_clusters(endog_test, exog=exog_test)
        true_clusters = sampler.clusters[ntrain:]

        label_mapping = get_label_mapping(clusters_pred, true_clusters)

        for i, _ in enumerate(clusters_pred):
            clusters_pred[i] = label_mapping[clusters_pred[i]]
        plot_confusion_matrix(clusters_pred, true_clusters)
        assert (
            torch.mean(
                (torch.tensor(true_clusters) == torch.tensor(clusters_pred)).float()
            )
            > 0.85
        )


def test_lda_right_prediction():
    ntrain, ntest = 1000, 200
    n_cluster = 4
    for nb_cov in [0, 1]:
        sampler = PlnLDASampler(
            n_samples=ntrain + ntest,
            nb_cov=nb_cov,
            n_cluster=n_cluster,
            add_const=False,
            dim=300,
        )
        endog = sampler.sample()
        known_exog = sampler.known_exog
        clusters = sampler.clusters
        endog_train, endog_test = endog[:ntrain], endog[ntrain:]
        if nb_cov > 0:
            known_exog_train, known_exog_test = known_exog[:ntrain], known_exog[ntrain:]
        else:
            known_exog_train, known_exog_test = None, None
        clusters_train, clusters_test = clusters[:ntrain], clusters[ntrain:]
        lda = PlnLDA(
            endog_train, clusters=clusters_train, exog=known_exog_train, add_const=False
        ).fit()
        pred = lda.predict_clusters(endog_test, exog=known_exog_test)
        mean_right_pred = torch.mean((torch.tensor(pred) == clusters_test).float())
        assert mean_right_pred > 0.9
