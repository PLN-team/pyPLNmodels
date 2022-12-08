from VEM import PLNnoPCA, PLNPCA
import pandas as pd
import torch
import numpy as np
from IMPSPLNmodel import IMPSPLN
import matplotlib.pyplot as plt
from utils import lissage


Y = pd.read_csv("../example_data/Y_test")
O = np.exp(pd.read_csv("../example_data/O_test"))
covariates = pd.read_csv("../example_data/cov_test")
true_Sigma = torch.from_numpy(pd.read_csv("../example_data/true_Sigma_test").values)
true_beta = torch.from_numpy(pd.read_csv("../example_data/true_beta_test").values)
n = 20


nbMonteCarloSamplesBig = 1000
nbMonteCarloSamplesSmall = 25
nbEpochMax = 25
q = 10
lr = 0.1
window = 3
doLissage = True
nbCriterionMax = 30


fig, axes = plt.subplots(2,1, figsize = (20,10))




bigBothImps = IMPSPLN(q=q)
bigBothImps.fit(
    Y.iloc[:n, :],
    O.iloc[:n, :],
    covariates.iloc[:n, :],
    nbEpochMax=nbEpochMax,
    criterionMax=nbCriterionMax,
    batchSize=n,
    nbMonteCarloSamples=nbMonteCarloSamplesBig,
    method="both",
    lr=lr,
)
lx, ly = np.arange(0, len(bigBothImps.listMeanESS)), 1 - np.array(bigBothImps.listMeanESS)
if doLissage:
    lx, ly = lissage(lx, ly, window)
label = "moyenne IMPS+MG  et variance deterministe,1000 particules"
axes[0].plot(
    lx, ly, color="black", label=label
)

toPlot = bigBothImps.listDiffGradRecycling 
lxDiff, lyDiff = np.arange(0,len(toPlot)), toPlot
if doLissage: 
    lxDiff, lyDiff = lissage(lxDiff, lyDiff, window)
axes[1].plot(lxDiff, lyDiff, color = 'black', label= label)

print('first lx,ly', lx,ly)
print('secon lx,ly', lxDiff,lyDiff)





smallBothImps = IMPSPLN(q=q)
smallBothImps.fit(
    Y.iloc[:n, :],
    O.iloc[:n, :],
    covariates.iloc[:n, :],
    nbEpochMax=nbEpochMax,
    criterionMax=nbCriterionMax,
    batchSize=n,
    nbMonteCarloSamples=nbMonteCarloSamplesSmall,
    method="both",
    lr=lr,
)
lx, ly = np.arange(0, len(smallBothImps.listMeanESS)), 1 - np.array(smallBothImps.listMeanESS)
if doLissage:
    lx, ly = lissage(lx, ly, window)
label = "moyenne IMPS+MG  et variance deterministe,25 particules"
axes[0].plot(
    lx, ly, color="black", label=label, linestyle = '--'
)

toPlot = smallBothImps.listDiffGradRecycling 
lxDiff, lyDiff = np.arange(0,len(toPlot)), toPlot
if doLissage: 
    lxDiff, lyDiff = lissage(lxDiff, lyDiff, window)
axes[1].plot(lxDiff, lyDiff, color = 'black', label= label, linestyle = '--')

print('first lx,ly', lx,ly)
print('secon lx,ly', lxDiff,lyDiff)





bigImps = IMPSPLN(q=q)
bigImps.fit(
    Y.iloc[:n, :],
    O.iloc[:n, :],
    covariates.iloc[:n, :],
    nbEpochMax=nbEpochMax,
    criterionMax=nbCriterionMax,
    batchSize=n,
    nbMonteCarloSamples=nbMonteCarloSamplesBig,
    method="gradient",
    lr=lr,
)
lx, ly = np.arange(0, len(bigImps.listMeanESS)), 1 - np.array(bigImps.listMeanESS)
if doLissage:
    lx, ly = lissage(lx, ly, window)

axes[0].plot(
    lx, ly, color="blue", label="moyenne MG et variance deterministe,1000 particules"
)

smallImps = IMPSPLN(q=q)
smallImps.fit(
    Y.iloc[:n, :],
    O.iloc[:n, :],
    covariates.iloc[:n, :],
    nbEpochMax=nbEpochMax,
    criterionMax=nbCriterionMax,
    batchSize=n,
    nbMonteCarloSamples=nbMonteCarloSamplesSmall,
    method="gradient",
    lr=lr,
)
lx, ly = np.arange(0, len(smallImps.listMeanESS)), 1 - np.array(smallImps.listMeanESS)
if doLissage:
    lx, ly = lissage(lx, ly, window)
axes[0].plot(lx, ly, color="blue", linestyle="--", label = "moyenne MG et variance deterministe, 25 particules")

bigImpsMeanCurrent = IMPSPLN(q=q)
bigImpsMeanCurrent.fit(
    Y.iloc[:n, :],
    O.iloc[:n, :],
    covariates.iloc[:n, :],
    nbEpochMax=nbEpochMax,
    criterionMax=nbCriterionMax,
    batchSize=n,
    nbMonteCarloSamples=nbMonteCarloSamplesBig,
    method="recycling",
    lr=lr,
)

lx, ly = (
    np.arange(0, len(bigImpsMeanCurrent.listMeanESS)),
    1 - np.array(bigImpsMeanCurrent.listMeanESS),
)
if doLissage:
    lx, ly = lissage(lx, ly, window)


label = r"moyenne IMPS et variance deterministe, $E_{\theta_{t}}$,1000 particules"
axes[0].plot(
    lx,
    ly,
    color="green",
    label=label,
)
toPlot = bigImpsMeanCurrent.listDiffGradRecycling 
lxDiff, lyDiff = np.arange(0,len(toPlot)), toPlot
if doLissage: 
    lxDiff, lyDiff = lissage(lxDiff, lyDiff, window)
axes[1].plot(lxDiff, lyDiff, color = 'green', label= label)

print('first lx,ly', lx,ly)
print('secon lx,ly', lxDiff,lyDiff)


smallImpsMeanCurrent = IMPSPLN(q=q)
smallImpsMeanCurrent.fit(
    Y.iloc[:n, :],
    O.iloc[:n, :],
    covariates.iloc[:n, :],
    nbEpochMax=nbEpochMax,
    criterionMax=nbCriterionMax,
    batchSize=n,
    nbMonteCarloSamples=nbMonteCarloSamplesSmall,
    method="recycling",
    lr=lr,
)

lx, ly = (
    np.arange(0, len(smallImpsMeanCurrent.listMeanESS)),
    1 - np.array(smallImpsMeanCurrent.listMeanESS),
)

if doLissage:
    lx, ly = lissage(lx, ly, window)
label = r"moyenne IMPS et variance deterministe, $E_{\theta_{t}}$,25 particules"
axes[0].plot(lx, ly, color="green", linestyle="--", label = label)



toPlot = smallImpsMeanCurrent.listDiffGradRecycling 
lxDiff, lyDiff = np.arange(0,len(toPlot)), toPlot
if doLissage: 
    lxDiff, lyDiff = lissage(lxDiff, lyDiff, window)
axes[1].plot(lxDiff, lyDiff, color = 'green', label= label, linestyle = "--")

print('first lx,ly', lx,ly)
print('secon lx,ly', lxDiff,lyDiff)



smallImpsMeanPrevious = IMPSPLN(q=q, current=False)
smallImpsMeanPrevious.fit(
    Y.iloc[:n, :],
    O.iloc[:n, :],
    covariates.iloc[:n, :],
    nbEpochMax=nbEpochMax,
    criterionMax=nbCriterionMax,
    batchSize=n,
    nbMonteCarloSamples=nbMonteCarloSamplesSmall,
    method="recycling",
    lr=lr,
)

lx, ly = (
    np.arange(0, len(smallImpsMeanPrevious.listMeanESS)),
    1 - np.array(smallImpsMeanPrevious.listMeanESS),
)
if doLissage:
    lx, ly = lissage(lx, ly, window)
axes[0].plot(lx, ly, color="red", linestyle="--", label = r"moyenne IMPS et variance deterministe, $E_{\theta_{t-1}}$, 25 particules")



bigImpsMeanPrevious = IMPSPLN(q=q, current=False)
bigImpsMeanPrevious.fit(
    Y.iloc[:n, :],
    O.iloc[:n, :],
    covariates.iloc[:n, :],
    nbEpochMax=nbEpochMax,
    criterionMax=nbCriterionMax,
    batchSize=n,
    nbMonteCarloSamples=nbMonteCarloSamplesBig,
    method="recycling",
    lr=lr,
)

lx, ly = (
    np.arange(0, len(bigImpsMeanPrevious.listMeanESS)),
    1 - np.array(bigImpsMeanPrevious.listMeanESS),
)
if doLissage:
    lx, ly = lissage(lx, ly, window)
axes[0].plot(lx, ly, color="red", label = r"moyenne IMPS et variance deterministe, $E_{\theta_{t-1}}$, 1000 particules")




axes[0].set_yscale("log")
axes[0].set_xlabel("Iteration number")
axes[0].set_ylabel("1 - ESS normalisé moyen")
axes[0].set_title("n = 20, dimension latent q =" + str(q))
axes[0].legend()
axes[1].legend()
axes[1].set_xlabel("Iteration number")
axes[1].set_ylabel(r"||argmax - $ E_{\theta_t}[W|Y]||_1$")
axes[0].set_ylabel("Difference with mode")
plt.show()
