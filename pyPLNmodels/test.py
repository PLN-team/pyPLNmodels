from VEM import PLNnoPCA, PLNPCA
import pandas as pd 
import torch
import numpy as np 
from IMPSPLNmodel import IMPSPLN
import matplotlib.pyplot as plt 



def lissage(Lx,Ly,p):
    '''Fonction qui débruite une courbe par une moyenne glissante
    sur 2P+1 points'''
    Lxout=[]
    Lyout=[]
    for i in range(p,len(Lx)-p):   
        Lxout.append(Lx[i])
    for i in range(p,len(Ly)-p):
        val=0
        for k in range(2*p):
            val+=Ly[i-p+k]
        Lyout.append(val/2/p)
            
    return Lxout,Lyout

Y = pd.read_csv('../example_data/Y_test')
O = np.exp(pd.read_csv('../example_data/O_test'))
covariates = pd.read_csv('../example_data/cov_test')
true_Sigma = torch.from_numpy(pd.read_csv('../example_data/true_Sigma_test').values)
true_beta = torch.from_numpy(pd.read_csv('../example_data/true_beta_test').values)
n = 20
#pln = PLNnoPCA()
#pln.fit(Y,O,covariates) 
#print(pln)
#plt.show()
#pca = PLNPCA(q = 5)
#pca.fit(Y,O,covariates, lr = 0.1)
#print(pca)
#plt.show()
nbMonteCarloSamplesBig = 1000
nbMonteCarloSamplesSmall = 25
nbEpochMax =100 
q = 2
window = 4 
bigImps= IMPSPLN(q=q)
bigImps.fit(Y.iloc[:n,:],O.iloc[:n,:],covariates.iloc[:n,:], nbEpochMax= nbEpochMax,criterionMax= 300, batchSize=n, nbMonteCarloSamples= nbMonteCarloSamplesBig, takeMoyenne = False, takeVariance = False)
#bigImpsMeanVar = IMPSPLN(q = q)
#bigImpsMeanVar.fit(Y.iloc[:n,:],O.iloc[:n,:],covariates.iloc[:n,:], nbEpochMax= nbEpochMax,criterionMax= 300, batchSize=n, nbMonteCarloSamples= nbMonteCarloSamplesBig, takeMoyenne = True, takeVariance = True)


bigImpsMean = IMPSPLN(q=q)
bigImpsMean.fit(Y.iloc[:n,:],O.iloc[:n,:],covariates.iloc[:n,:], nbEpochMax= nbEpochMax,criterionMax= 300, batchSize=n, nbMonteCarloSamples= nbMonteCarloSamplesBig, takeMoyenne = True, takeVariance = False)

#bigImpsVar = IMPSPLN(q = q)
#bigImpsVar.fit(Y.iloc[:n,:],O.iloc[:n,:],covariates.iloc[:n,:], nbEpochMax= nbEpochMax,criterionMax= 300, batchSize=n, nbMonteCarloSamples= nbMonteCarloSamplesBig, takeMoyenne = False, takeVariance = True)


lx,ly = lissage(np.arange(0, len(bigImps.listMeanESS)),1 -  np.array( bigImps.listMeanESS),window)
plt.plot(lx,ly,color = 'blue', label = '1 - ESS Moyen avec moyenne MG et variance deterministe')
#lx,ly = lissage(np.arange(0, len(bigImpsMeanVar.listMeanESS)),1- np.array( bigImpsMeanVar.listMeanESS),window)
#plt.plot(lx,ly,color = 'red', label = '1 - ESS Moyen avec moyenne IMPS et variance bigImps')
lx,ly = lissage(np.arange(0, len(bigImpsMean.listMeanESS)),1 -  np.array( bigImpsMean.listMeanESS),window)
plt.plot(lx,ly,color = 'green', label = '1 - ESS Moyen avec moyenne IMPS et variance deterministe')
#lx,ly = lissage(np.arange(0, len(bigImpsVar.listMeanESS)),1- np.array( bigImpsVar.listMeanESS),window)
#plt.plot(lx,ly,color = 'black', label = '1 - ESS Moyen avec moyenne MG et variance IMPS')

#smallImpsMeanVar = IMPSPLN(q = q)
#smallImpsMeanVar.fit(Y.iloc[:n,:],O.iloc[:n,:],covariates.iloc[:n,:], nbEpochMax= nbEpochMax,criterionMax= 300, batchSize=n, nbMonteCarloSamples= nbMonteCarloSamplesSmall, takeMoyenne = True, takeVariance = True)


smallImpsMean = IMPSPLN(q=q)
smallImpsMean.fit(Y.iloc[:n,:],O.iloc[:n,:],covariates.iloc[:n,:], nbEpochMax= nbEpochMax,criterionMax= 300, batchSize=n, nbMonteCarloSamples= nbMonteCarloSamplesSmall, takeMoyenne = True, takeVariance = False)

#smallImpsVar = IMPSPLN(q = q)
#smallImpsVar.fit(Y.iloc[:n,:],O.iloc[:n,:],covariates.iloc[:n,:], nbEpochMax= nbEpochMax,criterionMax= 300, batchSize=n, nbMonteCarloSamples= nbMonteCarloSamplesSmall, takeMoyenne = False, takeVariance = True)


smallImps= IMPSPLN(q=q)
smallImps.fit(Y.iloc[:n,:],O.iloc[:n,:],covariates.iloc[:n,:], nbEpochMax= nbEpochMax,criterionMax= 300, batchSize=n, nbMonteCarloSamples= nbMonteCarloSamplesSmall, takeMoyenne = False, takeVariance = False)

lx,ly =lissage (np.arange(0, len(smallImps.listMeanESS)),1 -  np.array( smallImps.listMeanESS),window)
plt.plot(lx,ly,color = 'blue',linestyle = '--')
#lx,ly =lissage (np.arange(0, len(smallImpsMeanVar.listMeanESS)),1- np.array( smallImpsMeanVar.listMeanESS),window)
#plt.plot(lx,ly,color = 'red', linestyle = '--')
lx,ly = lissage(np.arange(0, len(smallImpsMean.listMeanESS)),1 -  np.array( smallImpsMean.listMeanESS),window)
plt.plot(lx,ly,color = 'green',linestyle = '--')
#lx,ly = lissage(np.arange(0, len(smallImpsVar.listMeanESS)),1- np.array( smallImpsVar.listMeanESS),window)
#plt.plot(lx,ly, color = 'black',linestyle = '--')





plt.yscale('log')
plt.xlabel('Iteration number')
plt.ylabel('1 - ESS normalisé moyen')
plt.title('n = 20, dimension de lespace latent q = 20, pointille=25 particules, plein=1000 particules')
plt.legend()
plt.legend()
plt.show()  
##print(bigImpsMeanVar)