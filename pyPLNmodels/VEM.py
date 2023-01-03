import os
from elbos import ELBOnoPCA, ELBOPCA, ELBOZI
from closedForms import closedBeta, closedSigma, closedPi
from abc import ABC, abstractmethod
import torch
import pandas as pd
import numpy as np
from utils import PLNPlotArgs, poissonReg, initSigma, initC, initBeta
import time
import seaborn as sns
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

print('device:', device)

# shoudl add a good init for M. for plnnopca we should not put the maximum of the log posterior, for plnpca it may be ok.


class PLN():
    def __init__(self):
        self.window = 3
        self.fitted = False

    def formatDatas(self, Y, O, covariates):
        self.Y = self.formatData(Y)
        self.O = self.formatData(O)
        self.covariates = self.formatData(covariates)

    def goodInitModelParameters(self):
        self.beta = initBeta(self.Y, self.O, self.covariates)

    def randomInitModelParameters(self):
        self.beta = torch.randn((self.d, self.p), device=device)

    def formatData(self, data):
        if isinstance(data, pd.DataFrame):
            return torch.from_numpy(data.values).float().to(device)
        elif isinstance(data, np.array):
            return torch.from_numpy(data.values).float().to(device)
        elif isinstance(data, torch.tensor):
            return data
        else:
            raise Exception(
                'Please insert either a numpy array, pandas.DataFrame or torch.tensor'
            )

    def initFromData(self, Y, O, covariates, doGoodInit):
        self.formatDatas(Y, O, covariates)
        self.n, self.p = self.Y.shape
        self.d = self.covariates.shape[1]
        print('Initialization ...')
        if doGoodInit:
            self.goodInitModelParameters()
        else:
            self.randomInitModelParameters()
        self.basicInitVarParameters()
        print('Initialization finished')
        self.putParametersToDevice()


    def putParametersToDevice(self):
        for parameter in self.parametersNeddingGradientUpdateList:
            parameter.requires_grad_(True)

    @property
    @abstractmethod
    def parametersNeddingGradientUpdateList(self):
        pass

    @abstractmethod
    def basicInitVarParameters(self):
        pass

    def fit(self,
            Y,
            O,
            covariates,
            nbIterationMax=15000,
            lr=0.01,
            classOptimizer=torch.optim.Rprop,
            tol=1e-3,
            doGoodInit=True,
            verbose=False):
        self.t0 = time.time()
        if self.fitted == False:
            self.plotargs = PLNPlotArgs(self.window)
            self.initFromData(Y, O, covariates, doGoodInit)
        optim = classOptimizer(self.parametersNeddingGradientUpdateList, lr=lr)
        nbIterationDone = 0
        stopCondition = False
        while nbIterationDone < nbIterationMax and stopCondition == False:
            nbIterationDone += 1
            optim.zero_grad()
            loss = -self.computeELBO()
            loss.backward()
            optim.step()
            self.updateClosedForms()
            delta = self.computeCriterionAndUpdatePlotArgs(loss, tol)
            if abs(delta) < tol:
                stopCondition = True
            if verbose:
                self.printIterationStats()
        self.printEndFitMessage(stopCondition, tol)
        self.fitted = True

    def printEndFitMessage(self, stopCondition, tol):
        if stopCondition:
            print('Tolerance {} reached in {} iterations'.format(
                tol, self.plotargs.length))
        else:
            print('Maximum number of iterations reached : ',
                  self.plotargs.length, 'last delta = ',
                  np.round(self.plotargs.deltas[-1], 8))

    def printIterationStats(self):
        print('-------UPDATE-------')
        print('Iteration number: ', self.plotargs.length)
        print('Delta: ', np.round(self.plotargs.deltas[-1], 8))
        print('ELBO:', np.round(self.plotargs.normalizedELBOsList[-1], 6))

    def computeCriterionAndUpdatePlotArgs(self, loss, tol):
        self.plotargs.normalizedELBOsList.append(-loss.item() / self.n)
        self.plotargs.runningTimes.append(time.time() - self.t0)
        if self.plotargs.length > self.window:
            delta = abs(self.plotargs.normalizedELBOsList[-1] -
                        self.plotargs.normalizedELBOsList[-1 - self.window])
            self.plotargs.deltas.append(delta)
            return delta
        else:
            return tol

    def updateClosedForms(self):
        pass

    @abstractmethod
    def computeELBO(self):
        pass

    def showSigma(self, ax=None, savefig=False, name_doss=''):
        '''Displays Sigma
        args:
            'ax': AxesSubplot object. Sigma will be displayed in this ax
                if not None. If None, will simply create an axis. Default is None.
            'name_doss': str. The name of the file the graphic will be saved to.
                Default is 'fastPLNPCA_Sigma'.
        returns: None but displays Sigma.
        '''
        fig = plt.figure()
        sigma = self.getSigma()
        if self.p > 400:
            sigma = sigma[:400, :400]
        sns.heatmap(sigma, ax=ax)
        if savefig:
            plt.savefig(name_doss + self.NAME)
        plt.close()  # to avoid displaying a blanck screen

    def __str__(self):
        print('Best likelihood:', -self.plotargs.normalizedELBOsList[-1])
        fig, axes = plt.subplots(1, 3, figsize=(23, 5))
        self.plotargs.showLoss(ax=axes[0])
        self.plotargs.showCriterion(ax=axes[1])
        self.showSigma(ax=axes[2])
        plt.show()
        return ''

    @abstractmethod
    def getSigma(self):
        pass


class PLNnoPCA(PLN):
    NAME = 'PLNnoPCA'

    def goodInitModelParameters(self):
        super().goodInitModelParameters()
        self.Sigma = initSigma(self.Y, self.O, self.covariates, self.beta)

    def randomInitModelParameters(self):
        super().randomInitModelParameters()
        self.Sigma = torch.diag(torch.ones(self.p)).to(device)

    def basicInitVarParameters(self):
        self.S = 1 / 2 * torch.ones((self.n, self.p)).to(device)
        self.M = torch.ones((self.n, self.p)).to(device)

    @property
    def parametersNeddingGradientUpdateList(self):
        return [self.M, self.S]

    def computeELBO(self):
        return ELBOnoPCA(self.Y, self.O, self.covariates, self.M, self.S,
                         self.Sigma, self.beta)

    def updateClosedForms(self):
        self.beta = closedBeta(self.covariates, self.M)
        self.Sigma = closedSigma(self.covariates, self.M, self.S, self.beta,
                                 self.n)

    def getSigma(self):
        return self.Sigma.detach().cpu()


class PLNPCA(PLN):
    NAME = 'PLNPCA'

    def __init__(self, q):
        super().__init__()
        self.q = q

    def goodInitModelParameters(self):
        super().goodInitModelParameters()
        self.C = initC(self.Y, self.O, self.covariates, self.beta, self.q)

    def randomInitModelParameters(self):
        super().randomInitModelParameters()
        self.C = torch.randn((self.d, self.q)).to(device)

    def basicInitVarParameters(self):
        self.S = 1 / 2 * torch.ones((self.n, self.q)).to(device)
        self.M = torch.ones((self.n, self.q)).to(device)

    @property
    def parametersNeddingGradientUpdateList(self):
        return [self.C, self.beta, self.M, self.S]

    def computeELBO(self):
        return ELBOPCA(self.Y, self.O, self.covariates, self.M, self.S, self.C,
                       self.beta)

    def getSigma(self):
        return (self.C @ (self.C.T)).detach().cpu()


class ZIPLN(PLN):
    NAME = 'ZIPLN'

    def randomInitModelParameters(self):
        super().randomInitModelParameters()
        self.ThetaZero = torch.randn(self.d, self.p)
        self.Sigma = torch.diag(torch.ones(self.p)).to(device)

    # should change the good initialization, especially for ThetaZero
    def goodInitModelParameters(self):
        super().goodInitModelParameters()
        self.Sigma = initSigma(self.Y, self.O, self.covariates, self.beta)
        self.ThetaZero = torch.randn(self.d, self.p)

    def basicInitVarParameters(self):
        self.dirac = (self.Y == 0)
        self.M = torch.randn(self.n, self.p)
        self.S = torch.randn(self.n, self.p)
        self.pi = torch.empty(self.n, self.p).uniform_(
            0, 1).to(device)*self.dirac

    def computeELBO(self):
        return ELBOZI(self.Y, self.O, self.covariates, self.M, self.S, self.Sigma, self.beta, self.pi, self.ThetaZero, self.dirac)

    def getSigma(self):
        return self.Sigma.detach().cpu()


    @property
    def parametersNeddingGradientUpdateList(self):
        return [self.M, self.S,  self.ThetaZero]


    def updateClosedForms(self):
        self.beta = closedBeta(self.covariates, self.M)
        self.Sigma = closedSigma(self.covariates, self.M, self.S, self.beta,
                                 self.n)
        self.pi = closedPi(self.O, self.M, self.S, self.dirac, self.covariates, self.ThetaZero)








