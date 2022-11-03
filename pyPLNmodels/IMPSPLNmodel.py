import torch 
import pandas as pd 
import numpy as np 
from utils import sampleGaussians, logGaussianDensity, initC, initBeta, batchLogPWgivenY, plotList
import matplotlib.pyplot as plt 
import time 
from tqdm import tqdm 
from VRA import SAGA 
import seaborn as sns 
import torch.linalg as TLA 
if torch.cuda.is_available(): 
    device = 'cuda'
else: 
    device = 'cpu'

## pb avec batch et per sample. tres tres ambigu. 
## should change the criterion. I increment it two times. 


class IMPSPLN():
    '''Maximize the likelihood of the PLN-PCA model. The main method
    is the fit() method that fits the model. Most of the others
    functions are here to support the fit method. Any value of n can be taken.
    However, q should not be greater than 40. The greater q, the
    lower the accuracy parameter should be taken.
    '''
    NAME = 'IMPSPLN'


    def __init__(self, q, nbAverageParam=100, nbAverageLikelihood=8):
        '''Init method. Set some global parameters of the class, such as
        the dimension of the latent space and the number of elements took
        to set an average parameter that should be more accurate.

        Args :
            q : int. The dimension of the latent layer of the PLN-PCA model.
            nbAverageParam: int, optional. Will average the parameter to get
                parameters with lower variance. nbAverageParam tells
                the number of parameter took to build the mean. Should
                not be changed since not very important. Default is 100.
            nbAverageLikelihood: int, optional. Will average the logLikelihood
                of the model. nbAverage likelihood tells the number of
                likelihood took to build the mean likelihood. Should
                not be changed since not very important. Note that this
                parameter is actually changing the algorithm (just a little bit)
                since the stopping criterion depends directly on the average likelihood.
                Default is 8.
        Returns:
            An IMPSPLN object.
        '''
        self.q = q  # dimension of the latent space
        self.nbAverageLikelihood = nbAverageLikelihood
        self.nbAverageParam = nbAverageParam
        self.fitted = False

    def formatDatas(self, Y,O,covariates):
        self.Y = self.formatData(Y)
        self.O = self.formatData(O)
        self.covariates = self.formatData(covariates)

    def formatData(self,data): 
        if isinstance(data, pd.DataFrame): 
            return torch.from_numpy(data.values).float().to(device)
        elif isinstance(data, np.array): 
            return  torch.from_numpy(data).float().to(device)
        elif isinstance(data, torch.tensor): 
            return data.float() 
        else: 
            raise Exception('Please insert either a numpy array, pandas.DataFrame or torch.tensor')

    def initUsefulListsAndTensors(self): 
        self.CriterionCounterList = [0] * self.nbPlateauBeforeIncrementingCritertion  # counter for the criterion
        self.criterionList = [0]  # store the criterion to plot it after.
        self.timeToFindModeList = list()  # store the time took to find the mode at each iteration
        # store the time took to estimate the gradiens.
        self.timeToEstimGradList = list()
       # Initialize some lists
        self.runningTimes = list()  # store the running times for a nice plot
        self.logLikelihoodList = list()  # store the likelihood to plot it after
        self.lastLikelihoods = list()  # store the last likelihoods in order to take
        # the mean of those likelihoods to smooth it.
        # iteration we have done
        # Tensor that will store the starting point for the
        # gradient descent finding the mode for IMPS.
        self.startingPoint = torch.zeros(self.n, self.q, device=device, requires_grad=False)
        self.modeStepSizes = torch.zeros((self.n, self.q), device=device) + 0.3

        # Initialization of the last beta and C.
        self.lastBetas = [self.beta]*self.nbAverageParam
        self.lastCs = [self.C]* self.nbAverageParam         

    @property 
    def betaMean(self): 
        return torch.mean(torch.stack(self.lastBetas), axis = 0)

    @property 
    def CMean(self): 
        return torch.mean(torch.stack(self.lastCs), axis = 0)

    def goodInitModelParameters(self): 
        self.beta = initBeta(self.Y, self.O, self.covariates)
        self.C = initC(self.Y,self.O,self.covariates,self.beta,self.q).to(device)

        
    def initFromData(self, Y, O, covariates, doGoodInit):
        self.formatDatas(Y,O,covariates)
        self.n, self.p = self.Y.shape
        self.d = self.covariates.shape[1]
        print('Initialization ...')
        if doGoodInit:
            self.goodInitModelParameters()
        else:   
            self.randomInitModelParameters()
        print('Intialization done.')

    def randomInitModelParameters(self): 
        self.beta = torch.randn(self.d, self.p)
        self.C = torch.randn(self.p, self.q)

    def getBatch(self, batchSize):
        '''Get the batches required to do a  minibatch gradient ascent.

        Args:
            batchSize: int. The batch size. Should be lower than n.

        Returns: A generator. Will generate n//batchSize + 1 batches of
            size batchSize (except the last one since the rest of the
            division is not always 0)
        '''
        indices = np.arange(self.n)
        #np.random.shuffle(indices)
        self.batchSize = batchSize
        nbFullBatch, lastBatchSize = self.n // batchSize, self.n % batchSize
        for i in range(nbFullBatch):
            yield (self.Y[indices[i * batchSize: (i + 1) * batchSize]],
                   self.covariates[indices[i * batchSize: (i + 1) * batchSize]],
                   self.O[indices[i * batchSize: (i + 1) * batchSize]],
                    indices[i * batchSize: (i + 1) * batchSize]
                   )
        # Last batch
        if lastBatchSize != 0:
            self.batchSize = lastBatchSize
            yield (self.Y[indices[-lastBatchSize:]],
                   self.covariates[indices[-lastBatchSize:]],
                   self.O[indices[-lastBatchSize:]],
                   indices[-lastBatchSize:]
                   )

    def averageLikelihood(self):
        '''Average the likelihood to smooth it. Do so since we can only estimate
        the likelihood, thus it is random. However, there is a need it to be accurate
        since we use the likelihood as a stopping criterion. That is why an average is computed
        for smoothness.
        '''
        self.lastLikelihoods.append(self.logLike)
        # If enough likelihoods to build the mean, remove the oldest one.
        if len(self.lastLikelihoods) > self.nbAverageLikelihood:
            self.lastLikelihoods.pop()
        self.meanOfLastLogLikes = np.mean(np.array(self.lastLikelihoods))
        self.logLikelihoodList.append(self.meanOfLastLogLikes)

    def averageParams(self):
        '''Averages the parameters in order to smooth the variance.
        Will take, for example, the last self.nbAverageParam betas computed to make
        a better approximation of beta. Same for C. 
        Args :
            None
        Returns :
            None 
        '''

        # Remove the oldest parameters and add the more recent one.
        self.lastBetas.pop()
        self.lastCs.pop()
        self.lastBetas.append(self.beta)
        self.lastCs.append(self.C)

    def fit(self, Y, O, covariates, nbEpochMax=500, lr=0.1, classOptimizer=torch.optim.Adagrad,
            takeVarianceReducedGradients=True, batchSize=40, nbMonteCarloSamples = 200,
                criterionMax=15, nbPlateauBeforeIncrementingCritertion=5, doGoodInit=True, verbose=False, debiasing=False):
        '''Batch gradient ascent on the log likelihood given the data. Infer
        pTheta with importance sampling and then computes the gradients by hand.
        At each iteration, look for the right importance sampling law running
        findBatchMode. Given this mode, estimate the variance and  compute
        the weights required to estimate pTheta. Then, derive the gradients.
        Note that it only needs to know the weights to get gradients. The mean of the
        weights gives the estimated likelihood that is used as stopping criterion for
        the algorithm.

        Args :
               Y: pd.DataFrame of size (n, p). The counts
               O: pd.DataFrame of size (n,p). The offset
               covariates: pd.DataFrame of size (n,p).
               nbepochNumberMax: int, optional. The maximum number of times the algorithm will loop over
                   the data. Will see NepochNumber times each sample. Default is 500.
               lr: float greater than 0, optional. The learning rate of the batch gradient ascent.
                   Default is 0.1.
               classOptimizer : torch.optim.optimizer object, optional. The optimizer used.
               takeVarianceReducedGradients : string, optional. the Variance Reductor we want to use. Should be one of those :
                   - 'SAGA'
                   - 'SAG'
                   - 'SVRG'
                   - None
                   If None, we are not doing any variance reduction. Else the variance of
                   the gradient will be reduced using one of the three methods.
               batchSize : int between 2 and n (included), optional. The batch size of the batch
                   gradient ascent. Default is 40.
                   Default is torch.optim.Adagrad.
               acc: float strictly between 0 and 1, optional. The accuracy when computing
                   the estimation of pTheta. The lower the more accurate but
                   the slower the algorithm. Will sample int(1/acc) gaussians
                   to estimate the likelihood. Default is 0.005.
               criterionMax : int, optional. The criterion you want to use. The algorithm
                   will stop if the criterion go past criterionMax. Default is 15.
               nbPlateauBeforeIncrementingCritertion : int, optional. The criterion will increase if the average likelihood
                   has not increased in nbPlateauBeforeIncrementingCritertion epochNumber. Default is 5.
               goodInit: Bool, optional. If True, will do an initialization that is not random.
                   Takes some time. Default is True.
               verbose: Bool, optional. If True, will plot the evolution of the algorithm
                   in real time. Default is False.
       Returns:
           None, but updates the parameter beta and C. Note that betaMean
           and CMean are more accurate and achieve a better likelihood in general.
        '''
        self.passed = False
        self.listDiff = list()
        self.listMeanESS = list()
        nbIterMaxToFindMode = 100
        self.t0 = time.time()  # To keep track of the time
        self.criterionMax = criterionMax
        self.nbPlateauBeforeIncrementingCritertion = nbPlateauBeforeIncrementingCritertion
        self.batchSize = batchSize
        self.nbMonteCarloSamples = nbMonteCarloSamples
        if self.fitted == False: 
            self.initFromData(Y, O, covariates, doGoodInit)
            self.initUsefulListsAndTensors()
        self.optimizer = classOptimizer([self.C, self.beta], lr=lr)
        if takeVarianceReducedGradients:
            vr = SAGA([self.beta, self.C], self.n)
        self.maxLogLike = -1e10
        for epochNumber in tqdm(range(nbEpochMax)):
            logLike = 0
            self.epochNumber = epochNumber
            #no shuffle is performed for analysis, shoudl shuffle again
            for YB, covariatesB, OB, selectedIndices in self.getBatch(batchSize):
                if epochNumber>1: 
                    #print('mode', self.batchMode)
                    def batchUnLogPosterior(W):
                        return batchLogPWgivenY(self.YB, self.OB, self.covariatesB, W, self.C, self.beta)
                    self.batchUnLogPosterior = batchUnLogPosterior   
                    weights = self.getWeights()
                    #print('normalized weights :', self.normalizedWeights.shape)
                    #print('samples ', self.samples)
                    self.expectationMean = torch.sum(torch.multiply(self.normalizedWeights.unsqueeze(2), self.samples), axis = 0)
                    #print('naive mean: ', torch.mean(self.samples, axis = 0))
                    samplesCentered = self.samples - self.expectationMean
                    outerW = torch.matmul(samplesCentered.unsqueeze(3), samplesCentered.unsqueeze(2)) 
                    print('outer w . shape', outerW.shape)
                    self.expectationVar = torch.sum(torch.multiply(self.normalizedWeights.unsqueeze(2).unsqueeze(3),outerW), axis = 0) 
                    self.naiveVar = torch.mean(outerW, axis = 0)
                    #print('expectation outer', expectationOuter.shape)
                    #outerMean = torch.multiply(self.expectationMean.unsqueeze(2), self.expectationMean.unsqueeze(1))
                    #print('outermean shape', outerMean.shape)
                    #self.expectationVar =  expectationOuter - outerMean 
                    #print('expectationMean:',self.expectation)

                else: 
                    print('no weights')
                beginningBatchTime = time.time()
                self.optimizer.zero_grad()
                self.YB, self.covariatesB, self.OB = YB.to(device), covariatesB.to(device), OB.to(device)
                self.selectedIndices = selectedIndices
                batchLogLike, batchGradC, batchGradBeta = self.getGradientsAndLogLike(nbIterMaxToFindMode)
                try: 
                    self.listDiff.append(torch.mean(torch.abs(self.expectationMean - self.batchMode)))
                    print('me :', torch.mean(torch.abs(self.expectationMean - self.batchMode)))
                    print('me var', torch.mean(torch.abs(self.expectationVar - self.SigmaB)))
                except: 
                    print('not possible')
                logLike+= batchLogLike
                self.getStatWeights()
                if debiasing:
                    batchGradC = self.getUnbiasedGrad('C')
                    batchGradBeta = self.getUnbiasedGrad('beta')
                if takeVarianceReducedGradients:
                    vr.computeAndSetVarianceReducedGrad([-batchGradBeta, -batchGradC], selectedIndices)
                else:
                    self.beta.grad = -torch.mean(batchGradBeta, axis=0)
                    self.C.grad = -torch.mean(batchGradC, axis=0)
                self.keepTrackOfTimeTook(beginningBatchTime)
                self.optimizer.step()
                self.averageParams()  
                varWeights, effectiveSampleSize = self.getStatWeights()
                print('mean varWeights :', torch.mean(varWeights))
                self.listMeanESS.append(torch.mean(effectiveSampleSize))
                print('mean ess', torch.mean(effectiveSampleSize))
            self.runningTimes.append(time.time() - self.t0)
            self.logLike = logLike / self.n * batchSize
            self.averageLikelihood()
            crit = self.computeCriterion(verbose) 
            if crit > self.criterionMax - 1:
                print('Algorithm stopped after ', self.epochNumber, ' epochs.')
                self.fitted = True

                break
            self.passed = True 
        self.fitted = True
    def keepTrackOfTimeTook(self, beginningBatchTime): 
        timeToProcessBatch = time.time() - beginningBatchTime  
        timeToEstimGrad= timeToProcessBatch - self.timeToFindMode
        self.timeToEstimGradList.append(timeToEstimGrad)
        self.timeToFindModeList.append(self.timeToFindMode)

    def getGradientsAndLogLike(self, nbIterMaxToFindMode ):
        self.weights = self.computeGradientRequirementsAndGetWeights(nbIterMaxToFindMode)
        perSampleLogLike = torch.log(torch.mean(self.weights, axis=0)) + self.const
        batchLogLike = torch.mean(perSampleLogLike, axis = 0) 
        batchGradC = self.getGradC()
        batchGradBeta = self.getGradBeta()
        return batchLogLike, batchGradC, batchGradBeta

    def computeBestLoglike(self, nbMonteCarloSamples=1000, nbIterMaxToFindMode=300, lrMode=0.001):
        '''Estimate the best likelihood of the model, i.e. the likelihood
        estimated with betaMean and CMean.

        Args:
            acc: float greater than 0, optional. The accuracy wanted for
                the estimation of the likelihood. Default is 0.001.
            nbIterMaxToFindMode : int, optional. The maximum number of iteration
                to do to find the mode. Default is 300.
            lrMode : float greater than 0, optional. The learning of the gradient ascent
                finding the mode. Default is 0.001.
        Returns :
            float (non positive). The estimated log likelihood of betaMean and CMean
        '''
        self.YB, self.covariatesB, self.OB = self.Y, self.covariates, self.O
        self.selectedIndices = np.arange(0, self.n)
        self.nbMonteCarloSamples = nbMonteCarloSamples 
        # Set beta and C as betaMean and CMean to compute the likelihood with the best parameters. 
        self.beta = torch.clone(self.betaMean)
        self.C = torch.clone(self.CMean)
        self.bestLogLike = self.inferBatchPTheta(nbIterMaxToFindMode, lrMode)
        return self.bestLogLike

    def computeCriterion(self, verbose=True):
        '''Updates the criterion of the model. The criterion counts the
        number of times the likelihood has not improved. We also append
        the criterion in a list in order to plot it after.
        Args :
            verbose: bool. If True, will print the criterion whenever it increases.
                Default is True.
        Returns : int. The criterion.
        '''
        if verbose:
            print('Average log likelihood : ', self.meanOfLastLogLikes)
        if self.meanOfLastLogLikes > self.maxLogLike:
            self.updateMaxLogLikeAndDontIncrementCriterion()
        else:
            self.incrementCriterion()
        triggered = int(self.CriterionCounterList[-1]
                        - self.CriterionCounterList[-self.nbPlateauBeforeIncrementingCritertion - 1] > self.nbPlateauBeforeIncrementingCritertion - 1)
        self.criterionList.append(self.criterionList[-1] + triggered)
        if triggered > 0 and verbose:
            print('Criterion updated : ',
                  self.criterionList[-1], '/', self.criterionMax)
        return self.criterionList[-1]

    def incrementCriterion(self):
        self.CriterionCounterList.append(self.CriterionCounterList[-1] + 1)

    def updateMaxLogLikeAndDontIncrementCriterion(self): 
        self.CriterionCounterList.append(self.CriterionCounterList[-1])
        self.maxLogLike = self.meanOfLastLogLikes

    def inferBatchPTheta(self, nbIterMaxToFindMode):
        '''Infer pTheta that is computed for a batch of the dataset. The
        parameter Y,O,cov are in the object itself, so that there is no need
        to pass them in argument.
        Args :
            nbIterMaxToFindMode : int. The maximum number of iteration
                to do to find the mode of the posterior.
            lrMode: postive float. The learning rate of the gradient
                ascent finding the mode.
        '''
        self.weights = self.computeGradientRequirementsAndGetWeights(nbIterMaxToFindMode)
        perSampleLogLike = torch.log(torch.mean(self.weights, axis=0)) + self.const
        return torch.mean(perSampleLogLike)

    def computeGradientRequirementsAndGetWeights(self, nbIterMaxToFindMode):
        '''Does all the operation needed to compute the gradients.
        The requirement are the gaussian samples and the weights, which are
        computed here. The gaussians samples needs to be sampled from the
        right mean and variance, found by calling findBatchMode and
        getBatchBestVar methods. The formula of the variance
        try: 
            #pass 
            self.batchMode = self.expectationMean
        except: 
            pass 
        can be found in the mathematical description.
        Args:
            NIterMode : int. The maximum number of iterations to do
                to find the mode.
            lrMode : float greater than 0. The learning rate of the
                gradient ascent finding the mode.
        Returns:
            None but computes the weights stored in the object.
        '''
        # get the mode
        self.findBatchMode(nbIterMaxToFindMode)
        # Thanks to the mode, the best variance can be computed.
        self.getBatchBestVar()
        self.tGradEstim = time.time()
        self.samples = sampleGaussians(
            self.nbMonteCarloSamples, self.batchMode, self.sqrtSigmaB)
        weights = self.getWeights()
        return weights

    def getWeights(self):
        '''Compute the weights of the IMPS formula. Given the gaussian samples
        stored in the object,the weights are computed as the ratio of the
        likelihood of the posterior and the likelihood of the gaussian samples.
        Note that it first compute the logarithm of the likelihood of the posterior
        and the logarithm of the gaussian samples, then remove the maximum of
        the difference to avoid numerical zero, and takes the exponential.
        We keep in memory the constant removed to get it back later.

        Args: None

        Returns: torch.tensor of size (nbMonteCarloSamples,NBatch). The computed weights.
        '''
        # Log likelihood of the posterior
        self.logF = self.batchUnLogPosterior(self.samples)
        # Log likelihood of the gaussian density
        self.logG = logGaussianDensity(
            self.samples, self.batchMode, self.SigmaB)
        # Difference between the two logarithm
        diffLog = self.logF - self.logG
        self.const = torch.max(diffLog, axis=0)[0]
        # remove the maximum to avoid numerical zero.
        diffLog -= torch.max(diffLog, axis=0)[0]
        weights = torch.exp(diffLog)
        self.normalizedWeights = torch.div(weights, torch.sum(weights, axis=0))
        return weights 
    def getBatchBestVar(self):
        '''Compute the best variance for the importance law. Given the mode,
        derive the best variance that fits the posterior. Please check the
        mathematical description of the package to find out why those
        computations are made.

        Args: None

        Returns: None but compute the best covariance matrix and
            its square root, stored in the IMPSPLN object.
        '''
        batchMatrix = torch.matmul(
            self.C.unsqueeze(2),
            self.C.unsqueeze(1)).unsqueeze(0)
        CW = torch.matmul(
            self.C.unsqueeze(0),
            self.batchMode.unsqueeze(2)).squeeze()
        common = torch.exp(
            self.OB
            + self.covariatesB @ self.beta
            + CW
        ).unsqueeze(2).unsqueeze(3)
        prod = batchMatrix * common
        # The hessian of the posterior
        HessPost = torch.sum(prod, axis=1) + torch.eye(self.q).to(device)
        self.oldSigmaB = torch.clone(self.SigmaB)
        self.SigmaB = torch.inverse(HessPost.detach())
        if self.passed: 
            self.SigmaB = self.expectationVar
            #self.SigmaB = self.expectationVar + torch.diag(torch.full((self.q, 1), 1e-8).squeeze()).to(device)
            trueMat = torch.inverse(HessPost.detach())
            for i in range(self.batchSize): 
                mat = self.expectationVar[i]
            #    trueMati = trueMat[i]
                eigvals, _ = TLA.eigh(mat)
                
                print('mat :', mat)
                print('old mat',self.oldSigmaB[i])
            #    trueEigvals, _ = TLA.eigh(trueMati)
            #    print('trueEigvals = ', trueEigvals)
                print('eigvals:', eigvals)
                TLA.cholesky(mat)
            #    
            #print('eigvals:', eigvals)
            #print('shape', eigvals.shape)
            self.batchMode = self.expectationMean

        else:
            print('impossible')
        # Add a term to avoid non-invertible matrix.
        #eps = torch.diag(torch.full((self.q, 1), 1e-8).squeeze()).to(device)
        self.sqrtSigmaB = TLA.cholesky(self.SigmaB) 

    def getBatchGradLogPostBeta(self):
        '''
        Computes the gradient of the log posterior with respect to beta. See the README for the formula.

        Args: None

        Returns: torch.tensor of size (nbMonteCarloSamples, batchSize,p,q)
        '''
        XY = torch.matmul(
            self.covariatesB.unsqueeze(2),
            self.YB.unsqueeze(1))
        XB = torch.matmul(self.covariatesB.unsqueeze(1),
                          self.beta.unsqueeze(0)).squeeze()
        CV = torch.matmul(self.C.reshape(1, 1, self.p, 1, self.q),
                          self.samples.unsqueeze(2).unsqueeze(4)).squeeze()
        Xexp = torch.matmul(self.covariatesB.unsqueeze(0).unsqueeze(3),
                            torch.exp(self.OB + XB + CV).unsqueeze(2))
        return XY.unsqueeze(0) - Xexp

    def getBiasBeta(self, i):
        '''
        non vectorized version of getBias for the parameter beta.
        It calculates the bias for one sample only. Just here as a double check for
        the more general function getBias.
        '''
        pThetaI = torch.exp(
            (torch.log(torch.mean(self.weights, axis=0)) + self.const)[i])
        IChap = self.getBatchGradBeta()[i]
        Dbar = self.weights[:, i]*torch.exp(self.const[i])
        gradLogPostI = self.getBatchGradLogPostBeta()[:, i]
        Nbar = torch.multiply(Dbar.unsqueeze(1).unsqueeze(2), gradLogPostI)
        NbarCentered = Nbar - torch.mean(Nbar, axis=0)
        DbarCentered = Dbar - torch.mean(Dbar)
        cov = torch.mean(torch.multiply(DbarCentered.unsqueeze(
            1).unsqueeze(2), NbarCentered), axis=0)
        return 1/(pThetaI**2)*(IChap*torch.var(Dbar) - cov)

    def getVarianceGrad(self, stringParameter):
        '''Compute the variance of the gradient for either C or beta for every sample
        in the dataset.

        Args:
            stringParameter: string. Eiter 'C' or 'beta'. If none of those, will raise an error.
                We will compute the bias of the gradient with respect of stringParameter
        Returns:
            torch.tensor of size (batchSize, prod(gradientSize)), where prod is the product
            of all the two dimensions.
        '''
        pTheta = torch.exp(
            (torch.log(torch.mean(self.weights, axis=0)) + self.const))
        Dbar = torch.multiply(self.weights, torch.exp(self.const).unsqueeze(0))
        if stringParameter == 'C':
            IChap = self.getGradC()
            gradLogPost = self.getBatchGradLogPostC()
        elif stringParameter == 'beta':
            IChap = self.getGradBeta()
        else: 
            raise ValueError(
                'You can calculate the variance with respect to C or beta only')
        Nbar = torch.multiply(Dbar.unsqueeze(2).unsqueeze(3), gradLogPost)
        vecNbar = Nbar.flatten(startDim=-2)
        vecNbarCentered = vecNbar - torch.mean(vecNbar, axis=0)
        varNbar = torch.mean(torch.matmul(vecNbarCentered.unsqueeze(
            3), vecNbarCentered.unsqueeze(2)), axis=0)
        vecIChap = IChap.flatten(startDim=-2)
        DbarCentered = Dbar - torch.mean(Dbar, axis=0)
        covDbarNbar = torch.mean(torch.multiply(
            DbarCentered.unsqueeze(2), vecNbarCentered), axis=0)
        IChapcov = torch.matmul(
            vecIChap.unsqueeze(2), covDbarNbar.unsqueeze(1))
        covIChap = torch.matmul(
            covDbarNbar.unsqueeze(2), vecIChap.unsqueeze(1))
        varDIIt = torch.multiply(torch.matmul(vecIChap.unsqueeze(
            2), vecIChap.unsqueeze(1)), torch.var(Dbar, axis=0).unsqueeze(1).unsqueeze(2))
        return torch.div(varNbar - IChapcov - covIChap + varDIIt, (pTheta**2).unsqueeze(1).unsqueeze(2))

    def getUnbiasedGrad(self, stringParameter):
        '''Compute the estimated bias of the estimator of the gradient
        for either beta or C (for every sample in the dataset).

        Args:
            stringParameter: string. Eiter 'C' or 'beta'. If none of those, will raise an error.
                We will compute the bias of the gradient with respect of stringParameter
        Returns:
            torch.tensor of size (batchSize, gradientSize)
        '''
        pTheta = torch.exp(
            (torch.log(torch.mean(self.weights, axis=0)) + self.const))
        Dbar = torch.multiply(self.weights, torch.exp(self.const).unsqueeze(0))
        
        
        if stringParameter == 'C':
            gradLogPost = self.getBatchGradLogPostC()
            IChap = self.getGradC()
        elif stringParameter == 'beta':
            gradLogPost = self.getBatchGradLogPostBeta()
            IChap = self.getGradBeta()
        else:
            ValueError(
                'You can calculate the bias with respect to C or beta only')
        Nbar = torch.multiply(Dbar.unsqueeze(2).unsqueeze(3), gradLogPost)
        NbarCentered = Nbar - torch.mean(Nbar, axis=0)
        DbarCentered = Dbar - torch.mean(Dbar, axis=0)
        cov = torch.mean(torch.multiply(DbarCentered.unsqueeze(
            2).unsqueeze(3), NbarCentered), axis=0)
        IChapMuSquared = torch.multiply(
            IChap, (pTheta**2).unsqueeze(1).unsqueeze(2))
        var = torch.var(Dbar, axis=0)
        return torch.div(IChapMuSquared + cov, (var + pTheta**2).unsqueeze(1).unsqueeze(2))

    def getGradBeta(self):
        ''' Computes the gradient with respect to beta of the log likelihood
        for the batch. The derivation of the formula is in the README.
        We only multiply the gradient of the log posterior with the normalized weights
        and sum the first axis.

        Args: None

        Returns: torch.tensor of size (batchSize,d,p). The gradient wrt beta.
        '''
        gradLogPost = self.getBatchGradLogPostBeta()
        return torch.sum(torch.multiply(gradLogPost, self.normalizedWeights.unsqueeze(2).unsqueeze(3)), axis=0)

    def getGradC(self):
        '''Computes the gradient with respect to C of the log likelihood for
        the batch. The derivation of the formula is in the README.
        We only multiply the gradient of the log posterior with the normalized weights
        and sum the first axis.

        Args: None

        Returns: torch.tensor of size (batchSize,d,p). The gradient wrt C.
        '''
        gradLogPost = self.getBatchGradLogPostC()
        return torch.sum(torch.multiply(gradLogPost, self.normalizedWeights.unsqueeze(2).unsqueeze(3)), axis=0)

    def getBatchGradLogPostC(self):
        '''
        Computes the gradient of the log posterior with respect to C. See the README for the formula.

        Args: None

        Returns: torch.tensor of size (nbMonteCarloSamples, batchSize,p,q)
        '''
        XB = torch.matmul(
            self.covariatesB.unsqueeze(1),
            self.beta.unsqueeze(0)).squeeze()
        CV = torch.matmul(
            self.C.reshape(1, 1, self.p, 1, self.q),
            self.samples.unsqueeze(2).unsqueeze(4)
        ).squeeze()
        Ymoinsexp = self.YB - torch.exp(self.OB + XB + CV)
        outer = torch.matmul(Ymoinsexp.unsqueeze(3), self.samples.unsqueeze(2))
        return outer

    def showSigma(self, ax=None, save=False, nameDoss='IMPSPLNSigma'):
        '''Displays Sigma
        args:
            'ax': AxesSubplot object. Sigma will be displayed in this ax
                if not None. If None, will simply create an axis. Default is None.
            'nameDoss': str. The name of the file the graphic will be saved to.
                Default is 'IMPSPLNSigma'.
        returns: None but displays Sigma.
        '''
        fig = plt.figure()
        if self.p > 400:
            print('The heatmap only displays Sigma[:400,:400]')
            sns.heatmap(self.getSigma()[:400, :400].cpu().detach(), ax=ax)
        else:
            sns.heatmap(self.getSigma().cpu().detach(), ax=ax)
        # save the graphic if needed
        if save:
            plt.savefig(nameDoss)
        if ax is None:
            plt.show()

    def getStatWeights(self):
        varWeights = torch.var(self.normalizedWeights, axis=0)
        effectiveSampleSize = torch.div(torch.sum(self.normalizedWeights, axis=0)**2, torch.sum(self.normalizedWeights**2, axis=0))/self.nbMonteCarloSamples
        return varWeights, effectiveSampleSize
    def showLoss(self, ax=None, save=False, nameDoss='IMPSPLNLogLikelihood'):
        '''Show the log likelihood of the algorithm along the iterations.

        args:
            'ax': AxesSubplot object. The ELBO will be displayed in this ax
                if not None. If None, will simply create an axis. Default is None.
            'nameDoss': str. The name of the file the graphic will be saved to.
                Default is 'IMPSPLNLogLikelihood'.
        returns: None but displays the log likelihood.
        '''
        if not self.fitted:
            raise AttributeError(
                'Please fit the model before by calling model.fit(Y,O,covariates)')
        if ax is None:
            ax = plt.gca()
            toShow = True
        else:
            toShow = False
        ax.plot(self.runningTimes,
                -np.array(self.logLikelihoodList), label = self.NAME)
        ax.set_title('Smoothed negative log likelihood')
        ax.set_ylabel('Negative loglikelihood')
        ax.set_xlabel('Seconds')
        ax.set_yscale('log')
        # save the graphic if needed
        if save:
            plt.savefig(nameDoss)
        if toShow:
            plt.show()

    def showCriterion(self, ax=None, save=False, nameDoss='IMPSPLNCriterion'):
        '''Show the criterion of the algorithm along the iterations.

        args:
            'ax': AxesSubplot object. The criterion will be displayed in this ax
                if not None. If None, will simply create an axis. Default is None.
            'nameDoss': str. The name of the file the graphic will be saved to.
                Default is 'IMPSPLNCriterion'.

        returns: None but displays the criterion.
        '''
        if not self.fitted:
            raise AttributeError(
                'Please fit the model before by calling model.fit(Y,O,covariates)')
        if ax is None:
            ax = plt.gca()
            toShow = True
        else:
            toShow = False

        ax.plot(self.runningTimes, self.criterionList[1:])
        ax.set_title('Number of epochNumber the likelihood has not improved')
        ax.set_xlabel('Seconds')
        # save_ the graphic if needed
        if save:
            plt.savefig(nameDoss)
        if toShow:
            plt.show()

    def __str__(self):
        '''Show the criterion of the algorithm, Sigma and the log likelihood.'''
        self.bestLogLike = max(self.logLikelihoodList)
        print('Best likelihood: ', self.bestLogLike)
        fig, axes = plt.subplots(1, 3, figsize=(20, 5))
        self.showLoss(ax=axes[0])
        self.showCriterion(ax=axes[1])
        self.showSigma(ax=axes[2])
        self.fig = fig
        plt.show()
        return ''

    def saveStat(self, nameDoss='IMPSPLNGraphic'):
        '''save some useful stats of the model.
        Args:
            'nameDoss' : str. The name of the file the graphic will be saved to.
                Default is 'IMPSPLNGraphic'.
        Returns :
                None but displays the figure.
        '''
        print(self)
        self.fig.savefig(nameDoss)

    def getBeta(self):
        '''Getter for beta. Returns the mean of the last betas computed to reduce variance.'''
        return self.betaMean.detach()

    def getSigma(self):
        '''Getter. Get Sigma by computing CMean@CMean^T. We take CMean to reduce variance. '''
        return (self.CMean.detach()) @ (self.CMean.detach().T)

    def getC(self):
        '''Getter for C. We return CMean to reduce variance'''
        return self.CMean.detach()

    def findBatchMode(self, nbepochNumberMax, eps=9e-3):
        '''Find the mode of the posterior with a gradient ascent.
        The last mode computed is used as starting point. However,
        each mode depends on the batch (YB,OB, covariatesB), so that
        there is a need to know from which indices we have selected the batch.

        Args:
            nbepochNumberMax: int. The maximum number of iteration to do
                to find the mode.
            lr: positive float. The learning rate of the optimizer for the
                gradient ascent.
            eps: positive float, optional. The tolerance. The algorithm will
                stop if the maximum of |WT-W{t-1}| is lower than eps, where WT
                is the t-th iteration of the algorithm.This parameter changes a lot
                the resulting time of the algorithm. Default is 9e-3.

        Returns :
            None, but compute and stock the mode in self.batchMode and the starting point.
        '''
        # The loss used use for the gradient ascent.
        beginningTime = time.time()
        def batchUnLogPosterior(W):
            return batchLogPWgivenY(
                self.YB, self.OB, self.covariatesB, W, self.C, self.beta)
        self.batchUnLogPosterior = batchUnLogPosterior
        # Get the corresponding starting point.
        W = self.startingPoint[self.selectedIndices]
        W.requires_grad = True
        # If we have seen enough data, we set the learning rate to zero
        # since we will actually use the previous learning rate.
        if self.epochNumber > 5:
            lr = 0
        optim = torch.optim.Rprop([W], lr= 0.1)
        criterion = 2 * eps
        oldW = torch.clone(W)
        i = 0
        stopCondition = False
        while i < nbepochNumberMax and stopCondition == False:
            # When self.epochNumber >5, will move just a little bit from the previous C and beta.
            # Thus, the mode won't move very much. We take the previous stepSizes of the optimizer
            # in order to reach the maximum faster.
            if i == 1 and self.epochNumber > 5:
                optim.state_dict()['state'][0]['step_size'] = 5 *self.modeStepSizes[self.selectedIndices, :]
            loss = -torch.mean(self.batchUnLogPosterior(W))
            loss.backward()
            optim.step()
            crit = torch.max(torch.abs(W - oldW))
            optim.zero_grad()
            if crit < eps and i > 2:  # we want to do at least 3 iteration per loop.
                stopCondition = True
            oldW = torch.clone(W)
            i += 1
        # Stock the mode
        self.batchMode = torch.clone(W.detach())
        self.startingPoint[self.selectedIndices] = torch.clone(W.detach())
        self.modeStepSizes[self.selectedIndices,
                             :] = optim.state_dict()['state'][0]['step_size']
        self.timeToFindMode = time.time() - beginningTime
    
    def plotRuntime(self):
        '''
        Shows different runtimes of the .fit() method. It shows what computation takes
        time. Do so to estimate what should be lowered or increased. For example,
        if the Gradient estimation running time is very low compared to the time took to find the mode,
        then the overall running time won't increase very much if the accuracy parameter is lowered.
        Args:
            None
        Returns:
            None but displays a figure.
        '''
        lEstim = self.timeToEstimGradList
        lMode = self.timeToFindModeList
        lTotal = np.array(lEstim) + np.array(lMode)
        toPlot = {'Gradient estimation': lEstim, 'Finding mode': lMode,'Total runtime': lTotal}
        for key,values in toPlot.items():
            plotList(values, key)
        plt.xlabel('Iteration')
        plt.legend()
        print('Total time :', np.sum(lTotal))
        plt.show()




