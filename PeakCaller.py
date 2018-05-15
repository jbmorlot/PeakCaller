import numpy as np
from scipy.optimize import minimize
from scipy.special import gamma
import multiprocessing

import warnings

def ZIMB(M,param):
    """
    param[0]: pi  -> Proportion of unmappable genome
    param[1]: alpha  -> shape parameter dictating the distribution of reads
    param[2]: p1  -> average number of reads per window
    param[3]: p0  -> 1-p1
    """
    pi = param[0]
    alpha = param[1]
    p1 = param[2]
    p0 = param[3]

    PY_Theta = np.zeros(M)
    PY_Theta[0] = pi + (1-pi)*np.power(p0,alpha)
    PY_Theta[1:] = np.exp(np.log(1-pi)
                          + np.log(gamma(np.sum(param[1:])))
                          -(np.log(gamma(alpha)) + logfactorial(np.arange(1,M)))
                          + alpha*np.log(p0)
                          + np.arange(1,M)*np.log(p1)
                          )

    #Normalization
    PY_Theta /= PY_Theta.sum()

    return PY_Theta


def Loss_ZIMB(param,M,Ytrue):

    Ytrain = ZIMB(M,param)

    return ((Ytrue-Ytrain)**2).sum()
    ##return ((np.log10(Ytrue)-np.log10(Ytrain))**2).sum()
    #return (np.abs(np.log10(Ytrue[:1])-np.log10(Ytrain[:1]))/np.log10(Ytrue[:1])).sum()

def logfactorial(X):
    N = len(X)
    Y = np.zeros(N)
    for i in range(N):
        Y[i] = np.sum(np.log(range(1,X[i]+1)))

    return Y


def Peaks_Caller(vect,WS=50000,NR=10,Ncore=1,verbose=False,Pval=5,getPval=False):
    '''
        Peak caller which transform a raw NGS experiment vector in a binary Vector
        indentifying the position of the peaks.
        The algorithm first fit the distribution of the entire vector with
        a Zero Inflated Negative Binomial distribution.
        Then, the parameters of this optimization in non overlapping windows of
        size WS using the parameters of the global optimization as starting point.

        Inputs:

        vect: vector to binarize
        WS: Windows on which the optimization is done
        NR: Number of optimization with random start done for the parameters initislization
        Pval: Threshold above which the signal is discretized (default=5)
        Ncore: Number of core used to compute the different windows
        getPval: The
        out:
            vector of pval or binary of the same size than the original vector
    '''

    warnings.filterwarnings("ignore")

    if verbose: print 'Optimizing globally the ZINB distribution parameters'
    #Outlier Detection
    #The original signal can have outliers that increase dramatically the
    #size of the vector to optimize: Thus the histogram is cutted a 99.9 percentile.
    bins = np.arange(np.max(vect))
    hist, bins = np.histogram(vect, bins=bins)
    bins = bins[:-1]
    hist = hist.astype(np.float64)
    hist/=np.sum(hist)
    M =  np.where(np.log10(1-np.cumsum(hist))<-3)[0][0]
    if verbose: print '\tMaximal histogram value: ' + str(M)


    if verbose: print '\tParameters initialisation on all the genome'
    #Ytrue
    bins = np.arange(np.max(vect))
    Ytrue, bin_edges = np.histogram(vect, bins=bins)
    bins = bins[:M]
    Ytrue = Ytrue.astype(np.float64)[:M]
    Ytrue/=np.sum(Ytrue)

    l_min=np.inf
    while l_min==np.inf: #In case the minimizer has not converged after the NR iterations
        for i in range(NR):
            if verbose: print '\tRandom iteration: ' + str(i)
            #Parameters NB
            pi = np.random.rand(1)[0]*0.1
            alpha = np.random.rand(1)[0]*0.1
            p1 = np.random.rand(1)[0]*0.1
            p0 = 1-p1

            param = [pi,alpha,p1,p0]

            res = minimize(Loss_ZIMB,x0=param,args=(M,Ytrue),method="Nelder-Mead")
            l = res.fun
            if l<l_min:
                param_opt = res.x
                l_min = l
                res_min = res
                if verbose: print '\t--> Optimal Cost Function: ' + str(l_min)

    if verbose: print 'Optimizing locally the ZINB distribution parameters'
    N = vect.shape[0]
    W = np.append(np.arange(0,N,WS),N-WS,N).astype(np.int32)
    NW = len(W)
    X = np.copy(vect)


    if Ncore==-1: Ncore = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(Ncore)
    args = [[X[W[i]:W[i+1]],M,param_opt,i,len(W),verbose] for i in range(NW-1)]
    Pval_vect_c = pool.map(wrapper_Peaks_Caller_Pval_MP,args)
    pool.close()
    pool.join()

    if verbose: print 'Signal Concatenation ...'
    vectP = np.copy(vect)
    for i in range(NW-1):
        if i!=NW-3:
            vectP[W[i]:W[i+1]] = Pval_vect_c[i][vect[W[i]:W[i+1]].astype(np.int32)]

    if getPval:
        return vectP
    else:
        vectB = vectP.copy()
        vectB[vectP<Pval] = 0
        vectB[vectP>=Pval] = 1
        return vectB


def wrapper_Peaks_Caller_Pval_MP(args):

    X,M,param_opt,i,lW,verbose = args

    if verbose: print 'Part: ' + str(i+1) + ' / ' + str(lW-1)

    X = X[X>0]

    #If there is not enougth non zeros sites, just set all the points to zeros
    if len(X)<=10:
        return np.zeros(int(X.max()+1))

    else:
        #Ytrue
        bins = np.arange(M+1)
        Ytrue, bin_edges = np.histogram(X, bins=bins)
        bins = bins[:M]
        Ytrue = Ytrue.astype(np.float64)[:M]
        Ytrue/=np.sum(Ytrue)

        #Fit by changing the global parameters
        param_opt_i = minimize(Loss_ZIMB,x0=param_opt,args=(M,Ytrue[:M]),method="Nelder-Mead").x
        Pval_vect = -np.log10(1-np.cumsum(ZIMB(M,param_opt_i)))
        #If the value is too close to 0, NaN or Inf arise--> Set them to high pvalue
        Pval_vect[np.isnan(Pval_vect)]=20
        Pval_vect[np.isinf(Pval_vect)]=20


    return Pval_vect
