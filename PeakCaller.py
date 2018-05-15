import numpy as np
from scipy.optimize import minimize
from scipy.special import gamma

import warnings

print 'Hello'

"""
Function list

vectB = Peaks_Caller(vect,WS=10000,NR=10,Pval=5)

vectB = Global_threshold_selection(vect,WS=10000,NR=10,Pval=5)

"""


def Loss_NB(param,M,Ytrue):

    Ytrain = NegativeBinomiale(M,param)

    return ((Ytrue[1:]-Ytrain[1:])**2).sum()


def NegativeBinomiale(M,param):
    n = param[0]
    p = param[1]
    k = np.arange(M)

    PY_theta = np.exp(np.log(gamma(k+n)) - np.log(gamma(n))-logfactorial(k)+n*np.log(p) + k*np.log(1-p))

    #Normalization
    PY_theta /= PY_theta.sum()

    return PY_theta


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

def EXP(M,param):
    """
    param[0]: pi  -> Proportion of unmappable genome
    param[1]: alpha  -> shape parameter dictating the distribution of reads
    param[2]: p1  -> average number of reads per window
    param[3]: p0  -> 1-p1
    """
    a = param[0]
    k = param[1]

    PY_Theta = np.log(a)-k*np.log(np.arange(1,M))
    PY_Theta = np.exp(PY_Theta)

    #Normalization
    PY_Theta /= PY_Theta.sum()

    return PY_Theta


def Loss_EXP(param,M,Ytrue):

    Ytrain = EXP(M,param)

    #return ((np.log10(Ytrue[1:])-np.log10(Ytrain))**2).sum()
    return ((Ytrue[1:]-Ytrain)**2).sum()

def Poisson(M,param):
    """
    param[0]: pi  -> Proportion of unmappable genome
    param[1]: alpha  -> shape parameter dictating the distribution of reads
    param[2]: p1  -> average number of reads per window
    param[3]: p0  -> 1-p1
    """
    lmbd = param[0]

    PY_Theta = np.log(lmbd)*np.arange(M) - lmbd - logfactorial(np.arange(M))
    PY_Theta = np.exp(PY_Theta)

    #Normalization
    PY_Theta /= PY_Theta.sum()

    return PY_Theta


def Loss_Poisson(param,M,Ytrue):

    Ytrain = Poisson(M,param)

    #return ((np.log10(Ytrue[1:])-np.log10(Ytrain[1:]))**2).sum()
    return ((Ytrue[1:]-Ytrain[1:])**2).sum()

def logfactorial(X):
    N = len(X)
    Y = np.zeros(N)
    for i in range(N):
        Y[i] = np.sum(np.log(range(1,X[i]+1)))

    return Y

def Peaks_Caller(vect,WS=50000,NR=10,Pval=5):


    '''
        vect: vector to binarize
        WS: Windows on which the optimization is done
        NR: Number of optimization with random start done for the parameters initislization
        Pval: Threshold above which the signal is discretized (default=5)

        out:
            vect_pval : binarized version of vect
    '''

    warnings.filterwarnings("ignore")

    #Initailizing vectB
    vectB = np.copy(vect)

    #Defining threshold to fit the histogram ()
    bins = np.arange(np.max(vect))
    hist, bins = np.histogram(vect, bins=bins)
    bins = bins[:-1]
    hist = hist.astype(np.float64)
    hist/=np.sum(hist)
    M =  np.where(np.log10(1-np.cumsum(hist))<-3)[0][0]
    print 'Maximal histogram value: ' + str(M)

    print '\nParameters initialisation on all the genome'
    #Ytrue
    bins = np.arange(np.max(vect))
    Ytrue, bin_edges = np.histogram(vect, bins=bins)
    bins = bins[:M]
    Ytrue = Ytrue.astype(np.float64)[:M]
    Ytrue/=np.sum(Ytrue)

    l_min=np.inf
    while l_min==np.inf: #In case the minimizer has not CVG after the NR iterations
        for i in range(NR):
            print 'Random iteration: ' + str(i)
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
                print '--> Optimal Cost Function: ' + str(l_min)

    N = vect.shape[0]
    W = np.append(np.arange(0,N,WS),N)
    for i in range(len(W)-1):
        print 'Part: ' + str(i+1) + ' / ' + str(len(W)-1)
        X = np.copy(vect[W[i]:W[i+1]])
        X = X[X>0]

        if len(X)<=10:
            vectB[W[i]:W[i+1]] = 0
            continue

        #Ytrue
        bins = np.arange(M+1)
        Ytrue, bin_edges = np.histogram(X, bins=bins)
        bins = bins[:M]
        Ytrue = Ytrue.astype(np.float64)[:M]
        Ytrue/=np.sum(Ytrue)

        #Mi = np.max([np.min([Ytrue.shape[0],M]),20])

        #Fit by changing the global parameters
        param_opt_i = minimize(Loss_ZIMB,x0=param_opt,args=(M,Ytrue[:M]),method="Nelder-Mead").x
        T = np.where(np.log10(1-np.cumsum(ZIMB(M,param_opt_i)))<-Pval)[0]#We cut the fit at 10^-pval
        k=1
        while len(T)==0:
            T = np.where(np.log10(1-np.cumsum(ZIMB(M,param_opt_i)))<-Pval+k)[0]
            k+=1

        T=T[0]
        print 'Threshold value = ' + str(T)

        v = vectB[W[i]:W[i+1]]
        if not np.isnan(T):
            v[v<=T] = 0
            v[v>T] = 1
        else:
            v[v>0] = 0


    return vectB

def Peaks_Caller_Pval(vect,WS=50000,NR=10):


    '''
        vect: vector to binarize
        WS: Windows on which the optimization is done
        NR: Number of optimization with random start done for the parameters initislization
        Pval: Threshold above which the signal is discretized (default=5)

        out:
            vect_pval : binarized version of vect
    '''

    warnings.filterwarnings("ignore")

    #Initailizing vectB
    vectB = np.copy(vect)

    #Defining threshold to fit the histogram ()
    bins = np.arange(np.max(vect))
    hist, bins = np.histogram(vect, bins=bins)
    bins = bins[:-1]
    hist = hist.astype(np.float64)
    hist/=np.sum(hist)
    M =  np.where(np.log10(1-np.cumsum(hist))<-3)[0][0]
    print 'Maximal histogram value: ' + str(M)

    print '\nParameters initialisation on all the genome'
    #Ytrue
    bins = np.arange(np.max(vect))
    Ytrue, bin_edges = np.histogram(vect, bins=bins)
    bins = bins[:M]
    Ytrue = Ytrue.astype(np.float64)[:M]
    Ytrue/=np.sum(Ytrue)

    l_min=np.inf
    while l_min==np.inf: #In case the minimizer has not CVG after the NR iterations
        for i in range(NR):
            print 'Random iteration: ' + str(i)
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
                print '--> Optimal Cost Function: ' + str(l_min)

    N = vect.shape[0]
    W = np.append(np.arange(0,N,WS),N)
    for i in range(len(W)-1):
        print 'Part: ' + str(i+1) + ' / ' + str(len(W)-1)
        X = np.copy(vect[W[i]:W[i+1]])
        X = X[X>0]

        if len(X)<=10:
            vectB[W[i]:W[i+1]] = 0
            continue

        #Ytrue
        bins = np.arange(M+1)
        Ytrue, bin_edges = np.histogram(X, bins=bins)
        bins = bins[:M]
        Ytrue = Ytrue.astype(np.float64)[:M]
        Ytrue/=np.sum(Ytrue)

        #Mi = np.max([np.min([Ytrue.shape[0],M]),20])

        #Fit by changing the global parameters
        param_opt_i = minimize(Loss_ZIMB,x0=param_opt,args=(M,Ytrue[:M]),method="Nelder-Mead").x
        Pval_vect = -np.log10(1-np.cumsum(ZIMB(int(X.max()+1),param_opt_i)))

        Pval_vect[np.isnan(Pval_vect)]=20
        Pval_vect[np.isinf(Pval_vect)]=20

        vectB[W[i]:W[i+1]] = Pval_vect[vectB[W[i]:W[i+1]].astype(np.int32)]

    return vectB



def Peaks_Caller_Pval_MP(vect,WS=50000,NR=10,Ncore=1):
    import multiprocessing
    '''
        vect: vector to binarize
        WS: Windows on which the optimization is done
        NR: Number of optimization with random start done for the parameters initislization
        Pval: Threshold above which the signal is discretized (default=5)

        out:
            vect_pval : binarized version of vect
    '''

    warnings.filterwarnings("ignore")


    #Defining threshold to fit the histogram ()
    bins = np.arange(np.max(vect))
    hist, bins = np.histogram(vect, bins=bins)
    bins = bins[:-1]
    hist = hist.astype(np.float64)
    hist/=np.sum(hist)
    M =  np.where(np.log10(1-np.cumsum(hist))<-3)[0][0]
    #if M>1000:
        #M=1000
    print 'Maximal histogram value: ' + str(M)

    print '\nParameters initialisation on all the genome'
    #Ytrue
    bins = np.arange(np.max(vect))
    Ytrue, bin_edges = np.histogram(vect, bins=bins)
    bins = bins[:M]
    Ytrue = Ytrue.astype(np.float64)[:M]
    Ytrue/=np.sum(Ytrue)

    l_min=np.inf
    while l_min==np.inf: #In case the minimizer has not CVG after the NR iterations
        for i in range(NR):
            print 'Random iteration: ' + str(i)
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
                print '--> Optimal Cost Function: ' + str(l_min)

    N = vect.shape[0]
    W = np.append(np.arange(0,N,WS),N).astype(np.int32)
    X = np.copy(vect)


    pool = multiprocessing.Pool(Ncore)
    args = [[X[W[i]:W[i+1]],M,param_opt,i,len(W)] for i in range(len(W)-1)]
    #Pval_vect_c = map(wrapper_Peaks_Caller_Pval_MP,args)
    Pval_vect_c = pool.map(wrapper_Peaks_Caller_Pval_MP,args)
    pool.close()
    pool.join()

    print 'Signal Concatenation ...'
    vectP = np.copy(vect)
    for i in range(len(W)-1):
        #print vectP[W[i]:W[i+1]]
        #print vect[W[i]:W[i+1]]
        #print Pval_vect_c[i][vect[W[i]:W[i+1]].astype(np.int32)]

        vectP[W[i]:W[i+1]] = Pval_vect_c[i][vect[W[i]:W[i+1]].astype(np.int32)]

    return vectP

def wrapper_Peaks_Caller_Pval_MP(args):

    X,M,param_opt,i,lW = args

    print 'Part: ' + str(i+1) + ' / ' + str(lW-1)

    X = X[X>0]

    #If there is not enougth sites, just set all the points to zeros
    if len(X)<=10:
        return np.zeros(int(X.max()+1))

    else:
    #Ytrue
        bins = np.arange(M+1)
        Ytrue, bin_edges = np.histogram(X, bins=bins)
        bins = bins[:M]
        Ytrue = Ytrue.astype(np.float64)[:M]
        Ytrue/=np.sum(Ytrue)

        #Mi = np.max([np.min([Ytrue.shape[0],M]),20])

        #Fit by changing the global parameters
        param_opt_i = minimize(Loss_ZIMB,x0=param_opt,args=(M,Ytrue[:M]),method="Nelder-Mead").x
        Pval_vect = -np.log10(1-np.cumsum(ZIMB(int(X.max()+1),param_opt_i)))

        Pval_vect[np.isnan(Pval_vect)]=20
        Pval_vect[np.isinf(Pval_vect)]=20

        return Pval_vect

def Global_threshold_selection(vect,NR=10,Pval=5):


    '''
        vect: vector to binarize
        WS: Windows on which the optimization is done
        NR: Number of optimization with random start done for the parameters initislization
        Pval: Threshold above which the signal is discretized (default=5)

        out:
            vect_pval : binarized version of vect
    '''

    warnings.filterwarnings("ignore")

    #Initailizing vectB
    vectB = np.copy(vect)

    #Defining threshold to fit the histogram ()
    bins = np.arange(np.max(vect))
    hist, bin_edges = np.histogram(vect, bins=bins)
    bins = bins[:-1]
    hist = hist.astype(np.float64)
    hist/=np.sum(hist)
    #Select a threshold in order to not compute all values (outliers included) -> rm 1e3 lhighest peaks
    M =  np.where(np.log10(1-np.cumsum(hist))<-1)[0][0]
    print 'Maximal histogram value: ' + str(M)


    print '\nParameters initialisation'
    #Ytrue
    bins = np.arange(np.max(vect))
    Ytrue, bin_edges = np.histogram(vect, bins=bins)
    bins = bins[:M]
    Ytrue = Ytrue.astype(np.float64)[:M]
    Ytrue/=np.sum(Ytrue)

    l_min=np.inf
    while l_min==np.inf:
        for i in range(NR):
            print 'Random iteration: ' + str(i)
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
                print '--> Optimal Cost Function: ' + str(l_min)

    #Define a threshold at Pvalue = 10-5
    T = np.where(np.log10(1-np.cumsum(ZIMB(M,param_opt)))<-Pval)[0]
    k=1
    while len(T)==0:
        T = np.where(np.log10(1-np.cumsum(ZIMB(M,param_opt_i)))<-Pval+k)[0]
        k+=1

    T=T[0]
    print '\nThreshold value = ' + str(T)
    #Binarization
    vectB[vect<=T] = 0
    vectB[vect>T] = 1

    return vectB

from scipy.optimize import basinhopping

def Global_threshold_selection_Pval(vect,NR=10):


    '''
        vect: vector to binarize
        WS: Windows on which the optimization is done
        NR: Number of optimization with random start done for the parameters initislization
        Pval: Threshold above which the signal is discretized (default=5)

        out:
            vect_pval : binarized version of vect
    '''

    warnings.filterwarnings("ignore")

    #Defining threshold to fit the histogram ()
    bins = np.arange(np.max(vect))
    hist, bin_edges = np.histogram(vect, bins=bins)
    bins = bins[:-1]
    hist = hist.astype(np.float64)
    hist/=np.sum(hist)
    #Select a threshold in order to not compute all values (outliers included) -> rm 1e3 lhighest peaks
    M =  np.where(np.log10(1-np.cumsum(hist))<-3)[0][0]
    print 'Maximal histogram value: ' + str(M)


    print '\nParameters initialisation'
    #Ytrue
    bins = np.arange(np.max(vect))
    Ytrue, bin_edges = np.histogram(vect, bins=bins)
    bins = bins[:M]
    Ytrue = Ytrue.astype(np.float64)[:M]
    Ytrue/=np.sum(Ytrue)

    l_min=np.inf
    while l_min==np.inf:
        for i in range(NR):
            print 'Random iteration: ' + str(i)
            ##Parameters NB
            #pi = np.random.rand(1)[0]*0.1
            #alpha = np.random.rand(1)[0]*0.1
            #p1 = np.random.rand(1)[0]*0.1
            #p0 = 1-p1

            #param = [pi,alpha,p1,p0]

            #res = minimize(Loss_ZIMB,x0=param,args=(M,Ytrue),method="Nelder-Mead")

            #a = np.random.rand(1)[0]*1
            #k = np.random.rand(1)[0]*100

            #param = [a,k]

            #res = minimize(Loss_EXP,x0=param,args=(M,Ytrue),method="Nelder-Mead")

            lmbd = np.random.rand(1)[0]

            param = [lmbd]
            res = minimize(Loss_Poisson,x0=param,args=(M,Ytrue),method="Nelder-Mead")




            l = res.fun
            if l<l_min:
                param_opt = res.x
                l_min = l
                res_min = res
                print '--> Optimal Cost Function: ' + str(l_min)


    #Pval_vect = -np.log10(1-np.cumsum(ZIMB(int(vect.max()+1),param_opt)))
    #Pval_vect = -np.log10(1-np.cumsum(EXP(int(vect.max()+2),param_opt)))
    Pval_vect = -np.log10(1-np.cumsum(Poisson(int(vect.max()+2),param_opt)))

    Pval_vect[np.isnan(Pval_vect)]=20
    Pval_vect[np.isinf(Pval_vect)]=20

    return Pval_vect[vect.astype(np.int32)],param_opt
