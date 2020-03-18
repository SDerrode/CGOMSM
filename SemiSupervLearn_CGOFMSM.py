#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import sys

import matplotlib
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

def main():

    """
        Programmes pour simuler et restaurer des siganux réels avec CGOFMSM.
 
        :Example:

        >> python3 SemiSupervLearn_CGOFMSM.py ./Data/Traffic/Zied1/TMU6048TrainX_extract.txt  ./Data/Traffic/Zied1/TMU6048TrainY_extract.txt 10 1 4 2
        >> python3 SemiSupervLearn_CGOFMSM.py ./Data/Traffic/Zied1/TMU6048TrainX.txt  ./Data/Traffic/Zied1/TMU6048TrainY.txt 10 1 4 2
        
        argv[1] : filename for learning X (states) parameters
        argv[2] : filename for learning Y (observations) parameters
        argv[3] : nb of iterations for SEM-based learning
        argv[4] : nb of realizations for SEM-based learning
        argv[5] : Number of discrete fuzzy steps (so-called 'F' or 'STEPS')
        argv[6] : verbose (0/1/2)
    """

    print('Ligne de commandes : ', sys.argv, flush=True)

    if len(sys.argv) != 7:
        print('CAUTION : bad number of arguments - see help')
        exit(1)

    # Default value for parameters
    # fileTrainX = './Data/Traffic/Zied1/TMU6048TrainX.txt'
    # fileTrainY = './Data/Traffic/Zied1/TMU6048TrainY.txt'
    # nbIterSEM  = 10
    # nbRealSEM  = 1
    # STEPS      = 4
    # verbose    = 2

    # Parameters from argv
    fileTrainX = sys.argv[1]
    fileTrainY = sys.argv[2]
    nbIterSEM  = int(sys.argv[3])
    nbRealSEM  = int(sys.argv[4])
    STEPS      = int(sys.argv[5])
    verbose    = int(sys.argv[6])

    if verbose>0:
        print(' . fileTrainX =', fileTrainX)
        print(' . fileTrainY =', fileTrainY)
        print(' . nbIterSEM  =', nbIterSEM)
        print(' . nbRealSEM  =', nbRealSEM)
        print(' . STEPS      =', STEPS)
        print(' . verbose    =', verbose)
        print('\n')

    # Lecture des données
    Xtrain     = np.loadtxt(fileTrainX, delimiter=',').reshape(1, -1)
    Ytrain     = np.loadtxt(fileTrainY, delimiter=',').reshape(1, -1)
    n_x, len_x = np.shape(Xtrain)
    n_y, len_y = np.shape(Ytrain)
    if len_x != len_y:
        print('The number of values in the 2 files are differents!!!\n')
        exit(1)

    # plt.figure()
    # plt.plot(Ytrain[0,:], color='r', label='Ytrain')
    # plt.legend()
    # plt.savefig('Ytrain.png', bbox_inches='tight', dpi=150)    
    # plt.close()

    # plt.figure()
    # plt.plot(Xtrain[0,:], color='b', label='Xtrain')
    # plt.legend()
    # plt.savefig('Xtrain.png', bbox_inches='tight', dpi=150)    
    # plt.close()

    # Learning
    Ztrain = np.zeros(shape=(n_x+n_y, len_x))
    Ztrain[0  :n_x,     :] = Xtrain
    Ztrain[n_x:n_x+n_y, :] = Ytrain
    aCGOFMSM_learn = CGOFMSMlearn(STEPS, Ztrain, n_x, n_y, verbose)
    aCGOFMSM_learn.run_several(nbIterSEM, nbRealSEM)


def Check_CovMatrix(Mat):
    w, v = np.linalg.eig(Mat)
    # print("eig value:", w)
    # print(np.all(w>0.))
    # print(np.all(np.logical_not(np.iscomplex(w))))
    # input('pause')
    if np.all(np.logical_not(np.iscomplex(w))) == False or np.all(w>0.) == False:
        return False
    return True


from OFAResto.LoiDiscreteFuzzy_TMC    import Loi1DDiscreteFuzzy_TMC, Loi2DDiscreteFuzzy_TMC, calcF, calcB
from Fuzzy.APrioriFuzzyLaw_Series2ter import LoiAPrioriSeries2ter
from GeneMatCov                       import Check_CovMatrix

###################################################################################################
class CGOFMSMlearn:
    def __init__(self, STEPS, Ztrain, n_x, n_y, verbose):
        
        self.__n_r     = 2
        self.__N       = np.shape(Ztrain)[1]
        self.__Ztrain  = Ztrain
        self.__verbose = verbose
        self.__STEPS   = STEPS
        # Les constantes
        self.__EPS     = 1E-8

        self.__n_x     = n_x
        self.__n_y     = n_y
        self.__n_z     = self.__n_x + self.__n_y

        if self.__STEPS != 0:
            self.__Rcentres = np.linspace(start=1./(2.*self.__STEPS), stop=1.0-1./(2.*self.__STEPS), num=self.__STEPS, endpoint=True)
        else:
            self.__Rcentres = np.empty(shape=(0,))

        # used for simualtion of discretized fuzzy r (in [0..self.__STEPS+1])
        self.__Rsimul = np.zeros(shape=(self.__N), dtype=int)

        # Init of parameters : kmeans with 2+STEPS classes
        ###################################################
        kmeans = KMeans(n_clusters=2+self.__STEPS, random_state=4, init='random', n_init=200).fit(np.transpose(Ztrain))
        #kmeans = KMeans(n_clusters=2+self.__STEPS, random_state=None, init='random', n_init=200).fit(np.transpose(Ztrain))
        #kmeans = KMeans(n_clusters=2+self.__STEPS, n_init=200).fit(np.transpose(Ztrain))
        print(kmeans.inertia_, kmeans.n_iter_)
        #input('attente keamns ')
        # Sorting of labels according to the X-coordinate of cluster centers
        # print('kmeans.cluster_centers_=', kmeans.cluster_centers_)
        sortedlabel = np.argsort(kmeans.cluster_centers_[:, 0])
        #print('sortedlabel=', sortedlabel)
        for n in range(self.__N):
            self.__Rsimul[n] =  np.where(sortedlabel == kmeans.labels_[n])[0][0]
       
        # Parameter for fuzzy Markov model called APrioriFuzzyLaw_serie2ter.py
        self.__alpha0, self.__alpha1, self.__beta = self.EmpiricalFuzzyJointMatrix()
        self.__FS = LoiAPrioriSeries2ter(EPS=self.__EPS, discretization=0, alpha0=self.__alpha0, alpha1=self.__alpha1, beta=self.__beta)
        
        # Parameters of parametrization 3 (M=[A, B, C, D], Lambda**2, and P=[F, G], Pi**2)
        self.__M, self.__Lambda2, self.__P, self.__Pi2 = self.initParam_P3()
        if self.__verbose >= 1:
            print('Initialisation')
            print('  --> loi jointe à priori=', self.__FS)
            # print('  --> param du modele (param 3)\n    M=', self.__M, '\n    Lambda**2=', self.__Lambda2, '\n    P=', self.__P, '\n    Pi**2=', self.__Pi2)
            # print('  --> param du modele (param 3)\n    M=', self.__M)
            # print('  --> param du modele (param 3)\n    Lambda**2=', self.__Lambda2)
            # print('  --> param du modele (param 3)\n    P=', self.__P)
            # print('  --> param du modele (param 3)\n    Pi**2=', self.__Pi2)
            # if self.__verbose >= 2:
            #     input('pause')

    def getParams(self):
        return self.__alpha0, self.__alpha1, self.__beta, self.__FS.getEta()

    def EmpiricalFuzzyJointMatrix(self):
        
        alpha0, alpha1, beta = 0., 0., 0.
        for n in range(1, self.__N):
            if self.__Rsimul[n-1] == 0 and self.__Rsimul[n] == 0:
                alpha0 += 1.
            if self.__Rsimul[n-1] == self.__STEPS+1 and self.__Rsimul[n] == self.__STEPS+1:
                alpha1 += 1.
            if (self.__Rsimul[n-1] == 0 and self.__Rsimul[n] == self.__STEPS+1) or (self.__Rsimul[n-1] == self.__STEPS+1 and self.__Rsimul[n] == 0):
                beta   += 1.
        beta /= 2. # ca compte les transitions 0-1, 1-0, donc on divise par deux

        alpha0 /= (self.__N-1.)
        alpha1 /= (self.__N-1.)
        beta   /= (self.__N-1.)

        return alpha0, alpha1, beta


    def run_several(self, nbIterSEM, nbRealSEM):

        for i in range(nbIterSEM):
            print('ITERATION ', i)
            ok = self.run_one(nbRealSEM)
        return self.getParams()

    def run_one(self, nbRealSEM):

        # MAJ des proba sur la base des paramètres courants
        ProbaForwardNorm, tab_normalis      = self.compute_fuzzyjumps_forward ()
        ProbaBackwardNorm                   = self.compute_fuzzyjumps_backward(ProbaForwardNorm)
        ProbaGamma, ProbaPsi, ProbaJumpCond = self.compute_fuzzyjumps_gammapsicond(ProbaForwardNorm, ProbaBackwardNorm)
        # input('Fin proba FB')

        for r in range(nbRealSEM):

            # Simuler une chaines de Markov a posteriori sur la base de la valeur des paramètres courants
            print('  --> REALIZATION ', r)
            self.simulRealization(ProbaGamma[0], ProbaJumpCond)

            input('Fin realization')

            # Estimateurs empiriques

            input('Fin Estimateur empiriques')

        # Mean of parameters estimation

        return True


    def simulRealization(self, ProbaGamma_0, ProbaJumpCond):

        # sampling of the first from gamma
        np1=0
        self.__Rsimul[np1] = ProbaGamma_0.getSample()
        print('n=', np1, ', self.__Rsimul[np1]=', self.__Rsimul[np1])

        # next ones according to loi cond
        for np1 in range(1, self.__N):
            self.__Rsimul[np1] = ProbaJumpCond[np1].getSample(self.__Rsimul[np1]-1)
            print('n=', np1, ', self.__Rsimul[np1]=', self.__Rsimul[np1])

        input('end simulRealization')


    def compute_fuzzyjumps_forward(self):
        
        ProbaForward = []
        tab_normalis = np.zeros(shape=(self.__N))

        ######################
        # Initialisation

        # Parameters for the first p(r_1 | z_1)
        # necessaire pour le forward n=1
        aMeanCovFuzzy = MeanCovFuzzy(self.__Ztrain, self.__n_x, self.__n_y, self.__n_r, self.__STEPS, self.__Rcentres, self.__verbose)
        aMeanCovFuzzy.update(self.__Rsimul)

        np1  = 0
        znp1 = self.__Ztrain[:, np1]
        ProbaForward.append(Loi1DDiscreteFuzzy_TMC(self.__EPS, self.__STEPS, self.__Rcentres))
        ProbaForward[np1].CalcForw1(self.__FS, znp1, aMeanCovFuzzy)
        tab_normalis[np1] = ProbaForward[np1].Integ()
        # normalisation (devijver)
        ProbaForward[np1].normalisation(tab_normalis[np1])
        #ProbaForward[np1].print()
        #ProbaForward[np1].plot('$p(r_n | y_1^n)$')

        ###############################
        # Boucle
        for np1 in range(1, self.__N):
            if self.__verbose >= 2:
                print('\r         forward np1=', np1, ' sur N=', self.__N, end='', flush = True)

            zn   = self.__Ztrain[:, np1-1]
            znp1 = self.__Ztrain[:, np1]
            ProbaForward.append(Loi1DDiscreteFuzzy_TMC(self.__EPS, self.__STEPS, self.__Rcentres))
            #print('self.__Rcentres=', self.__Rcentres)
            ProbaForward[np1].CalcForB(calcF, ProbaForward[np1-1], self.__FS, self.__M, self.__Lambda2, self.__P, self.__Pi2, zn, znp1)
            #ProbaForward[np1].nextAfterZeros() # on evite des proba de zero
            # ProbaForward[np1].print()
            # print('sum forw=', ProbaForward[np1].Integ())
            tab_normalis[np1] = ProbaForward[np1].Integ()
            # normalisation (devijver)
            ProbaForward[np1].normalisation(tab_normalis[np1])
            # print('sum forw=', ProbaForward[np1].Integ())
            # ProbaForward[np1].print()
            #input('attent forward n=' + str(np1))

        if self.__verbose >= 2:
            print(' ')

        return ProbaForward, tab_normalis


    def compute_fuzzyjumps_backward(self, ProbaForwardNorm):

        # Proba backward
        ProbaBackward = []

        loinorm = Loi1DDiscreteFuzzy_TMC(self.__EPS, self.__STEPS, self.__Rcentres)

        # on créé la liste de tous les proba
        for n in range(self.__N):
            ProbaBackward.append(Loi1DDiscreteFuzzy_TMC(self.__EPS, self.__STEPS, self.__Rcentres))

        ##############################
        # initialisation de beta
        n  = self.__N-1
        # print('n=', n)
        # input('temp')
        ProbaBackward[n].setone_1D()
        # normalisation (devijver)
        loinorm.ProductFB(ProbaBackward[n], ProbaForwardNorm[n])
        #loinorm.nextAfterZeros() # on evite des proba de zero
        ProbaBackward[n].normalisation(loinorm.Integ())
        # ProbaBackward[n].print()
        # input('backward')

        ###############################
        # Boucle pour backward
        for n in range(self.__N-2, -1, -1):
            if self.__verbose >= 2:
                print('\r         backward n=', n, ' sur N=', self.__N, end='             ', flush = True)

            #print('n=', n)
            # input('temp')

            zn   = self.__Ztrain[:, n]
            znp1 = self.__Ztrain[:, n+1]
            # if n== 2791 or n== 2792:
            #     print('  n=', n)
            #     print('ProbaBackward[n+1]=')
            #     ProbaBackward[n+1].print()
            #     input('pb a venir')
            try:
                ProbaBackward[n].CalcForB(calcB, ProbaBackward[n+1], self.__FS, self.__M, self.__Lambda2, self.__P, self.__Pi2, zn, znp1)
            except:
                print('ProbaBackward[n+1]=', ProbaBackward[n+1])
                input('pb')

            #ProbaBackward[n].print()
            # print('sum back AVANT=', ProbaBackward[n].Integ())
            # input('pause')

            # normalisation (devijver)
            loinorm.ProductFB(ProbaForwardNorm[n], ProbaBackward[n])
            #loinorm.nextAfterZeros() # on evite des proba de zero
            if loinorm.Integ() == 0.:
                print('loinorm.Integ()=', loinorm.Integ())
                print('ProbaForwardNorm[n]=', ProbaForwardNorm[n])
                print('ProbaBackward[n]=', ProbaBackward[n])
                input('pb loinorm.Integ() == 0.')
            ProbaBackward[n].normalisation(loinorm.Integ())
            #ProbaBackward[n].print()
            # print('sum back=', ProbaBackward[n].Integ())
            #ProbaBackward[n].nextAfterZeros() # on evite des proba de zero
            #print('\n  -->sum backw=', ProbaBackward[n].Integ())
            #print('  -->tab_normalis[n+1]=', tab_normalis[n+1])
            #ProbaBackward[n].normalisation(tab_normalis[n+1])
            #print('  -->sum backw=', ProbaBackward[n].Integ())
            #ProbaBackward[n].plot('$p(r_{n+1} | y_1^{n+1})$')
            # ProbaBackward[n].print()
            # input('backward')

        if self.__verbose >= 2:
            print(' ')

        return ProbaBackward


    def compute_fuzzyjumps_gammapsicond(self, ProbaForward, ProbaBackward):

        tab_gamma = []
        tab_psi   = []
        tab_cond  = []

        ###############################
        # Boucle sur gamma et psi
        for n in range(self.__N-1):
            if self.__verbose >= 2:
                print('\r         proba gamma psi cond n=', n, ' sur N=', self.__N, end='   ', flush = True)

            # calcul du produit forward norm * backward norm
            tab_gamma.append(Loi1DDiscreteFuzzy_TMC(self.__EPS, self.__STEPS, self.__Rcentres))
            tab_gamma[n].ProductFB(ProbaForward[n], ProbaBackward[n])
            #tab_gamma[n].nextAfterZeros() # on evite des proba de zero

            # normalisation : uniquement due pour compenser des pb liés aux approximations numeriques de forward et de backward
            # Si F= 20, on voit que la normalisation n'est pas necessaire (deja la somme == 1.)
            if abs(1.-tab_gamma[n].Integ()) > 1E-3:
                print('tab_gamma[n].Integ()=', tab_gamma[n].Integ())
                input('PB PB PB Gamma')
                tab_gamma[n].normalisation(tab_gamma[n].Integ())

            # cacul de psi
            zn   = self.__Ztrain[:, n]
            znp1 = self.__Ztrain[:, n+1]
            tab_psi.append(Loi2DDiscreteFuzzy_TMC(self.__EPS, self.__STEPS, self.__Rcentres))
            tab_psi[n].CalcPsi(ProbaForward[n], ProbaBackward[n+1], self.__FS, self.__M, self.__Lambda2, self.__P, self.__Pi2, zn, znp1)

            # normalisation
            # print('tab_psi[n].Integ()=', tab_psi[n].Integ())
            
            integ = tab_psi[n].Integ()
            if integ == 0.:
                print("Tab Spi normalisation")
                tab_psi[n].print()
                input('pause')
            tab_psi[n].normalisation(tab_psi[n].Integ())
            # print('tab_psi[n].Integ()=', tab_psi[n].Integ())
            # input('STOP')

            # cacul de p(r_2 | r_1, z_1^M)
            # on créé une liste de lois 1D (c'est mieux que du 2D)
            Liste = []
            for qn in range(self.__STEPS+2):
                Liste.append(Loi1DDiscreteFuzzy_TMC(self.__EPS, self.__STEPS, self.__Rcentres))
                if   qn == 0:                 rn=0.
                elif qn == self.__STEPS+1:    rn=1.
                else:                         rn=self.__Rcentres[qn-1]
                # if 0, then Liste[qn] is all 0
                if tab_gamma[n].get(rn) != 0.:
                    Liste[qn].CalcCond(rn, tab_gamma[n].get(rn), tab_psi[n])

            tab_cond.append(Liste)

        # le dernier pour gamma
        n = self.__N-1
        tab_gamma.append(Loi1DDiscreteFuzzy_TMC(self.__EPS, self.__STEPS, self.__Rcentres))
        tab_gamma[n].ProductFB(ProbaForward[n], ProbaBackward[n])
        #tab_gamma[n].nextAfterZeros() # on evite des proba de zero
        if abs(1.-tab_gamma[n].Integ()) > 1E-3:
            print('tab_gamma[n].Integ()=', tab_gamma[n].Integ())
            input('PB PB PB Gamma')
            tab_gamma[n].normalisation(tab_gamma[n].Integ())

        if self.__verbose >= 2:
            print(' ')

        return tab_gamma, tab_psi, tab_cond


    def initParam_P3(self):

        # M = [A, B, C, D] ##################################################################################
        MNum     = np.zeros(shape=(self.__STEPS+2, self.__STEPS+2, 1, 4))
        MatNum   = np.zeros(shape=(1, 4))
        MDenom   = np.zeros(shape=(self.__STEPS+2, self.__STEPS+2, 4, 4))
        MatDenom = np.zeros(shape=(4, 4))
        # P = [F, G] ##################################################################################
        PNum      = np.zeros(shape=(self.__STEPS+2, self.__STEPS+2, 1, 2))
        PMatNum   = np.zeros(shape=(1, 2))
        PDenom    = np.zeros(shape=(self.__STEPS+2, self.__STEPS+2, 2, 2))
        PMatDenom = np.zeros(shape=(2, 2))

        for n in range(self.__N-1):
            qn    = self.__Rsimul[n]
            qnpun = self.__Rsimul[n+1]
            if   qn == 0:                 rn=0.
            elif qn == self.__STEPS+1:    rn=1.
            else:                         rn=self.__Rcentres[qn-1]
            if   qnpun == 0:              rnpun=0.
            elif qnpun == self.__STEPS+1: rnpun=1.
            else:                         rnpun=self.__Rcentres[qnpun-1]
            #print('qn=', qn, ', qn+1=', qnpun, ', rn=', rn, ', rnpun=', rnpun)
            # input('attente')

            zn    = self.__Ztrain[:, n].reshape((2, 1))
            yn    = (self.__Ztrain[self.__n_x:self.__n_z, n])  .item()
            ynpun = (self.__Ztrain[self.__n_x:self.__n_z, n+1]).item()
            xn    = (self.__Ztrain[0:self.__n_x, n])           .item()
            xnpun = (self.__Ztrain[0:self.__n_x, n+1])         .item()

            ################## M ########################
            MatDenom[0:2, 0:2] = np.dot(zn, np.transpose(zn))
            MatDenom[0:2, 2:3] = zn * ynpun
            MatDenom[2:3, 0:2] = np.transpose(MatDenom[0:2, 2:3])
            MatDenom[0:2, 3:4] = zn
            MatDenom[3:4, 0:2] = np.transpose(MatDenom[0:2, 3:4])
            MatDenom[2:3, 2:3] = ynpun*ynpun
            MatDenom[2:3, 3:4] = ynpun
            MatDenom[3:4, 2:3] = np.transpose(MatDenom[2:3, 3:4])
            MatDenom[3:4, 3:4] = 1.
            #print('MatDenom=', MatDenom)
            MDenom[qn, qnpun, :, :] += self.__FS.probaR1R2(rn, rnpun) * MatDenom
            MatNum[0, 0] = xn 
            MatNum[0, 1] = yn 
            MatNum[0, 2] = ynpun 
            MatNum[0, 3] = 1. 
            #print(np.shape(MatNum))
            # print(np.shape(self.__Ztrain[0:self.__n_x, n+1]))
            # print(np.shape(self.__Ztrain[0:self.__n_x, n+1] * MatNum))
            # print(self.__FS.probaR1R2(self.__Rcentres[qn], self.__Rcentres[qnpun] ))
            #print('MatNum=', MatNum)
            MNum[qn, qnpun, :, :] += self.__FS.probaR1R2(rn, rnpun) * xnpun * MatNum
            #print('MNum=', MNum)
            #input('initParam_P3 - M')

            ################## P ########################
            PMatDenom[0, 0] = yn*yn
            PMatDenom[0, 1] = yn
            PMatDenom[1, 0] = np.transpose(PMatDenom[0, 1])
            PMatDenom[1, 1] = 1.
            #print('PMatDenom=', PMatDenom)
            PDenom[qn, qnpun, :, :] += self.__FS.probaR1R2(rn, rnpun) * PMatDenom
            PMatNum[0, 0] = yn 
            PMatNum[0, 1] = 1. 
            PNum[qn, qnpun, :, :] += self.__FS.probaR1R2(rn, rnpun) * ynpun * PMatNum
            #print('PNum=', PNum)
            #input('initParam_P3 - P')
        
        M = np.zeros(shape=(self.__STEPS+2, self.__STEPS+2, 1, 4))
        P = np.zeros(shape=(self.__STEPS+2, self.__STEPS+2, 1, 2))
        for q1 in range(self.__STEPS+2):
            for q2 in range(self.__STEPS+2):
                try:
                    numinverse = np.linalg.inv(MDenom[q1, q2, :, :])
                    M[q1, q2, :, :] = np.dot(MNum[q1, q2, :, :], numinverse)
                except np.linalg.LinAlgError:
                    M[q1, q2, :, :] = 0.

                try:
                    numinverse = np.linalg.inv(PDenom[q1, q2, :, :])
                    P[q1, q2, :, :] = np.dot(PNum[q1, q2, :, :], numinverse)
                except np.linalg.LinAlgError:
                    P[q1, q2, :, :] = 0.


        # Pi2 and Lambda2 ##################################################################################
        Pi2     = np.zeros(shape=(self.__STEPS+2, self.__STEPS+2))
        Lambda2 = np.zeros(shape=(self.__STEPS+2, self.__STEPS+2))
        Denom   = np.zeros(shape=(self.__STEPS+2, self.__STEPS+2))
        for n in range(self.__N-1):
            qn    = self.__Rsimul[n]
            qnpun = self.__Rsimul[n+1]
            if   qn == 0:                 rn=0.
            elif qn == self.__STEPS+1:    rn=1.
            else:                         rn=self.__Rcentres[qn-1]
            if   qnpun == 0:              rnpun=0.
            elif qnpun == self.__STEPS+1: rnpun=1.
            else:                         rnpun=self.__Rcentres[qnpun-1]
            #print('qn=', qn, ', qn+1=', qnpun, ', rn=', rn, ', rnpun=', rnpun)
            # input('attente')

            yn    = (self.__Ztrain[self.__n_x:self.__n_z, n])  .item()
            ynpun = (self.__Ztrain[self.__n_x:self.__n_z, n+1]).item()
            xn    = (self.__Ztrain[0:self.__n_x, n])           .item()
            xnpun = (self.__Ztrain[0:self.__n_x, n+1])         .item()

            proba = self.__FS.probaR1R2(rn, rnpun)
            Lambda2[qn, qnpun] += proba * (xnpun - (xn * M[q1, q2, 0, 0] + yn * M[q1, q2, 0, 1] + ynpun * M[q1, q2, 0, 2] + M[q1, q2, 0, 3]))**2
            Pi2[qn, qnpun]     += proba * (ynpun - (yn * P[q1, q2, 0, 0] + P[q1, q2, 0, 1]))**2
            Denom[qn, qnpun]   += proba

        for q1 in range(self.__STEPS+2):
            for q2 in range(self.__STEPS+2):
                if Pi2[q1, q2] != 0.:
                    Pi2    [q1, q2] /= Denom[q1, q2]
                if Lambda2[q1, q2] != 0.:
                    Lambda2[q1, q2] /= Denom[q1, q2]
        # print('Pi2=', Pi2)
        # print('Lambda2=', Lambda2)
        # input('pause')

        return M, Lambda2, P, Pi2


###################################################################################################
class MeanCovFuzzy:

    def __init__(self, Ztrain, n_x, n_y, n_r, STEPS, Rcentres, verbose):
        self.__n_r      = n_r
        self.__verbose  = verbose
        self.__STEPS    = STEPS
        self.__Rcentres = Rcentres
        self.__Ztrain  = Ztrain

        self.__n_x      = n_x
        self.__n_y      = n_y
        self.__n_z      = self.__n_x + self.__n_y

        self.__Mean_Zf  = np.zeros(shape=(self.__STEPS+2, self.__n_z))
        self.__Cov_Zf   = np.zeros(shape=(self.__STEPS+2, self.__n_z, self.__n_z))

    def update(self, Rlabels):

        N   = np.shape(Rlabels)[0]
        cpt = np.zeros(shape=(self.__STEPS+2))
    
        # The means
        for n in range(N):
            self.__Mean_Zf[Rlabels[n], :] += self.__Ztrain[:, n]
            cpt[Rlabels[n]] += 1.
        for q in range(self.__STEPS+2):
            self.__Mean_Zf[q, :] /= cpt[q]
        # print(self.__Mean_Zf)
        # input('hard mean')
        
        # The variance
        VectZ = np.zeros(shape=(self.__n_z, 1))
        for n in range(N):
            VectZ = (np.transpose(self.__Ztrain[:, n]) - self.__Mean_Zf[Rlabels[n], :]).reshape(self.__n_z, 1)
            self.__Cov_Zf[Rlabels[n], :, :] += np.dot(VectZ, np.transpose(VectZ))
        for q in range(self.__STEPS+2):
            self.__Cov_Zf[q, :, :] /= cpt[q]
            # check if cov matrix
            if Check_CovMatrix( self.__Cov_Zf[q, :, :]) == False:
                print(self.__Cov_Zf[q, :, :])
                input('This is not a cov matrix!!')

        # print(self.__Cov_Zf)
        # input('hardcov')

    def getMean(self, q):
        return self.__Mean_Zf[q, :]

    def getCov(self, q):
        return self.__Cov_Zf[q, :]

if __name__ == '__main__':
    main()
