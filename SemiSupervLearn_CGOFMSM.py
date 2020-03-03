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
        argv[4] : nb of realization for SEM-based learning
        argv[5] : Number of discrete fuzzy steps (so-called 'F' or STEPS)
        argv[6] : verbose (0/1/2)
    """

    print('Ligne de commandes : ', sys.argv, flush=True)

    if len(sys.argv) != 7:
        print('CAUTION : bad number of arguments - see help')
        exit(1)

    # Default value for parameters
    fileTrainX = './Data/Traffic/Zied1/TMU6048TrainX.txt'
    fileTrainY = './Data/Traffic/Zied1/TMU6048TrainY.txt'
    nbIterSEM  = 10
    nbRealSEM  = 1
    STEPS      = 4
    verbose    = 2

    # Parameters form argv
    fileTrainX = sys.argv[1]
    fileTrainY = sys.argv[2]
    nbIterSEM  = int(sys.argv[3])
    nbRealSEM  = int(sys.argv[4])
    STEPS      = int(sys.argv[5])
    verbose    = int(sys.argv[6])
   
    print(' . fileTrainX =', fileTrainX)
    print(' . fileTrainY =', fileTrainY)
    print(' . nbIterSEM  =', nbIterSEM)
    print(' . nbRealSEM  =', nbRealSEM)
    print(' . STEPS      =', STEPS)
    print(' . verbose    =', verbose)
    print('\n')

    # Lecture des données
    Xtrain = np.loadtxt(fileTrainX, delimiter=',').reshape(1, -1)
    Ytrain = np.loadtxt(fileTrainY, delimiter=',').reshape(1, -1)
    n_x, len_x = np.shape(Xtrain)
    n_y, len_y = np.shape(Ytrain)
    if len_x != len_y:
        print('The number of values of the two files are differents!!!\n')
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



from OFAResto.LoiDiscreteFuzzy_HMC import Loi1DDiscreteFuzzy_HMC, calcF, calcB
from Fuzzy.APrioriFuzzyLaw_Series2ter import LoiAPrioriSeries2ter

###################################################################################################
class CGOFMSMlearn:
    def __init__(self, STEPS, Ztrain, n_x, n_y, verbose):
        
        self.__n_r     = 2
        self.__N       = np.shape(Ztrain)[1]
        self.__Ztrain  = Ztrain
        self.__verbose = verbose
        self.__STEPS   = STEPS
        # Les constantes
        self.__EPS = 1E-8

        self.__n_x    = n_x
        self.__n_y    = n_y
        self.__n_z    = self.__n_x + self.__n_y

        if self.__STEPS != 0:
            self.__Rcentres = np.linspace(start=1./(2.*self.__STEPS), stop=1.0-1./(2.*self.__STEPS), num=self.__STEPS, endpoint=True)
        else:
            self.__Rcentres = np.empty(shape=(0,))

        # Init of parameters : kmeans with 2+STEPS hard classes
        kmeans = KMeans(n_clusters=2+self.__STEPS, random_state=0).fit(np.transpose(self.__Ztrain))
        I      = np.argsort(kmeans.cluster_centers_[:, 0])
        hard0  = I[0]   # le plus petit
        hard1  = I[-1]  # le plus grand
        
        # Parameter for fuzzy markov model called APrioriFuzzyLaw_serie2ter.py
        self.__alpha0, self.__alpha1, self.__beta = self.EmpiricalFuzzyJointMatrix(kmeans.labels_, hard0, hard1)
        self.__FS = LoiAPrioriSeries2ter(self.__EPS, discretization=0, alpha0=self.__alpha0, alpha1=self.__alpha1, beta=self.__beta)
        #print(self.__FS)
        #input('pause')

        # Parameters for class conditional densities
        self.__CovZ, self.__MeanZ = self.EmpiricalClassCond(kmeans.labels_, hard0, hard1)
        print('self.__CovZ=', self.__CovZ)
        # print('self.__MeanZ=', self.__MeanZ)
        # input('pause')

        # from Fuzzy.InterFuzzy import InterLineaire_Matrix, InterLineaire_Vector
        # r = np.linspace(0, 1, 10)
        # print('r=',r)
        # for i in r:
        #     CovZ_rnp1  = InterLineaire_Matrix(self.__CovZ,  i)
        #     print('i=', i)
        #     print('CovZ_rnp1=', CovZ_rnp1)
        # input('attente')

    def getParams(self):
        return self.__alpha0, self.__alpha1, self.__beta, self.__FS.getEta()

    def EmpiricalFuzzyJointMatrix(self, Rsimul, hard0, hard1):
        
        alpha0, alpha1, beta = 0., 0., 0.
        
        for n in range(1, self.__N):
            if Rsimul[n-1] == hard0 and Rsimul[n] == hard0:
                alpha0 += 1
            if Rsimul[n-1] == hard1 and Rsimul[n] == hard1:
                alpha1 += 1
            if (Rsimul[n-1] == hard0 and Rsimul[n] == hard1) or (Rsimul[n-1] == hard1 and Rsimul[n] == hard0):
                beta +=1
        beta /= 2. # ca compte les transitions 0-1, 1-0, donc on divise par deux

        alpha0 /= self.__N
        alpha1 /= self.__N
        beta   /= self.__N

        return alpha0, alpha1, beta


    def EmpiricalClassCond(self, Rsimul, hard0, hard1):
        
        # The means
        Mean_Z = np.zeros(shape=(self.__n_r, self.__n_z))
        cpt = np.zeros(shape=(self.__n_r))
        for n in range(self.__N):
            if (Rsimul[n] == hard0 or Rsimul[n] == hard1):
                if Rsimul[n] == hard0: q=0
                if Rsimul[n] == hard1: q=1
                Mean_Z[q, :] += self.__Ztrain[:, n]
                cpt[q] += 1.
        Mean_Z[0, :] /= cpt[0]
        Mean_Z[1, :] /= cpt[1]
        
        # The variance
        Cov_Z = np.zeros(shape=(self.__n_r, self.__n_z, self.__n_z))
        VectZ = np.zeros(shape=(self.__n_z, 1))
        for n in range(self.__N):
            if (Rsimul[n] == hard0 or Rsimul[n] == hard1):
                if Rsimul[n] == hard0: q=0
                if Rsimul[n] == hard1: q=1
                VectZ = (np.transpose(self.__Ztrain[:, n]) - Mean_Z[q, :]).reshape(self.__n_z, 1)
                Cov_Z[q, :] += np.dot(VectZ, np.transpose(VectZ))
        Cov_Z[0, :] /= cpt[0]
        Cov_Z[1, :] /= cpt[1]

        return Cov_Z, Mean_Z


    def run_several(self,  nbIterSEM, nbRealSEM):

        for i in range(nbIterSEM):
            print('ITERATION ', i)
            ok = self.run_one(nbRealSEM)

        return self.getParams()


    def run_one(self, nbRealSEM):

        # Simuler une chaines de Markov a posteriori
        ##############################################
        # Proba sauts
        ProbaForward, tab_normalis = self.compute_fuzzyjumps_forward ()
        ProbaBackward              = self.compute_fuzzyjumps_backward(tab_normalis)
        ProbaLissage               = self.compute_fuzzyjumps_smooth  (ProbaForward, ProbaBackward)

        input('Fin proba FB')

        for r in range(nbRealSEM):

            # Simuler une chaînes de Markov a posteriori
            print('  --> REALIZATION ', r)

            input('Fin realization')

            # Estimateurs empiriques

            input('Fin Estimateur empiriques')

        # Mean of parameters estimation

        return True


    def compute_fuzzyjumps_forward(self):
        
        ProbaForward = []
        tab_normalis = []

        ######################
        # Initialisation
        np1  = 0
        znp1 = self.__Ztrain[:, np1]
        ProbaForward.append(Loi1DDiscreteFuzzy_HMC(self.__EPS, self.__Rcentres))
        ProbaForward[np1].CalcForw1(self.__FS.probaR, znp1, self.__CovZ, self.__MeanZ)

        tab_normalis.append(ProbaForward[np1].sum())
        ProbaForward[np1].normalisation(tab_normalis[np1])
        #ProbaForward[np1].print()
        #ProbaForward[np1].plot('$p(r_n | y_1^n)$')

        ###############################
        # Boucle
        for np1 in range(1, self.__N):
            if self.__verbose >= 2:
                print('\r         forward np1=', np1, ' sur N=', self.__N, end='', flush = True)

            znp1 = self.__Ztrain[:, np1]
            ProbaForward.append(Loi1DDiscreteFuzzy_HMC(self.__EPS, self.__Rcentres))
            ProbaForward[np1].CalcForB(calcF, ProbaForward[np1-1], self.__FS, znp1, self.__CovZ, self.__MeanZ)
            ProbaForward[np1].nextAfterZeros() # on evite des proba de zero
            #ProbaForward[np1].print()
            # print('sum forw=', ProbaForward[np1].sum())
            tab_normalis.append(ProbaForward[np1].sum())
            ProbaForward[np1].normalisation(tab_normalis[np1])
            # print('sum forw=', ProbaForward[np1].sum())
            # ProbaForward[np1].print()
            # input('attent forward')

        if self.__verbose >= 2:
            print(' ')

        return ProbaForward, tab_normalis


    def compute_fuzzyjumps_backward(self, tab_normalis):

        # Proba backward
        ProbaBackward = []

        # on créé la liste
        for n in range(self.__N):
            ProbaBackward.append(Loi1DDiscreteFuzzy_HMC(self.__EPS, self.__Rcentres))

        ######################
        # initialisation de beta
        n  = self.__N-1
        # print('n=', n)
        # input('temp')
        ProbaBackward[n].setone_1D()
        # ProbaBackward[n].print()
        # input('backward')

        ###############################
        # Boucle pour backward
        for n in range(self.__N-2, -1, -1):
            if self.__verbose >= 2:
                print('\r         backward n=', n, ' sur N=', self.__N, end='             ', flush = True)

            # print('n=', n)
            # input('temp')

            znp1 = self.__Ztrain[:, n+1]
            ProbaBackward[n].CalcForB(calcB, ProbaBackward[n+1], self.__FS, znp1, self.__CovZ, self.__MeanZ)
            #ProbaBackward[n].nextAfterZeros() # on evite des proba de zero
            print('sum backw=', ProbaBackward[n].sum())
            print('tab_normalis[n+1]=', tab_normalis[n+1])
            ProbaBackward[n].normalisation(tab_normalis[n+1])
            print('sum backw=', ProbaBackward[n].sum())

            #ProbaBackward[n].plot('$p(r_{n+1} | y_1^{n+1})$')
            # ProbaBackward[n].print()
            input('backward')

        if self.__verbose >= 2:
            print(' ')

        return ProbaBackward


    def compute_fuzzyjumps_smooth(self, ProbaForward, ProbaBackward):

        tab_p_rn_dp_y1_to_yN = []

        ###############################
        # Boucle sur backward
        for n in range(self.__N):
            if self.__verbose >= 2:
                print('\r         proba lissage n=', n, ' sur N=', self.__N, end='   ', flush = True)

            # calcul du produit forward * backward
            tab_p_rn_dp_y1_to_yN.append(Loi1DDiscreteFuzzy_HMC(self.__EPS, self.__Rcentres))
            tab_p_rn_dp_y1_to_yN[n].ProductFB(ProbaForward[n], ProbaBackward[n])

            # normalisation : uniquement due pour compenser des pb liés aux approximations numeriques de forward et de backward
            # Si F= 20, on voit que la normalisation n'est pas necessaire (deja la somme == 1.)
            if tab_p_rn_dp_y1_to_yN[n].sum() > 1+1E-3 or tab_p_rn_dp_y1_to_yN[n].sum()< 1.-1E-3:
                print('tab_p_rn_dp_y1_to_yN[n].sum()=', tab_p_rn_dp_y1_to_yN[n].sum())
                input('Attente')
            tab_p_rn_dp_y1_to_yN[n].normalisation(tab_p_rn_dp_y1_to_yN[n].sum())
            #tab_p_rn_dp_y1_to_yN[n].print()
            #print('sum gamma=', tab_p_rn_dp_y1_to_yN[n].sum())
            #print('sum=', tab_p_rn_dp_y1_to_yN[n].sum())
            #input('Attente')

        if self.__verbose >= 2:
            print(' ')

        return tab_p_rn_dp_y1_to_yN

if __name__ == '__main__':
    main()
