#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
import os, sys
#sys.path.insert(0, os.path.abspath("."))

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from CGPMSMs.CGPMSMs import GetParamNearestCGO_cov, GetBackCov_CGPMSM
from CommonFun.CommonFun import Test_isCGOMSM_from_Cov, From_Cov_to_FQ

mpl.rc('text', usetex=True)
mpl.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

import time
plt.close('all')
dpi = 100 #300


def main():

    """
        Programmes pour generer des matrices de covariance qui forment un CGPMSM à n_r sauts".

        :Example:

        >> python3 GeneMatCov.py 2 CGP1.param

        argv[1] : number of jumps
        argv[2] : Parameters file name

    """

    if len(sys.argv) == 3:
        n_r           = int(sys.argv[1])
        filenameParam = sys.argv[2]
    else:
        # Default value for parameters
        n_r           = 2
        filenameParam = './Parameters/DFBSmoothing/test.param'


    print('Command line : python3', n_r, filenameParam)

    seed = random.randrange(sys.maxsize)
    # seed = 5876911974124870942
    random.seed(seed)
    print("Seed was:", seed)

    n_x = 1
    n_y = 1
    n_z = n_x + n_y
    s_xz = slice(n_x, n_z)
    s_0x = slice(0,   n_x)
    s_0z = slice(0,   n_z)
    s_z2z = slice(n_z, 2*n_z)

    MatSigma = np.zeros((n_r,    n_z, n_z))
    MatCov   = np.zeros((n_r**2, 2*n_z, 2*n_z))
    MatCorr  = np.zeros((n_r**2, 2*n_z, 2*n_z))

    # Modele Cov_CGPMSM3
    # MatSigma[0, 0, :] = [0.5000, 0.3000]
    # MatSigma[0, 1, :] = [0.3000, 1.0000]
    # MatSigma[1, 0, :] = [1.0000, 0.1500]
    # MatSigma[1, 1, :] = [0.1500, 0.5000]
    # Modele Cov_CGPMSM3 - n_r == 2
    # MatSigma[0, 0, :] = [0.5000, 0.3000/3.]
    # MatSigma[0, 1, :] = [0.3000/3., 1.0000]
    # MatSigma[1, 0, :] = [1.0000, 0.1500/3.]
    # MatSigma[1, 1, :] = [0.1500/3., 0.5000]
    # Cov_CGPMSM.param
    MatSigma[0, 0, :] = [0.5000, 0.3000]
    MatSigma[0, 1, :] = [0.3000, 2.0000]
    MatSigma[1, 0, :] = [1.0000, 0.5000]
    MatSigma[1, 1, :] = [0.5000, 0.5000]

    for i in range(n_r):
        for j in range(n_r):
            l=i*n_r+j
            MatCov[l, s_0z, s_0z]   = MatSigma[i, :, :]
            MatCov[l, s_z2z, s_z2z] = MatSigma[j, :, :]
            # MatCorr
            for p in range(2*n_z):
                for q in range(2*n_z):
                    MatCorr[l, p, q] = MatCov[l, p, q]/np.sqrt(MatCov[l, p, p] * MatCov[l, q, q])
    # print('Array MatCorr = ', MatCorr)
    # input('pause')

    OK = False
    cpt = 0
    while OK == False:
        cpt = cpt + 1
        print('******************* cpt=', cpt)

        # print('  Fill des Mat Cov CGPMSM Direct')
        MatCov = Fill_MatCovCGPMSM(MatCorr, MatCov, n_r, n_z)
        # print('  TEST des Mat Cov CGPMSM Direct')
        OK1 = Check_CovMatrices(MatCov, n_r)
        print('   ->', OK1)
        if OK1 == True:
            print('---> OK1 = true, cpt=', cpt)

        # print('  Calcul des Mat Cov CGPMSM Reverse')
        MatCov_REV = GetBackCov_CGPMSM(MatCov, n_x=n_x)
        # print('  TEST des Mat Cov CGPMSM Reverse')
        OK2 = Check_CovMatrices(MatCov_REV, n_r)
        if OK2 == True:
            print('---> OK2 = true, cpt=', cpt)

        # print('  Calcul des Mat Cov CGO Direct')
        Cov_CGOMSM_DIR = GetParamNearestCGO_cov(MatCov, n_x=n_x)
        P1 = Test_isCGOMSM_from_Cov(Cov_CGOMSM_DIR, n_x)
        if P1 == False:
            print('ATTENTION: Le modele directe n''est pas un CGOMSM!!!')
            print('Cov_CGOMSM_DIR=\n', Cov_CGOMSM_DIR)
            input('pause')
        else:
            print('Le modele direct est un CGOMSM.')
        # print('   TEST des Mat Cov CGO Direct')
        OK3 = Check_CovMatrices(Cov_CGOMSM_DIR, n_r)
        if OK3 == True:
            print('---> OK3 = true, cpt=', cpt)

        # print('  Calcul des Mat Cov CGO Reverse')
        Cov_CGOMSM_REV = GetParamNearestCGO_cov(MatCov_REV, n_x=n_x)
        print('Array Cov_CGOMSM_REV = \n', Cov_CGOMSM_REV)
        F_CGO_REV, Q_CGO_REV = From_Cov_to_FQ(Cov_CGOMSM_REV)
        print('Array F_CGO_REV = \n', F_CGO_REV)
        input('pause')
        P2 = Test_isCGOMSM_from_Cov(Cov_CGOMSM_REV, n_x)
        if P2 == False:
            print('ATTENTION: Le modele reverse n''est pas un CGOMSM!!!')
            print('Cov_CGOMSM_REV=\n', Cov_CGOMSM_REV)
            input('pause')
        else:
            print('Le modele reverse est un CGOMSM.')
        # print('TEST des Mat Cov CGO Reverse')
        OK4 = Check_CovMatrices(Cov_CGOMSM_REV, n_r)
        if OK4 == True:
            print('---> OK4 = true, cpt=', cpt)

        OK = OK1 and OK2 and OK3 and OK4
        if OK == True:
            print('  OK final = ', OK)
            input('  pause')

    # Impression des matcov
    Impression(MatCov, n_r, n_z)


def Fill_MatCovCGPMSM(MatCorr, MatCov, n_r, n_z):

    for l in range(n_r**2):
        # on rempli avec des nombres au hasard
        MatCorr[l, 0, 2] = random.random()*0.5 #- 0.5 #1. - 0.5
        MatCorr[l, 0, 3] = random.random()*0.5 #- 0.5 #1. - 0.5
        MatCorr[l, 1, 2] = random.random()*0.5 #- 0.5 #1. - 0.5
        MatCorr[l, 1, 3] = random.random()*0.5 #- 0.5 #1. - 0.5

        MatCorr[l, 2:4, 0:2] = np.transpose(MatCorr[l, 0:2, 2:4])

        # MatCorr
        diagonale = np.diag(MatCov[l, :, :])
        for p in range(2*n_z):
            for q in range(2*n_z):
                MatCov[l, p, q] = MatCorr[l, p, q] * np.sqrt(diagonale[p] * diagonale[q])

    # print('Array MatCorr = ', MatCorr)
    # print('Array MatCov = ', MatCov)
    # input('pause')
    return MatCov

def Check_CovMatrix(MatCov):

    w, v = np.linalg.eig(MatCov)

    # print("eig value:", w)
    # print(np.all(w>0.))
    # print(np.all(np.logical_not(np.iscomplex(w))))
    # input('pause')

    if np.all(np.logical_not(np.iscomplex(w))) == False or np.all(w>0.) == False:
        return False

    return True

def Check_CovMatrices(MatCov, n_r):

    for l in range(n_r**2):

        if Check_CovMatrix(MatCov[l, :, :]) == False:
            return False

    return True



def Impression(Cov, n_r, n_z, decim=3):

    print('# Cov')
    print('# ===============================================#')
    print('#')
    for i in range(n_r**2):
        j = i//n_r
        k = i%n_r
        print('# Cov_xy%d%d'%(j, k))
        print('# ----------------------------#')
        for l in range(2*n_z):
            for m in range(2*n_z):
                print(np.around(Cov[i,l,m], decimals=decim), end=' ')
            print(end='\n')
        print('#')

if __name__ == '__main__':
    main()
