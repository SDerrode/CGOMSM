#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy as sc
import sys

import warnings

from CommonFun.CommonFun import Readin_CovMeansProba, Readin_ABMeansProba, From_Cov_to_FQ, From_FQ_to_Cov_Lyapunov, ImpressionMatLatex


def main():

    """
        Programmes pour vérifier et convertir les matirices de cov CGPMSM

        :Examples:

        >> python3 Convert_Cov_FQ.py 1 Parameters/DFBSmoothing/AB.param
        >> python3 Convert_Cov_FQ.py 0 Parameters/DFBSmoothing/Cov_CGPMSM.param

        argv[1] : 0 means reading Cov, 1 means reading A,B
        argv[2] : Nom du fichier de paramètres
    """

    print(sys.argv)
    if len(sys.argv) == 3:
        coding        = int(sys.argv[1])
        filenameParam = sys.argv[2]
    else:
        coding        = 0
        filenameParam = 'Parameters/DFBSmoothing/test5.param'
        #filenameParam   = 'Parameters/Fuzzy/SP2018.param'

    print('Command line : python3', coding, filenameParam)

    if coding == 0:
        liste_arg = Readin_CovMeansProba(filenameParam)
    else:
        liste_arg = Readin_ABMeansProba(filenameParam)

    n_r    = liste_arg[0]
    A      = liste_arg[1]
    B      = liste_arg[2]
    Q      = liste_arg[3]
    Cov    = liste_arg[4]
    Mean_X = liste_arg[5]

    useless, n_z_mp2, useless2 = np.shape(Cov)
    n_z = n_z_mp2//2

    # Impression pour simplifier le copier-coller
    Impression     (Cov, A, B, n_r, n_z, decim=7)
    ImpressionLatex(Cov, A, B, n_r, n_z, decim=3)

    # Conversion arrière
    n_x = np.shape(Mean_X)[1]
    CovBis = From_FQ_to_Cov_Lyapunov(A, Q, n_x)

    if np.allclose(Cov, CovBis) == False:
        # print('Cov AVANT=', Cov)
        # print('Cov APRES=', CovBis)
        print('PROBLEM:')
        print('Cov DIFF=', np.around(Cov-CovBis, decimals=2))



def ImpressionLatex(Cov, A, B, n_r, n_z, decim=3):

    print('# matrice Cov')
    ImpressionMatLatex(Cov, '\\Gamma', n_r, n_z*2, decim=decim)
    print('\n# matrice A')
    ImpressionMatLatex(A, 'A', n_r, n_z, decim=decim)
    print('\n# matrice B')
    ImpressionMatLatex(B, 'B', n_r, n_z, decim=decim)

def Impression(Cov, A, B, n_r, n_z, decim=3):

    # pour les matrices Cov
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

    # pour les matrices A
    print('# process control matrix A')
    print('# ===============================================#')
    print('#')
    for i in range(n_r**2):
        j = i//n_r
        k = i%n_r
        print('# A%d%d'%(j, k))
        print('# ----------------------------#')
        for l in range(n_z):
            for m in range(n_z):
                print(np.around(A[i,l,m], decimals=decim), end=' ')
            print(end='\n')
        print('#')

    # pour les matrices B
    print('# noise control matrix B')
    print('# ===============================================#')
    print('#')
    for i in range(n_r**2):
        j = i//n_r
        k = i%n_r
        print('# B%d%d'%(j, k))
        print('# ----------------------------#')
        for l in range(n_z):
            for m in range(n_z):
                print(np.around(B[i,l,m], decimals=decim), end=' ')
            print(end='\n')
        print('#')


if __name__ == '__main__':
    main()
