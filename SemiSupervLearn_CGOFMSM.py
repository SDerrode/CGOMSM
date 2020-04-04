#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import sys
import pathlib

import matplotlib
import matplotlib.pyplot as plt

from OFAResto.CGOFMSM_SemiSupervLearn import CGOFMSM_SemiSupervLearn, MeanCovFuzzy

def main():

    """
        Programmes pour estimer les paramètres d'un CGOFMSM, lorsque l'on connait un échantillon Z=(X,Y).
 
        :Example:

        >> python3 SemiSupervLearn_CGOFMSM.py ./Data/Traffic/TMU5509/generated/TMU5509_trainX_300.txt ./Data/Traffic/TMU5509/generated/TMU5509_trainY_300.txt 10 1 4 2 0
        >> python3 SemiSupervLearn_CGOFMSM.py ./Data/Traffic/TMU5509/generated/TMU5509_trainX.txt ./Data/Traffic/TMU5509/generated/TMU5509_trainY.txt 10 1 4 2 0
        
        argv[1] : filename for learning X (states) parameters
        argv[2] : filename for learning Y (observations) parameters
        argv[3] : nb of iterations for SEM-based learning
        argv[4] : nb of realizations for SEM-based learning
        argv[5] : number of discrete fuzzy steps (so-called 'F' or 'STEPS')
        argv[6] : verbose (0/1/2)
        argv[7] : plot the graphics (0:none/1:few/2:many/3:all)
    """

    print('Ligne de commandes : ', sys.argv, flush=True)

    if len(sys.argv) != 8:
        print('CAUTION : bad number of arguments - see help')
        exit(1)

    # Default value for parameters
    # fileTrainX = './Data/Traffic/TMU5509/generated/TMU5509_trainX.txt'
    # fileTrainY = './Data/Traffic/TMU5509/generated/TMU5509_trainY.txt'
    # nbIterSEM  = 10
    # nbRealSEM  = 1
    # STEPS      = 4
    # verbose    = 2
    # graphics   = 1

    # Parameters from argv
    fileTrainX = sys.argv[1]
    fileTrainY = sys.argv[2]
    nbIterSEM  = int(sys.argv[3])
    nbRealSEM  = int(sys.argv[4])
    STEPS      = int(sys.argv[5])
    verbose    = int(sys.argv[6])
    graphics   = int(sys.argv[7])

    if verbose>0:
        print(' . fileTrainX =', fileTrainX)
        print(' . fileTrainY =', fileTrainY)
        print(' . nbIterSEM  =', nbIterSEM)
        print(' . nbRealSEM  =', nbRealSEM)
        print(' . STEPS      =', STEPS)
        print(' . verbose    =', verbose)
        print(' . graphics   =', graphics)
        print('\n')

    # Lecture des données
    Xtrain     = np.loadtxt(fileTrainX, delimiter=',').reshape(1, -1)
    Ytrain     = np.loadtxt(fileTrainY, delimiter=',').reshape(1, -1)
    n_x, len_x = np.shape(Xtrain)
    n_y, len_y = np.shape(Ytrain)
    if len_x != len_y:
        print('The number of values in the 2 files are differents!!!\n')
        exit(1)

    if graphics >= 1:
        plt.figure()
        maxi=500 # maxi=len_x, maxi=500
        plt.plot(Ytrain[0,:maxi], color='r', label='Ytrain')
        plt.xlim(xmax=maxi, xmin=0)
        plt.legend()
        plt.savefig('./Result/Fuzzy/SimulatedR/Ytrain.png', bbox_inches='tight', dpi=150)    
        plt.close()

        plt.figure()
        plt.plot(Xtrain[0,:maxi], color='b', label='Xtrain')
        plt.xlim(xmax=maxi, xmin=0)
        plt.legend()
        plt.savefig('./Result/Fuzzy/SimulatedR/Xtrain.png', bbox_inches='tight', dpi=150)    
        plt.close()

    # Learning
    Ztrain = np.zeros(shape=(n_x+n_y, len_x))
    Ztrain[0  :n_x,     :] = Xtrain
    Ztrain[n_x:n_x+n_y, :] = Ytrain
    aCGOFMSM_learn = CGOFMSM_SemiSupervLearn(STEPS, nbIterSEM, nbRealSEM, Ztrain, n_x, n_y, verbose, graphics)
    aCGOFMSM_learn.run_several()

    # Convert parametrization 3 to parametrization 1 
    filenameParam = './Parameters/Fuzzy/' + pathlib.Path(fileTrainX).stem + '_' + pathlib.Path(fileTrainY).stem + '_F=' + str(STEPS) + '.param'
    Cov, MeanX, MeanY = aCGOFMSM_learn.ConvertParameters()
    aCGOFMSM_learn.SaveParameters(filenameParam, Cov, MeanX, MeanY)


if __name__ == '__main__':
    main()
