#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import sys
import pathlib

import matplotlib
import matplotlib.pyplot as plt

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from OFAResto.CGOFMSM_SemiSupervLearn import CGOFMSM_SemiSupervLearn, MeanCovFuzzy

if __name__ == '__main__':
    """
        Programmes pour estimer les paramètres d'un CGOFMSM, lorsque l'on connait un échantillon Z=(X,Y).
 
        :Example:

        >> python3 SemiSupervLearn_CGOFMSM.py ./Data/Traffic/TMUSite5509-2/TMUSite5509-2_train.csv 10 1 4 2 1
        >> python3 SemiSupervLearn_CGOFMSM.py ./../Data_CGPMSM/OpenEI/BuildingTemp/input/building1retail_June_Week_672.csv 5 1 4 2 1

        >> python3 SemiSupervLearn_CGOFMSM.py ./../Data_CGPMSM/OpenEI/BuildingTemp/input/building1retail_June_Week_672.csv 100 1 10 2 2 --> forward converge vers 0.!!
        
        argv[1] : csv filename with timestamp in col 0, observations in col 1 and states in col2
        argv[2] : nb of iterations for SEM-based learning
        argv[3] : nb of realizations for SEM-based learning
        argv[4] : number of discrete fuzzy steps (so-called 'F' or 'STEPS')
        argv[5] : verbose (0/1/2)
        argv[6] : plot the graphics (0:none/1:few/2:many/3:all)
    """

    print('Ligne de commandes : ', sys.argv, flush=True)

    if len(sys.argv) != 7:
        print('CAUTION : bad number of arguments - see help')
        exit(1)

    # Parameters from argv
    fileTrain = sys.argv[1]
    nbIterSEM = int(sys.argv[2])
    nbRealSEM = int(sys.argv[3])
    STEPS     = int(sys.argv[4])
    verbose   = int(sys.argv[5])
    graphics  = int(sys.argv[6])

    # check parameter's value
    if nbIterSEM<0: 
        print('The number of iterations should be greater or equal to 0 --> set to 10')
        nbIterSEM=10
    if nbRealSEM<1: 
        print('The number of realizations by iteration should be greater or equal to 1 --> set to 1')
        nbRealSEM=1
    if STEPS<0: 
        print('The number of fuzzy steps should be greater or equal to 0 --> set to 3')
        STEPS=3
    if verbose<0 or verbose>2: 
        print('verbose should be 0, 1 or 2 --> set to 1')
        verbose=1
    if graphics<0 or graphics>3: 
        print('graphics should be 0, 1, 2 or 3 --> set to 1')
        graphics=1

    if verbose>0:
        print(' . fileTrain =', fileTrain)
        print(' . nbIterSEM =', nbIterSEM)
        print(' . nbRealSEM =', nbRealSEM)
        print(' . STEPS     =', STEPS)
        print(' . verbose   =', verbose)
        print(' . graphics  =', graphics)

    # Lecture des données
    Datatrain   = pd.read_csv(fileTrain, parse_dates=[0])
    listeHeader = list(Datatrain)
    pd.to_datetime(Datatrain[listeHeader[0]])
    Datatrain.sort_values(by=[listeHeader[0]])
    if verbose>1:
        print('Time series of data')
        print('  Entête des colonnes du fichier : ', listeHeader)
        print('  -->Date début série = ', Datatrain[listeHeader[0]].iloc[0])
        print('  -->Date fin   série = ', Datatrain[listeHeader[0]].iloc[-1])

    # Learning
    filestem  = pathlib.Path(fileTrain).stem
    aCGOFMSM_learn = CGOFMSM_SemiSupervLearn(STEPS, nbIterSEM, nbRealSEM, Datatrain, filestem, verbose, graphics)
    aCGOFMSM_learn.run_several()
    exit(1)

    # Convert parametrization 3 to parametrization 1 
    filenameParam = './Parameters/Fuzzy/' + filestem + '_F=' + str(STEPS) + '.param'
    Cov, MeanX, MeanY = aCGOFMSM_learn.ConvertParameters()
    aCGOFMSM_learn.SaveParameters(filenameParam, Cov, MeanX, MeanY)
    filenameParam = './Parameters/Fuzzy/' + filestem + '_interpolation.param'
    aCGOFMSM_learn.SaveParametersInterpolation(filenameParam, Cov, MeanX, MeanY)
