#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import numpy as np
import pandas as pd
import datetime as dt
import sys
import pathlib

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as md

from OFAResto.CGOFMSM_Restoration import CGOFMSM

# from pandas.plotting import register_matplotlib_converters
# register_matplotlib_converters()


if __name__ == '__main__':

    """
        Programmes pour simuler et restaurer des siganux réels avec CGOFMSM.
 
        :Example:

        >> python3 CGOFMSM_SignalRest.py Parameters/Signal.param 2:0.07:0.24:0.09 1,1,0 ./Data/Kaggle/input/JoseRizalBridgeNorth_ATemp.csv 3 2 1
        >> python3 CGOFMSM_SignalRest.py Parameters/Signal.param 2:0.07:0.24:0.09 1,1,0 ./Data/Kaggle/input/JoseRizalBridgeNorth_ATemp.csv 1,2,3,5,7,10 10 0 1
        >> nohup python3 CGOFMSM_SignalRest.py Parameters/Signal.param 4:0.15:0.15:0.1 1,1,0 ./Data/Kaggle/input/JoseRizalBridgeNorth_ATemp.csv 1,2,3,5,7,10 10 0 1 > serie2.out &
        >> python3 CGOFMSM_SignalRest.py ./Parameters/Fuzzy/TMU6048TrainX_extract_TMU6048TrainY_extract_F=1.param 2ter:0.4204:0.2328:0.0000 0,1,0,1 ./Data/Traffic/TMU5509/generated/TMU5509_train.csv -1 2 1

        argv[1] : Name of the file of parameters (cov and means)
        argv[2] : Fuzzy joint law model and parameters; e.g. 2ter:0.3:0.3:0.05
                  If -1 then the model is to be read in the parameter file
        argv[3] : Hard filter & smoother (0/1), filter (0/1), smoother (0/1), predictor (horizon size); e.g. 0,1,0,2
                  The horizon size is 0 if we don't need prediction, 2 if we need a 2-horizon prediction
        argv[4] : Observed signal filename
        argv[5] : If interpolation required, number of discrete fuzzy steps, aka 'F'; e.g. 3.  
                  If -1 then F is to be read in the parameter file
        argv[6] : Debug(3), pipelette (2), normal (1), presque muet (0)
        argv[7] : Plot graphique (0/1)
    """

    print('Ligne de commandes : ', sys.argv, flush=True)

    if len(sys.argv) != 8:
        print('CAUTION : bad number of arguments - see help')
        exit(1)

    # Default value for parameters
    filenameParamCov = 'Parameters/Signal.param'
    FSParametersStr  = '2:0.07:0.24:0.09'
    work             = [1,1,0,1]
    filename         = './Data/Kaggle/input/JoseRizalBridgeNorth_all_resample_1303_GT.csv'
    STEPS            = [3,7]
    NbExp            = 1
    verbose          = 2
    Plot             = True

    # Parameters form argv
    filenameParamCov = sys.argv[1]
    FSParametersStr  = sys.argv[2]
    work             = list(map(int, sys.argv[3].split(',')))
    filename         = sys.argv[4]
    STEPS            = list(map(int, sys.argv[5].split(',')))
    verbose          = int(sys.argv[6])
    Plot             = True
    if int(sys.argv[7]) == 0: Plot = False

    interpolation = True
    if STEPS[0] == -1:
        interpolation = False

    if verbose>0:
        print(' . filenameParamCov =', filenameParamCov)
        print(' . FSParametersStr  =', FSParametersStr)
        print(' . work             =', work)
        print(' . filename         =', filename)
        print(' . STEPS            =', STEPS)
        print(' . interpolation    =', interpolation)
        print(' . verbose          =', verbose)
        print(' . Plot             =', Plot)
        print('\n')

    hard, filt, smooth = True, True, True
    if work[0] == 0: hard   = False
    if work[1] == 0: filt   = False
    if work[2] == 0: smooth = False
    predic = int(work[3])
    if predic<0:
        print('predic=', predic, ' --> Not allowed !')
        exit(1)

    if hard==False and filt==False and smooth==False and predic==0:
        print('work=', work, ' --> Not allowed !')
        exit(1)

    # Data reading
    Data = pd.read_csv(filename, parse_dates=[0])
    listeHeader = list(Data)
    pd.to_datetime(Data[listeHeader[0]])
    Data.sort_values(by=[listeHeader[0]])
    datemin = Data[listeHeader[0]].iloc[0]
    datemax = Data[listeHeader[0]].iloc[-1]
    if verbose>1:
        print('Time series of data')
        print('  Entête des colonnes du fichier : ', listeHeader)
        print('  -->Date début série = ', datemin)
        print('  -->Date fin   série = ', datemax)
    
    ######### Extrait Données building
    # excerptData = Data[(Data[listeHeader[0]] >= datemin) & (Data[listeHeader[0]] <= '2010-06-01 16:00:00')]
    # excerptData = Data[(Data[listeHeader[0]] >= datemin) & (Data[listeHeader[0]] <= '2010-06-03 00:00:00')]
    # 2 jours avec un samedi
    # excerptData = Data[(Data[listeHeader[0]] >= '2010-06-04 00:00:00') & (Data[listeHeader[0]] <= '2010-06-06 00:00:00')]
    # excerptData = Data[(Data[listeHeader[0]] >= '2010-06-04 00:00:00') & (Data[listeHeader[0]] <= '2010-06-04 02:00:00')]
    ######### Extrait Données flot voitures
    # excerptData = Data[(Data[listeHeader[0]] >= datemin) & (Data[listeHeader[0]] <= '2018-01-04 02:59:00')]
    # excerptData = Data[(Data[listeHeader[0]] >= '2018-01-05 00:00:00') & (Data[listeHeader[0]] <= '2018-01-07 00:00:00')]
    excerptData = Data

    # filtrage
    N = excerptData[listeHeader[0]].count()
    filestem     = pathlib.Path(filename).stem
    aCGOFMSM     = CGOFMSM(N, filenameParamCov, verbose, FSParametersStr, interpolation)
    elapsed_time = aCGOFMSM.restore_signal(Data=excerptData, filestem=filestem, STEPS=STEPS, hard=hard, filt=filt, smooth=smooth, predic=predic, Plot=Plot)
    
