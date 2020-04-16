#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import numpy as np
import pandas as pd
import datetime as dt
import sys

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as md

from OFAResto.CGOFMSM_Restoration import CGOFMSM

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


if __name__ == '__main__':

    """
        Programmes pour simuler et restaurer des siganux réels avec CGOFMSM.
 
        :Example:

        >> python3 Test_CGOFMSM_Signals.py Parameters/Signal.param 2:0.07:0.24:0.09 1,1,0 ./Data/Kaggle/input/JoseRizalBridgeNorth_ATemp.csv 3 2 1
        >> python3 Test_CGOFMSM_Signals.py Parameters/Signal.param 2:0.07:0.24:0.09 1,1,0 ./Data/Kaggle/input/JoseRizalBridgeNorth_ATemp.csv 1,2,3,5,7,10 10 0 1
        >> nohup python3 Test_CGOFMSM_Signals.py Parameters/Signal.param 4:0.15:0.15:0.:0.1 1,1,0 ./Data/Kaggle/input/JoseRizalBridgeNorth_ATemp.csv 1,2,3,5,7,10 10 0 1 > serie2.out &
        >> python3 Test_CGOFMSM_Signals.py ./Parameters/Fuzzy/TMU6048TrainX_extract_TMU6048TrainY_extract_F=1.param 2ter:0.4204:0.2328:0.0000 0,1,0,1 ./Data/Traffic/TMU5509/generated/TMU5509_train.csv -1 2 1

        argv[1] : Name of the file of parameters (cov and means)
        argv[2] : Fuzzy joint law model and parameters; e.g. 2ter:0.3:0.3:0.05
        argv[3] : Hard filter & smoother(0/1), filter (0/1), smoother (0/1), predictor (0/1); e.g. 0,1,0,1
        argv[4] : Observed signal filename
        argv[5] : If interpolation, number of discrete fuzzy, aka 'F'; e.g. 3.  If -1 then F is to be read in the parameter file
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

    print(' . filenameParamCov =', filenameParamCov)
    print(' . FSParametersStr  =', FSParametersStr)
    print(' . work             =', work)
    print(' . filename         =', filename)
    print(' . STEPS            =', STEPS)
    print(' . verbose          =', verbose)
    print(' . Plot             =', Plot)
    print('\n')

    hard, filt, smooth, predic = True, True, True, True
    if work[0] == 0: hard   = False
    if work[1] == 0: filt   = False
    if work[2] == 0: smooth = False
    if work[3] == 0: predic = False
    if predic==True: filt=True
    if hard==False and filt==False and smooth==False and predic==False:
        print('work=', work, ' --> Not allowed !')
        exit(1)

    # Lecture des données
    df = pd.read_csv(filename, parse_dates=[0])
    listeHeader = list(df)
    pd.to_datetime(df[listeHeader[0]])
    df.sort_values(by=[listeHeader[0]])
    datemin = df[listeHeader[0]].iloc[0]
    datemax = df[listeHeader[0]].iloc[-1]
    print('  -->Date départ série temporelle = ', datemin)
    print('  -->Date fin    série temporelle = ', datemax)
    # df.set_index('Timestamp', inplace=True)
    # listeHeader = list(df)
    print('Entête des columns : ', listeHeader)

    excerpt = df[(df[listeHeader[0]] >= datemin) & (df[listeHeader[0]] <= '2018-01-04 02:59:00')]


    # filtrage
    N = excerpt['Y'].count()
    aCGOFMSM     = CGOFMSM(N, filenameParamCov, verbose, FSParametersStr, interpolation)
    elapsed_time = aCGOFMSM.restore_signal(Data=excerpt, ch='flow', STEPS=STEPS, hard=hard, filt=filt, smooth=smooth, predic=predic, Plot=Plot)
    
