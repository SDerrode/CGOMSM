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

from Test_CGOFMSM import CGOFMSM

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# matplotlib.rc('xtick', labelsize=fontS)
# matplotlib.rc('ytick', labelsize=fontS)


if __name__ == '__main__':

    """
        Programmes pour simuler et restaurer des siganux réels avec CGOFMSM.
 
        :Example:

        >> python3 Test_CGOFMSM_Signals.py Parameters/Signal.param 2:0.07:0.24:0.09 1,1,0 ./Data/Kaggle/input/JoseRizalBridgeNorth_ATemp.csv 3 2 1
        >> python3 Test_CGOFMSM_Signals.py Parameters/Signal.param 2:0.07:0.24:0.09 1,1,0 ./Data/Kaggle/input/JoseRizalBridgeNorth_ATemp.csv 1,2,3,5,7,10 10 0 1
        >> nohup python3 Test_CGOFMSM_Signals.py Parameters/Signal.param 4:0.15:0.15:0.:0.1 1,1,0 ./Data/Kaggle/input/JoseRizalBridgeNorth_ATemp.csv 1,2,3,5,7,10 10 0 1 > serie2.out &

        argv[1] : Nom du fichier de paramètres
        argv[2] : Fuzzy joint law model and parameters. e.g. 2:0.07:0.24:0.09, or 4:0.15:0.15:0.:0.1
        argv[3] : hard(0/1),filter(0/1),smoother(0/1)
        argv[4] : Signal filename
        argv[5] : Valeurs de F (une seule, ou plusieurs séparées par des virgules)
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
    work             = [1,1,0]
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
    if int(sys.argv[7]) == 0: 
        Plot = False

    print(' . filenameParamCov =', filenameParamCov)
    print(' . FSParametersStr  =', FSParametersStr)
    print(' . work             =', work)
    print(' . filename         =', filename)
    print(' . STEPS            =', STEPS)
    print(' . verbose          =', verbose)
    print(' . Plot             =', Plot)
    print('\n')

    hard   = True
    filt   = True
    smooth = True
    if work[0] == 0: hard   = False
    if work[1] == 0: filt   = False
    if work[2] == 0: smooth = False
    if hard==False and filt==False and smooth==False:
        print('work=', work, ' --> Not allowed !')
        exit(1)

    # Lecture des données
    df = pd.read_csv(filename, parse_dates=[0])
    pd.to_datetime(df['Timestamp'])
    df.sort_values(by=['Timestamp'])

    datemin = df['Timestamp'].iloc[0]
    datemax = df['Timestamp'].iloc[-1]
    print('  -->Date départ série temporelle = ', datemin)
    print('  -->Date fin    série temporelle = ', datemax)
    df.set_index('Timestamp', inplace=True)
    listeHeader = list(df)
    print('Entête des columns : ', listeHeader)

    # filtrage
    N = df['Y'].count()
    aCGOFMSM     = CGOFMSM(N, filenameParamCov, verbose, FSParametersStr)
    elapsed_time = aCGOFMSM.restore_signal(Data=df, ch='Temper', STEPS=STEPS, hard=hard, filt=filt, smooth=smooth, Plot=Plot)
