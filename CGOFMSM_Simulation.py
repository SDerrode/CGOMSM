#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import random
import numpy as np

from OFAResto.CGOFMSM_Restoration import CGOFMSM


if __name__ == '__main__':

    """
        Program to sample and restore data using the CGOFMSM.
 
        :Example:

        >> python3 CGOFMSM_Simulation.py Parameters/Fuzzy/SP2018.param 2:0.12:0.12:0.09 1,1,1,1 40000 0 1 2 0
        >> python3 CGOFMSM_Simulation.py Parameters/Fuzzy/SP2018.param 2:0.07:0.24:0.09 1,1,1,1 72    3 5 2 1
        >> python3 CGOFMSM_Simulation.py Parameters/Fuzzy/SP2018.param 2:0.07:0.24:0.09 1,1,0,1 500 1,2,3,5,7,10 10 1 0 1
        >> nohup python3 CGOFMSM_Simulation.py Parameters/Fuzzy/SP2018.param 4:0.15:0.15:0.05 1,1,0,1 1000 1,2,3,5,7,10 10 1 0 1 > serie2.out &

        argv[1] : Parameters file name
        argv[2] : Fuzzy joint law model and parameters; e.g. 2ter:0.3:0.3:0.05
                  If -1 then the model is to be read in the parameter file
        argv[3] : Prefix for file saving
        argv[4] : Sample size
        argv[5] : F value (one or several separated by commas), e.g. 1,3,5. 
                  If -1 then the value is to be read in the parameter file.
        argv[6] : Verbose level: Debug (3), pipelette (2), normal (1), presque muet (0)
        argv[7] : Graphics? (0/1)
    """

    print('Ligne de commandes : ', sys.argv, flush=True)

    if len(sys.argv) != 8:
        print('CAUTION : bad number of arguments - see help')
        exit(1)

    # Default value for parameters
    filenameParamCov = 'Parameters/Fuzzy/SP2018.param'
    FSParametersStr  = '2:0.07:0.24:0.09'
    prefixName       = 'Simulation'
    N                = 3000
    STEPS            = [3, 7]
    verbose          = 2
    Plot             = True

    # Parameters from argv
    filenameParamCov = sys.argv[1]
    FSParametersStr  = sys.argv[2]
    prefixName       = sys.argv[3]
    N                = int(sys.argv[4])
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
    print(' . N                =', N)
    print(' . STEPS            =', STEPS)
    print(' . NbExp            =', NbExp)
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

    if predic>0: filt=True
    if hard==False and filt==False and smooth==False and predic==0:
        print('work=', work, ' --> Not allowed !')
        exit(1)

    np.random.seed(None)
    #np.random.seed(1), print('-----------------> ATTENTION : random fixé!!!<------------------')

    readData = False  # Data are read or resimulated ?

    # import time
    # start_time = time.time()

    # Moyenne de plusieurs expériences
    aCGOFMSM = CGOFMSM(N, filenameParamCov, verbose, FSParametersStr, interpolation)
    mean_tab_MSE, mean_tab_MSE_HARD, mean_time = aCGOFMSM.run_several(NbExp, STEPS=STEPS, hard=hard, filt=filt, smooth=smooth, predic=predic, Plot=Plot)

    # print("--- %s seconds ---" % (time.time() - start_time))
