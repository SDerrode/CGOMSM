#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 13:53:15 2018
Data simumated from general CGPMSM model (Fyx != 0)
Two restorations are applied:
    1. exact Kalman restoration (PKF known R)
    2. restoration of nearest CGOMSM model (unknown R)
@author: fzheng
"""

import numpy as np
import matplotlib.pyplot as plt
import os, sys
sys.path.insert(0, os.path.abspath("."))

import time
plt.close('all') 

from CommonFun.CommonFun import Readin_data, MSE_PK, Error_Ratio, SaveSimulatedData
from CommonFun.CommonFun import Readin_ABQCovMeans
from PKFResto.PKFResto import RestorationPKF
from OFAResto.OFAResto import RestorationOFA
from CGPMSMs import GetParamNearestCGO, simuCGPMSM


if __name__ == '__main__':
    np.random.seed(10)      #!!! random for debug
    start_time = time.time()
    
    N = 2000    # Sample length
    n_r = 2     # Switch component number
    # n_r = 3     # For multi-D 

    # =========== Data simulation ============ #    
    filenameXRY = 'Result/Fei/SimulatedData/X_R_Y.txt'
    # ----------- Test 1D ------------ #
    filenameParam = 'Parameters/Fei/PARAMETER_CGPMSM_General_series1_0.000_Mean0.0_jump0.90.txt'
    #filenameParam = 'Parameters/Fei/PARAMETER_CGPMSM_General_series1_0.400_Mean0.0_jump0.90.txt'
    #filenameParam ='Parameters/Fuzzy/SP2018.param'
    # ----------- Test MultiD ------------- #
    # filenameParam ='Parameters/PARAMETER_MultiD_4.txt'

    F, B, Q, Cov, Mean_X, Mean_Y, JProba, MProba, TProba = Readin_ABQCovMeans(filenameParam, n_r)
    X, R, Y = simuCGPMSM(F, B, Cov, Mean_X, Mean_Y, MProba, TProba, N) # n_r should be given or R should be given
    SaveSimulatedData(X, R, Y, filenameXRY)
    # X,R,Y = Readin_data(filenameXRY)
    
    #F, B, Q,      Mean_X, Mean_Y, transition_probabilities             = RestorationOFA().ParamFromFile(filenameParam, n_r)
    
    # =========== Supervised Restoration ============ #    
    # -------- Kalman filter for CGPMSM knowing switches --------- #
    E_X_PF, E_X_PS, P_n_N, C_n_np1_N = RestorationPKF().restore_withjump(Y, R, F, Q, Cov, Mean_X, Mean_Y, Likelihood=False)   # For CGOMSM filtering result = smoothing result
    

    # ------ Optimal filter approximation unknown switches ------- #
    # find the nearest CGOMSM paremeters 
    F_CGO, Q_CGO, useless           = GetParamNearestCGO(F, Q, n_x=np.shape(X)[1])
    E_X_OFA,E_X_OSA,E_R_OFA,E_R_OSA = RestorationOFA().restore_1D(Y, F_CGO, Q_CGO, Cov, Mean_X, Mean_Y, JProba, TProba, MProba)   # ATTENTION: parafile F_yx=0 for restoration always!!!
    # E_X_OFA, E_X_OSA, E_R_OFA, E_R_OSA = RestorationOFA().restore_MultiD(Y, F_CGO, Q_CGO, Mean_X, Mean_Y, TProba, MProba)   # ATTENTION: parafile F_yx=0 for restoration always!!!


    print("================ Supervised Restoration ===============")
    # print("  mse_Y :          %.3f" %(MSE_PK(Y,X)) )
    print("  mse_X_PKF :      %.3f" %(MSE_PK(E_X_PF,X)) )
    print("  mse_X_PKS :      %.3f" %(MSE_PK(E_X_PS,X)) )
    print("  mse_X_OFA :      %.3f" %(MSE_PK(E_X_OFA,X)) )
    print("  mse_X_OSA :      %.3f" %(MSE_PK(E_X_OSA,X)) )
    print("\n  ErrorRatio_OFA : %.3f" %(Error_Ratio(E_R_OFA,R)))
    print("  ErrorRatio_OSA : %.3f" %(Error_Ratio(E_R_OSA,R)))
    print("\n--- %s seconds ---" % (time.time() - start_time) )

    # plt.figure()
    # plt.plot(X[0,0:100],'b',label='Hidden states')
    # plt.plot(Y[0,0:100],'k',label='Observations')
    # plt.plot(E_X_PS[0,0:100],'g',label='KS')
    # plt.plot(E_X_OFA[0,0:100],'r',label='OSA')
    # plt.legend();plt.title('D-1')
    # plt.show()
