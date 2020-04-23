#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import numpy as np
import copy
import random

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates  as md
import pandas            as pd
import datetime          as dt

years    = md.YearLocator()   # every year
months   = md.MonthLocator()  # every month
days     = md.DayLocator()    # every day
yearsFmt = md.DateFormatter('%Y      ')
monthFmt = md.DateFormatter('%B %Y')
dayFmt   = md.DateFormatter('%d')

from Fuzzy.InterFuzzy       import simulateFuzzy
from CommonFun.CommonFun    import MSE_PK, Error_Ratio, From_Cov_to_FQ
from CommonFun.CommonFun    import SaveSimulatedFuzzyData, ReadSimulatedFuzzyData
from CommonFun.CommonFun    import Test_isCGOMSM_from_Cov, Test_isCGOMSM_from_F
from OFAResto.OFAFuzzyResto import RestorationOFAFuzzy
from OFAResto.OFAResto      import RestorationOFA
from PKFResto.PKFResto      import RestorationPKF
from CGPMSMs.CGPMSMs        import GetParamNearestCGO_cov

fontS = 13     # font size
DPI   = 150    # graphic resolution



class CGOFMSM:

    def __init__(self, N, filenameParamCov, verbose, FSParametersStr, interpolation=True):
        assert N > 1, print('number of samples must be greater than 2')

        self.__n_r              = 2 # default value, can be changed by reading of parameters (see below)
        self.__N                = N
        self.__filenameParamCov = filenameParamCov
        self.__verbose          = verbose
        self.__interpolation    = interpolation
        self.__FSParameters     = list(map(str, FSParametersStr.split(':')))

        # plage graphique pour les plots
        self.__graph_mini = 290
        self.__graph_maxi = min(290+190, self.__N) # maxi=self.__N, maxi=min(500, self.__N)

    def getParams(self):
        return self.__n_r, self.__N, self.__filenameParamCov, self.__verbose, self.__FSParameters

    def __str__(self):
        S  = 'n_r='                + str(self.__n_r)
        S += ', N='                + str(self.__N)
        S += ', filenameParamCov=' + self.__filenameParamCov
        S += ', verbose='          + self.__verbose
        S += ', FSParameters='     + self.__FSParameters
        return S

    def SimulateFuzzy(self, filenameSimulatedXRY=None, readData = False, Plot = False):

        if readData == False:
            self.__n_r, X, R, Y, self.__FSParameters = simulateFuzzy(self.__filenameParamCov, self.__FSParameters, self.__interpolation, self.__N)
            assert self.__n_r == 2, print('number of jumps must be 2!')
            if Plot is True:
                self.PlotSimul(X, R, Y)
        else:
            X, R, Y, self.__FSParameters = ReadSimulatedFuzzyData(filenameSimulatedXRY)
            self.__N = np.shape(X)[0]

        # print('self.__FSParameters=', self.__FSParameters)
        # input('pause SimulateFuzzy')

        cpt = 0
        for i in range(self.__N):
            if R[i] == 0.0 or R[i] == 1.0:
                cpt += 1
        pHNum = cpt / self.__N
        #print('pHNum=', pHNum)
        #input('Fin simulation')

        # Save simulated data if required
        if filenameSimulatedXRY is not None and readData == False:
            SaveSimulatedFuzzyData(X, R, Y, filenameSimulatedXRY)

        return X, R, Y, pHNum


    def restore_signal(self, Data, filestem, STEPS=[7], hard=True, filt=True, smooth=True,  predic=1, Plot=False):

        elapsed_time = 0
        start_time = time.time()

        # Create main objects
        Resto = RestorationOFAFuzzy(self.__filenameParamCov, STEPS[0], self.__FSParameters, self.__interpolation, self.__verbose)
        self.__n_r, n_x, n_y, n_z, STEPS, F, B, Q, Cov, Mean_X, Mean_Y, self.__FSParameters = Resto.getParams()
        # print(self.__n_r, n_x, n_y, n_z, STEPS, self.__interpolation)
        assert self.__n_r == 2, print('number of jumps must be 2!')
        
        Ok = Test_isCGOMSM_from_Cov(Cov, n_x)
        if Ok == False:
            print('ATTENTION : le modele nest pas un CGO! --> TRANSFORMATION EN CGO')
            Cov = GetParamNearestCGO_cov(Cov, n_x=n_x)
            F, Q = From_Cov_to_FQ(Cov)

        Data.set_index(list(Data)[0], inplace=True)
        listeHeader = list(Data)
        listeHeader=list(Data)
        Y          = np.zeros((self.__N, n_y))
        Y[:, 0]    = Data[listeHeader[0]].values
        X_GT       = np.zeros((self.__N, n_x))
        X_GT[:, 0] = Data[listeHeader[1]].values
        
        # Detect the weekend days switches
        weekend_indices_window = []
        BoolWE = False
        for i in range(self.__graph_mini,self.__graph_maxi):
            if Data.index[i].weekday() >= 5 and BoolWE == False:
                weekend_indices_window.append(i) 
                BoolWE = True
            if Data.index[i].weekday() < 5 and BoolWE == True:
                weekend_indices_window.append(i) 
                BoolWE = False
        # refermer si ouvert
        if BoolWE==True:
            weekend_indices_window.append(i)
        # print('weekend_indices_window=', weekend_indices_window)
        # input('weekday')

        weekend_indices = []
        BoolWE = False
        for i in range(0, self.__N):
            if Data.index[i].weekday() >= 5 and BoolWE == False:
                weekend_indices.append(i) 
                BoolWE = True
            if Data.index[i].weekday() < 5 and BoolWE == True:
                weekend_indices.append(i) 
                BoolWE = False
        # refermer si ouvert
        if BoolWE==True:
            weekend_indices.append(i)
        # print('weekend_indices=', weekend_indices)
        # input('weekday')

        # Save of MSE results
        MSE      = np.zeros((len(STEPS), 10))
        MSE_HARD = np.zeros((len(STEPS), 10))

        if hard == True:

            # Approximation des proba hard (par intégration des 4 quadrants)
            MProba, TProba, JProba = Resto.getFS().getTheoriticalHardTransition(self.__n_r)
            # print('Array MProba = ', MProba)
            # print('Array TProba = ', TProba)
            # print('Array JProba = ', JProba)
            # input('pause')

            # Filtrage Hard sans les jumps
            E_X_OFA_HARD, E2_X_OFA_HARD, Cov_X_OFA_HARD, \
            E_X_OSA_HARD, E2_X_OSA_HARD, Cov_X_OSA_HARD, \
            E_X_OPA_HARD, E2_X_OPA_HARD, Cov_X_OPA_HARD, \
            useless0, useless1, useless2, useless3, \
            E_R_OFA_HARD, E_R_OSA_HARD, E_R_OPA_HARD \
                = RestorationOFA().restore_withoutjumps(Y, F, Q, Resto.getCov(), Resto.getMean_X(), Resto.getMean_Y(), TProba, MProba)
            # print('E_R_OFA_HARD=', E_R_OFA_HARD[0:5])
            # print('E_X_OFA_HARD=', E_X_OFA_HARD[0:5])
            elapsed_time += time.time() - start_time

            MSE_HARD[0, 4] = MSE_PK(E_X_OFA_HARD, X)
            MSE_HARD[0, 5] = MSE_PK(E_X_OSA_HARD, X)
            MSE_HARD[0, 6] = MSE_PK(E_X_OPA_HARD, X)

            if Plot is True:
                chaine = Resto.getFSText() + '_' + filestem + '_FILT_HARD'
                self.PlotTrajectoriesSignal(weekend_indices, filestem, STEPS[0], chaine, 'Hard filter (CGOMSM)', Data, E_X_OFA_HARD, E_R_OFA_HARD)
                chaine = Resto.getFSText() + '_' + filestem + '_SMOO_HARD'
                self.PlotTrajectoriesSignal(weekend_indices, filestem, STEPS[0], chaine, 'Hard smoother (CGOMSM)', Data, E_X_OSA_HARD, E_R_OSA_HARD)

        if filt==True or smooth==True or predic>0:

            # Loop in discrete jumps F
            for i, steps in enumerate(STEPS):
                if self.__verbose >= 1:
                    print('    #####STEPS ', steps)

                Resto.resetSTEPS(STEPS[i])

                start_time = time.time()

                # FUZZY: filter (and smooth) with unknown jumps
                E_X_OFA, E_R_OFA, E_X_OSA, E_R_OSA, E_Z_OPA, E_R_OPA = Resto.restore_Fuzzy1D(Y, filt=filt, smooth=smooth, predic=predic)
                elapsed_time += time.time() - start_time

                # MSE
                MSE[i, 4] = MSE_PK(E_X_OFA, X_GT)

                if smooth:
                    MSE[i, 5] = MSE_PK(E_X_OSA, X_GT)

                if predic>0:
                    MSE[i, 6] = MSE_PK(E_Z_OPA[1:, 0], X_GT[1:, 0])

                if Plot is True:
                    chaine = Resto.getFSText() + '_' + filestem + '_ALLXinONE_window_FUZZY_STEP_' + str(steps)
                    self.PlotTrajectoriesAllSignal(weekend_indices_window, self.__graph_mini, self.__graph_maxi, filestem, STEPS[0], chaine, 'Fuzzy restoration (CGOMSM)', Data, E_X_OFA, E_X_OSA, E_Z_OPA[:, 0:n_x])
                    chaine = Resto.getFSText() + '_' + filestem + '_ALLXinONE_FUZZY_STEP_' + str(steps)
                    self.PlotTrajectoriesAllSignal(weekend_indices, 0, self.__N, filestem, STEPS[0], chaine, 'Fuzzy restoration (CGOMSM)', Data, E_X_OFA, E_X_OSA, E_Z_OPA[:, 0:n_x])
                    if filt:
                        chaine = Resto.getFSText() + '_' + filestem + '_FILT_FUZZY_STEP_' + str(steps)
                        self.PlotTrajectoriesSignal(weekend_indices, filestem, STEPS[0], chaine, 'Fuzzy filter (CGOMSM)', Data, E_X_OFA, E_R_OFA)
                    if smooth:
                        chaine = Resto.getFSText() + '_' + filestem + '_SMOO_FUZZY_STEP_' + str(steps)
                        self.PlotTrajectoriesSignal(weekend_indices, filestem, STEPS[0], chaine, 'Fuzzy smoother (CGOMSM)', Data, E_X_OSA, E_R_OSA)
                    if predic>0:
                        chaine = Resto.getFSText() + '_' + filestem + '_PRED_FUZZY_STEP_' + str(steps)
                        self.PlotTrajectoriesSignal(weekend_indices, filestem, STEPS[0], chaine, 'Fuzzy predictor (CGOMSM)', Data, E_Z_OPA[:, 0:n_x], None)

        if self.__verbose >= 0:
            self.printResultSignal(MSE, MSE_HARD, STEPS, elapsed_time, hard, filt, smooth, predic, ch= '(SIGNAL)')
    
        return elapsed_time


    def run_several(self, nb_exp, STEPS=[5], hard=True, filt=True, smooth=False, predic=1, Plot=False):

        tab_MSE          = np.zeros((nb_exp, len(STEPS), 10))
        tab_MSE_HARD     = np.zeros((nb_exp, len(STEPS), 10))
        tab_elapsed_time = np.zeros((nb_exp))

        for e in range(nb_exp):
            if self.__verbose >= 1:
                print('\n##########Experiment ', e)
            # result of one experiment
            tab_MSE[e,:], tab_MSE_HARD[e,:], tab_elapsed_time[e], FStext, pHNum = self.run_one("EXP"+ str(e+1), STEPS=STEPS, hard=hard, filt=filt, smooth=smooth, predic=predic, readData=False, Plot=Plot)

        #print('tab_MSE=', tab_MSE)
        #print('tab_MSE_HARD=', tab_MSE_HARD)

        # mean computations for screening
        mean_tab_MSE      = np.mean(tab_MSE,      axis=0)
        mean_tab_MSE_HARD = np.mean(tab_MSE_HARD, axis=0)
        mean_time         = np.mean(tab_elapsed_time)

        if self.__verbose >= 0 and nb_exp > 1:
            self.printResult(mean_tab_MSE, mean_tab_MSE_HARD, STEPS, mean_time, hard=hard, filt=filt, smooth=smooth, predic=predic, ch= '(mean)')
        if len(STEPS)>1:
            self.plotMSE(STEPS, mean_tab_MSE, mean_tab_MSE_HARD, FStext, hard=hard, filt=filt, smooth=smooth, predic=predic)

        return mean_tab_MSE, mean_tab_MSE_HARD, mean_time


    def run_one(self, ch, STEPS=[7], hard=True, filt=True, smooth=True, predic=1, readData = False, Plot=False):

        # Save of MSE results
        MSE      = np.zeros((len(STEPS), 10))
        MSE_HARD = np.zeros((len(STEPS), 10))

        header = 'Nombre de donnees : N = ' + str(self.__N)

        start_time   = time.time()
        elapsed_time = 0.

        # Create main objects
        Resto = RestorationOFAFuzzy(self.__filenameParamCov, STEPS[0], self.__FSParameters, self.__interpolation, self.__verbose)

        # Simulaton of a sample
        fname = 'Result/Fuzzy/SimulatedData/' + Resto.getFSText() + '_fuzzy_' + ch + '.txt'
        X, Rfuzzy, Y, pHNum = self.SimulateFuzzy(fname, readData, False)

        if self.__verbose > 0:
            print(Resto.getFS(), ', phNum = ', str(pHNum))

        MSE[0, 0] = MSE_PK(Y, X)
        elapsed_time += time.time() - start_time

        ## Test si modele est CGO ou non
        F, Q = From_Cov_to_FQ(Resto.getCov())
        n_x = np.shape(Resto.getMean_X())[1]
        if Test_isCGOMSM_from_F(F, n_x, verbose=self.__verbose>1) == False:
            print('F=', F)
            input('ATTENTION (Test_CGOFMSM.py / run_one): ce nest pas un CGOMSM!!! --> IMPOSSIBLE')

        # Hard : Restoration with known and unknown hard jumps
        #####################################################
        E_X_OFA_HARD = None
        E_R_OFA_HARD = None
        E_X_OSA_HARD = None
        E_R_OSA_HARD = None
        if hard:
            start_time = time.time()

            # Hardification des sauts flous
            Rhard = np.around(Rfuzzy).astype(int)

            # Approximation des proba hard (par intégration des 4 quadrants)
            MProba, TProba, JProba = Resto.getFS().getTheoriticalHardTransition(self.__n_r)

            # Filtrage Hard avec les jumps
            E_X_PF_HARD, E2_X_PF_HARD,  E_X_PS_HARD, E2_X_PS_HARD,  E_X_PP_HARD, E2_X_PP_HARD\
                = RestorationPKF().restore_withjump(Y, Rhard, F, Q, Resto.getCov(), Resto.getMean_X(), Resto.getMean_Y(), Likelihood=False)

            MSE_HARD[0, 1] = MSE_PK(E_X_PF_HARD, X)
            MSE_HARD[0, 2] = MSE_PK(E_X_PS_HARD, X)
            MSE_HARD[0, 3] = MSE_PK(E_X_PP_HARD[0:self.__N-1, 0:n_x], X[1:])

            # Filtrage Hard sans les jumps
            E_X_OFA_HARD, E2_X_OFA_HARD, Cov_X_OFA_HARD, \
            E_X_OSA_HARD, E2_X_OSA_HARD, Cov_X_OSA_HARD, \
            E_X_OPA_HARD, E2_X_OPA_HARD, Cov_X_OPA_HARD, \
            useless0, useless1, useless2, useless3, \
            E_R_OFA_HARD, E_R_OSA_HARD, E_R_OPA_HARD \
                = RestorationOFA().restore_withoutjumps(Y, F, Q, Resto.getCov(), Resto.getMean_X(), Resto.getMean_Y(), TProba, MProba)
            # print('E_R_OFA_HARD=', E_R_OFA_HARD[0:5])
            # print('E_X_OFA_HARD=', E_X_OFA_HARD[0:5])
            elapsed_time += time.time() - start_time

            MSE_HARD[0, 4] = MSE_PK(E_X_OFA_HARD, X)
            MSE_HARD[0, 5] = MSE_PK(E_X_OSA_HARD, X)
            MSE_HARD[0, 6] = MSE_PK(E_X_OPA_HARD, X)
            MSE_HARD[0, 7] = MSE_PK(E_R_OFA_HARD, Rfuzzy)
            MSE_HARD[0, 8] = MSE_PK(E_R_OSA_HARD, Rfuzzy)
            MSE_HARD[0, 9] = MSE_PK(E_R_OPA_HARD, Rfuzzy)

            if Plot is True:
                chaine = Resto.getFSText() + '_' + ch + '_FILT_HARD'
                self.PlotTrajectories(chaine, 'Hard filter (CGOMSM)',    X, Rhard, Y, E_X_PF_HARD, E_X_OFA_HARD, E_R_OFA_HARD, bottom=0.)
                chaine = Resto.getFSText() + '_' + ch + '_SMOO_HARD'
                self.PlotTrajectories(chaine, 'Hard smoother (CGOMSM)',  X, Rhard, Y, E_X_PS_HARD, E_X_OSA_HARD, E_R_OSA_HARD, bottom=0.)
                chaine = Resto.getFSText() + '_' + ch + '_PRED_HARD'
                self.PlotTrajectories(chaine, 'Hard predictor (CGOMSM)', X, Rhard, Y, E_X_PP_HARD, E_X_OPA_HARD, E_R_OPA_HARD, bottom=0.)


        # Fuzzy : Restoration with known and unknown fuzzy jumps
        #####################################################
        if filt==True or predic>0 or smooth==True:
            
            start_time = time.time()

            # Fuzzy : Restoration with known jumps
            E_X_PF, Cov_X_PF, E_X_PS, Cov_X_PS, E_Z_PP, Cov_Z_PP \
                = RestorationPKF().restore_withfuzzyjump(Y, Rfuzzy, Resto.getCov(), Resto.getMean_X(), Resto.getMean_Y(), Likelihood=False, smooth=smooth)
            MSE[0, 1] = MSE_PK(E_X_PF, X)
            MSE[0, 2] = MSE_PK(E_X_PS, X)
            MSE[0, 3] = MSE_PK(E_Z_PP[0:n_x, :], X)
            elapsed_time += time.time() - start_time

            # Loop in discrete jumps F
            E_X_OFA = None
            E_R_OFA = None
            for i, steps in enumerate(STEPS):
                
                start_time = time.time()

                if self.__verbose >= 1:
                    print('    #####STEPS ', steps)

                Resto.resetSTEPS(STEPS[i])

                # FUZZY: filter (and smooth) with unknown jumps
                E_X_OFA, E_R_OFA, E_X_OSA, E_R_OSA, E_Z_OPA, E_R_OPA = Resto.restore_Fuzzy1D(Y, filt=filt, smooth=smooth, predic=predic)
                elapsed_time += time.time() - start_time

                # MSE
                MSE[i, 4] = MSE_PK(E_X_OFA, X)
                MSE[i, 7] = MSE_PK(E_R_OFA, Rfuzzy)

                if smooth:
                    MSE[i, 5] = MSE_PK(E_X_OSA, X)
                    MSE[i, 8] = MSE_PK(E_R_OSA, Rfuzzy)

                if predic>0:
                    MSE[i, 6] = MSE_PK(E_Z_OPA[1:, 0], X[1:, 0])
                    MSE[i, 9] = MSE_PK(E_R_OPA[1:], Rfuzzy[1:])

                if Plot is True:
                    if filt:
                        chaine = Resto.getFSText() + '_' + ch + '_STEPS' + str(steps) + '_FILT'
                        self.PlotTrajectories(chaine, 'Fuzzy filter (CGOFMSM)', X, Rfuzzy, Y, E_X_PF, E_X_OFA, E_R_OFA, bottom=0.)
                        if hard:
                            chaine = Resto.getFSText() + '_' + ch + '_STEPS' + str(steps) + '_FILT_UJ_X'
                            self.PlotFuzzyHard(chaine, 'CGOFMSM vs CGOMSM filters (UJ)', X, E_X_OFA_HARD, E_X_OFA)
                            chaine = Resto.getFSText() + '_' + ch + '_STEPS' + str(steps) + '_FILT_KJ_X'
                            self.PlotFuzzyHard(chaine, 'CGOFMSM vs CGOMSM filters (KJ)', X, E_X_PF_HARD, E_X_PF)
                            chaine = Resto.getFSText() + '_' + ch + '_STEPS' + str(steps) + '_FILT_UJ_R'
                            self.PlotFuzzyHard_R(chaine, 'CGOFMSM vs CGOMSM filters (UJ)', Rfuzzy, E_R_OFA_HARD, E_R_OFA, bottom=0)

                    if smooth:
                        chaine = Resto.getFSText() + '_' + ch + '_STEPS' + str(steps) + '_SMOO'
                        self.PlotTrajectories(chaine, 'Fuzzy smoother (CGOFMSM)', X, Rfuzzy, Y, E_X_PS, E_X_OSA, E_R_OSA, bottom=0.)
                        if hard:
                            chaine = Resto.getFSText() + '_' + ch + '_STEPS' + str(steps) + '_SMOO_UJ_X'
                            self.PlotFuzzyHard(chaine, 'CGOFMSM vs CGOMSM smoothers (UJ)', X, E_X_OSA_HARD, E_X_OSA)
                            chaine = Resto.getFSText() + '_' + ch + '_STEPS' + str(steps) + '_SMOO_KJ_X'
                            self.PlotFuzzyHard(chaine, 'CGOFMSM vs CGOMSM smoothers (KJ)', X, E_X_PS_HARD, E_X_PS)
                            chaine = Resto.getFSText() + '_' + ch + '_STEPS' + str(steps) + '_SMOO_UJ_R'
                            self.PlotFuzzyHard_R(chaine, 'CGOFMSM vs CGOMSM smoothers (UJ)', Rfuzzy, E_R_OSA_HARD, E_R_OSA, bottom=0)

                    if predic>0:
                        chaine = Resto.getFSText() + '_' + ch + '_STEPS' + str(steps) + '_PRED'
                        self.PlotTrajectories(chaine, 'Fuzzy predictor (CGOFMSM)', X, Rfuzzy, Y, E_Z_PP[:, 0:n_x], E_Z_OPA[:, 0:n_x], E_R_OPA, bottom=0.)
                        if hard:
                            # chaine = Resto.getFSText() + '_' + ch + '_STEPS' + str(steps) + '_PRED_UJ_X'
                            # self.PlotFuzzyHard(chaine, 'CGOFMSM vs CGOMSM predictors (UJ)', X, E_X_OPA_HARD[:, 0:n_x], E_Z_OPA[:, 0:n_x])
                            chaine = Resto.getFSText() + '_' + ch + '_STEPS' + str(steps) + '_PRED_KJ_X'
                            self.PlotFuzzyHard(chaine, 'CGOFMSM vs CGOMSM predictors (KJ)', X, E_X_PP_HARD[:, 0:n_x], E_Z_PP[:, 0:n_x])
                            chaine = Resto.getFSText() + '_' + ch + '_STEPS' + str(steps) + '_PRED_UJ_R'
                            self.PlotFuzzyHard_R(chaine, 'CGOFMSM vs CGOMSM predictors (UJ)', Rfuzzy, E_R_OPA_HARD, E_R_OPA, bottom=0)
    

        if self.__verbose >= 1:
            self.printResult(MSE, MSE_HARD, STEPS, elapsed_time, hard=hard, filt=filt, smooth=smooth, predic=predic)

        # SAVE THE RESULT IN CSV FILE
        if filt or smooth or predic>0:
            fname='Result/Fuzzy/Tab_MSE/' + Resto.getFSText() + '_fuzzy_' + ch + '.txt'
            np.savetxt(fname, MSE, delimiter='\n', newline='\n', header=header, encoding=None)
        if hard:
            fname='Result/Fuzzy/Tab_MSE/' + Resto.getFSText() + '_hard_' + ch + '.txt'
            np.savetxt(fname, MSE_HARD, delimiter='\n', newline='\n', header=header, encoding=None)

        return MSE, MSE_HARD, elapsed_time, Resto.getFSText(), pHNum


    def printResult(self, MSE, MSE_HARD, STEPS, elapsed_time, hard, filt, smooth, predic, ch=''):
        MSE      = np.around(MSE,      decimals=3)
        MSE_HARD = np.around(MSE_HARD, decimals=3)

        print("\n================ Restoration results "+ ch + " ===============")
        # print("  mse_Y\t:", MSE[0, 0])
        print("  STEPS (F)        : ", STEPS, sep='')
        print("")
        if predic>0:
            print("  PREDICTOR---------------------------------")
            print("    Fuzzy X  (KJ)  : TO BE DONE!") #", MSE[0, 3], "TO BE DONE!", sep='')
            print("    Fuzzy X  (UJ)  : ", MSE[:, 6], sep='')
            print("    Fuzzy R  (UJ)  : ", MSE[:, 9], sep='')
        if hard and predic>0:
            print("      Hard X (KJ)  : ", MSE_HARD[0, 3], sep='')
            print("      Hard X (UJ)  : ", MSE_HARD[0, 6], sep='')
            print("      Hard R (UJ)  : ", MSE_HARD[0, 9], sep='')
            print("")
        if filt:
            print("  FILTER------------------------------------")
            print("    Fuzzy X  (KJ)  : ", MSE[0, 1], sep='')
            print("    Fuzzy X  (UJ)  : ", MSE[:, 4], sep='')
            print("    Fuzzy R  (UJ)  : ", MSE[:, 7], sep='')
        if hard and filt:
            print("      Hard X (KJ)  : ", MSE_HARD[0, 1], sep='')
            print("      Hard X (UJ)  : ", MSE_HARD[0, 4], sep='')
            print("      Hard R (UJ)  : ", MSE_HARD[0, 7], sep='')
            print("")
        if smooth:
            print("  SMOOTHER----------------------------------")
            print("    Fuzzy X  (KJ)  : ", MSE[0, 2], sep='')
            print("    Fuzzy X  (UJ)  : ", MSE[:, 5], sep='')
            print("    Fuzzy R  (UJ)  : ", MSE[:, 8], sep='')
        if hard and smooth:
            print("      Hard X  (KJ) : ", MSE_HARD[0, 2], sep='')
            print("      Hard X  (UJ) : ", MSE_HARD[0, 5], sep='')
            print("      Hard R  (UJ) : ", MSE_HARD[0, 8], sep='')
        if hard and filt==False and smooth==False and predic==0:
            print("  PREDICTOR---------------------------------")
            print("    Hard X    (KJ) : ", MSE_HARD[0, 3], sep='')
            print("    Hard X, R (UJ) : ", MSE_HARD[0, 6], ", ", MSE_HARD[0, 9], sep='')
            print("  FILTER------------------------------------")
            print("    Hard X    (KJ) : ", MSE_HARD[0, 1], sep='')
            print("    Hard X, R (UJ) : ", MSE_HARD[0, 4], ", ", MSE_HARD[0, 7], sep='')
            print("  SMOOTHER----------------------------------")
            print("    Hard X    (KJ) : ", MSE_HARD[0, 2], sep='')
            print("    Hard X, R (UJ) : ", MSE_HARD[0, 5], ", ", MSE_HARD[0, 8], sep='')

        print("--- %s seconds ---" % (int(elapsed_time)), flush=True)


    def printResultSignal(self, MSE, MSE_HARD, STEPS, elapsed_time, hard, filt, smooth, predic, ch=''):
        MSE      = np.around(MSE,      decimals=3)
        MSE_HARD = np.around(MSE_HARD, decimals=3)

        print("\n================ Restoration results "+ ch + " ===============")
        # print("  mse_Y\t:", MSE[0, 0])
        print("  STEPS (F)        : ", STEPS[0], sep='')
        print("")
        if predic>0:
            print("  PREDICTOR---------------------------------")
            print("    Fuzzy X  (UJ)  : ", MSE[:, 6], sep='')
        if hard and predic>0:
            print("      Hard X (UJ)  : ", MSE_HARD[0, 6], sep='')
            print("")
        if filt:
            print("  FILTER------------------------------------")
            print("    Fuzzy X  (UJ)  : ", MSE[:, 4], sep='')
        if hard and filt:
            print("      Hard X (UJ)  : ", MSE_HARD[0, 4], sep='')
            print("")
        if smooth:
            print("  SMOOTHER----------------------------------")
            print("    Fuzzy X  (UJ)  : ", MSE[:, 5], sep='')
        if hard and smooth:
            print("      Hard X  (UJ) : ", MSE_HARD[0, 5], sep='')
        if hard and filt==False and smooth==False and predic==0:
            print("  PREDICTOR---------------------------------")
            print("    Hard X    (UJ) : ", MSE_HARD[0, 6], sep='')
            print("  FILTER------------------------------------")
            print("    Hard X    (UJ) : ", MSE_HARD[0, 4], sep='')
            print("  SMOOTHER----------------------------------")
            print("    Hard X    (UJ) : ", MSE_HARD[0, 5], sep='')

        print("--- %s seconds ---" % (int(elapsed_time)), flush=True)



    def plotMSE(self, STEPS, MSE, MSE_HARD, FStext, hard, filt, smooth, predic):
        # Dessin MSE en fonction de F
        fig, axs = plt.subplots(2, 1, sharex=True)
        fig.subplots_adjust(hspace=0.1)

        ListeMaxiJumps = []
        if filt:
            axs[0].plot(STEPS, MSE[:, 7], dashes=[3, 1, 3, 1], color='b', label='Fuzzy filter')
            ListeMaxiJumps.append(max(MSE[:, 7]))
            if hard:
                hard_jump_filter = np.ones(len(MSE_HARD[:, 7])) * MSE_HARD[0, 7]
                axs[0].plot(STEPS, hard_jump_filter, dashes=[5, 2, 5, 2], color='k', label='Hard filter')
                ListeMaxiJumps.append(max(MSE_HARD[:, 7]))
        if smooth:
            axs[0].plot(STEPS, MSE[:, 8], dashes=[6, 6, 6, 6], color='c', label='Fuzzy smoother')
            ListeMaxiJumps.append(max(MSE[:, 8]))
            if hard:
                hard_jump_smoother = np.ones(len(MSE_HARD[:, 8])) * MSE_HARD[0, 8]
                axs[0].plot(STEPS, hard_jump_smoother, dashes=[1, 3, 1, 3], color='k', label='Hard smoother')
                ListeMaxiJumps.append(max(MSE_HARD[:, 8]))
        if predic>0:
            axs[0].plot(STEPS, MSE[:, 9], dashes=[6, 6, 6, 6], color='c', label='Fuzzy predictor')
            ListeMaxiJumps.append(max(MSE[:, 9]))
            if hard:
                hard_jump_smoother = np.ones(len(MSE_HARD[:, 9])) * MSE_HARD[0, 9]
                axs[0].plot(STEPS, hard_jump_smoother, dashes=[1, 3, 1, 3], color='k', label='Hard predictor')
                ListeMaxiJumps.append(max(MSE_HARD[:, 9]))
        axs[0].set_xticks(STEPS)
        axs[0].set_ylabel('MSE (Jumps)', fontsize=fontS)
        maxi = max(ListeMaxiJumps)
        axs[0].set_ylim((0, maxi*1.05))
        axs[0].legend()

        ListeMaxiStates = []
        if filt:
            superv_filter = np.ones(len(MSE[:, 1])) * MSE[0, 1]
            axs[1].plot(STEPS, MSE[:, 4], dashes=[3, 1, 3, 1], color='b', label='Fuzzy filter - UJ')
            axs[1].plot(STEPS, superv_filter, color='b', label='Fuzzy filter - KJ')
            ListeMaxiStates.append(max(MSE[:, 1]))
            ListeMaxiStates.append(max(MSE[:, 4]))
            ListeMaxiStates.append(max(superv_filter))
            if hard:
                hard_filter = np.ones(len(MSE_HARD[:, 4])) * MSE_HARD[0, 4]
                axs[1].plot(STEPS, hard_filter, dashes=[5, 2, 5, 2], color='k', label='Hard filter - UJ')
                ListeMaxiStates.append(max(MSE_HARD[:, 4]))
                ListeMaxiStates.append(max(hard_filter))
        if smooth:
            superv_smoother = np.ones(len(MSE[:, 2])) * MSE[0, 2]
            axs[1].plot(STEPS, MSE[:, 5], dashes=[6, 6, 6, 6], color='c', label='Fuzzy smoother - UJ')
            axs[1].plot(STEPS, superv_smoother, color='c', label='Fuzzy smoother - KJ')
            ListeMaxiStates.append(max(MSE[:, 2]))
            ListeMaxiStates.append(max(MSE[:, 5]))
            ListeMaxiStates.append(max(superv_smoother))
            if hard:
                hard_smoother = np.ones(len(MSE_HARD[:, 5])) * MSE_HARD[0, 5]
                axs[1].plot(STEPS, hard_smoother, dashes=[1, 3, 1, 3], color='k', label='Hard smoother - UJ')
                ListeMaxiStates.append(max(MSE_HARD[:, 5]))
                ListeMaxiStates.append(max(hard_smoother))
        if predic>0:
            superv_predictor = np.ones(len(MSE[:, 3])) * MSE[0, 3]
            axs[1].plot(STEPS, MSE[:, 6], dashes=[2, 4, 2, 4], color='m', label='Fuzzy predictor - UJ')
            axs[1].plot(STEPS, superv_predictor, color='m', label='Fuzzy predictor - KJ')
            ListeMaxiStates.append(max(MSE[:, 2]))
            ListeMaxiStates.append(max(MSE[:, 6]))
            ListeMaxiStates.append(max(superv_predictor))
            if hard:
                hard_smoother = np.ones(len(MSE_HARD[:, 6])) * MSE_HARD[0, 6]
                axs[1].plot(STEPS, hard_smoother, dashes=[1, 3, 1, 3], color='k', label='Hard predictor - UJ')
                ListeMaxiStates.append(max(MSE_HARD[:, 6]))
                ListeMaxiStates.append(max(hard_smoother))

        axs[1].set_xticks(STEPS)
        axs[1].set_ylabel('MSE (States)', fontsize=fontS)
        maxi = max(ListeMaxiStates)
        axs[1].set_ylim((0, maxi*1.05 ))
        axs[1].legend()

        plt.xlabel('$F$', fontsize=fontS)
        #plt.ylabel('MSE')
        #plt.title('Mean MSE according to $F$')
        fname='Result/Fuzzy/Figures/' + FStext + '_MSE.png'
        plt.savefig(fname, bbox_inches='tight', dpi=DPI)

        plt.close()

    def PlotSimul(self, X, Rfuzzy, Y):

        s_Graph = slice(self.__graph_mini, self.__graph_maxi)
        abscisse= np.linspace(start=self.__graph_mini, stop=self.__graph_maxi-1, num=self.__graph_maxi-self.__graph_mini)

        plt.figure()
        plt.plot(abscisse, X[s_Graph, 0], color='b', label='Simulated states')
        plt.plot(abscisse, Y[s_Graph, 0], color='k', label='Observations')
        plt.xlabel('n', fontsize=fontS)
        plt.xlim(right=self.__graph_maxi-1, left=self.__graph_mini)
        plt.legend()
        plt.savefig('./Result/Fuzzy/Figures/SimuXY_SerieX_Y', bbox_inches='tight', dpi=DPI)
        plt.close()

        plt.figure()
        plt.plot(abscisse, Rfuzzy[s_Graph], color='c', label='Simulated jumps')
        plt.xlabel('n', fontsize=fontS)
        plt.xlim(right=self.__graph_maxi-1, left=self.__graph_mini)
        plt.legend()
        plt.savefig('./Result/Fuzzy/Figures/SimuR_SerieX_Y', bbox_inches='tight', dpi=DPI)
        plt.close()

    def PlotTrajectories(self, ch, ch2, X, Rfuzzy, Y, E_X, E_X_O, E_R, bottom=None):
        
        s_Graph = slice(self.__graph_mini, self.__graph_maxi)
        abscisse= np.linspace(start=self.__graph_mini, stop=self.__graph_maxi-1, num=self.__graph_maxi-self.__graph_mini)

        plt.figure()
        plt.plot(abscisse, X[s_Graph, 0], color='g', label='Simulated states')
        #plt.plot(abscisse, Y[s_Graph, 0], color='k', label='Observations')
        plt.plot(abscisse, E_X[s_Graph, 0], color='b', dashes=[5, 2, 5, 2], label='Restored (KJ)')
        if len(np.shape(E_X_O)) != 0:
            plt.plot(abscisse, E_X_O[s_Graph, 0], color='r', dashes=[3, 1, 3, 1], label='Restored (UJ)')
        plt.xlabel('n', fontsize=fontS)
        plt.xlim(right=self.__graph_maxi-1, left=self.__graph_mini)
        plt.legend()
        plt.title(ch2, fontsize=fontS)
        #plt.title('Observations and states (simulated and restored using CGOFMSM)')
        #plt.title('States : simulated and restored using CGOFMSM')
        plt.savefig('./Result/Fuzzy/Figures/' + ch + '_XY_CGOFMSM_restored', bbox_inches='tight', dpi=DPI)
        plt.close()

        if len(np.shape(E_R)) != 0:
            plt.figure()
            plt.plot(abscisse, Rfuzzy[s_Graph], color='g', label='Simulated jumps')
            plt.plot(abscisse, E_R[s_Graph], color='r', dashes=[3, 1, 3, 1], label='Restored')
            plt.xlabel('n', fontsize=fontS)
            plt.xlim(right=self.__graph_maxi-1, left=self.__graph_mini)
            if bottom != None:
                plt.ylim(bottom=bottom)
            plt.legend()
            plt.title(ch2, fontsize=fontS)
            plt.savefig('./Result/Fuzzy/Figures/' + ch + '_R_CGOFMSM_restored', bbox_inches='tight', dpi=DPI)
            plt.close()


    def PlotTrajectoriesAllSignal(self, weekend_indices, mini, maxi, filestem, STEPS, ch, ch2, Data, E_X_OFA, E_X_OSA, E_X_OPA):

        s_Graph = slice(mini, maxi)

        # Dataframe preparation
        #####################################################
        if len(np.shape(E_X_OFA)) != 0:
            Data = Data.assign(E_X_OFA=E_X_OFA)
        if len(np.shape(E_X_OSA)) != 0:
            Data = Data.assign(E_X_OSA=E_X_OSA)
        if len(np.shape(E_X_OPA)) != 0:
            Data = Data.assign(E_X_OPA=E_X_OPA)
        listeHeader = list(Data)

        # Plot of the X data
        #####################################################@
        fig, ax = plt.subplots()

        color = 'tab:red'
        ax.set_ylabel(listeHeader[1], color=color, fontsize=fontS)
        ax.plot(Data.index[s_Graph], Data[listeHeader[1]].iloc[s_Graph], label='True states', color=color)
        if len(np.shape(E_X_OFA)) != 0:
            ax.plot(Data.index[s_Graph], Data.E_X_OFA[s_Graph], label='Filtered', color='tab:blue')
        if len(np.shape(E_X_OSA)) != 0:
            ax.plot(Data.index[s_Graph], Data.E_X_OSA[s_Graph], label='Smoothed', color='tab:orange')
        if len(np.shape(E_X_OPA)) != 0:
            ax.plot(Data.index[s_Graph], Data.E_X_OPA[s_Graph], label='Predicted', color='tab:purple')

        ax.tick_params(axis='y', labelcolor=color, labelsize=fontS-2)

        # surlignage des jours de WE
        i = 0
        while i < len(weekend_indices)-1:
            ax.axvspan(Data.index[weekend_indices[i]], Data.index[weekend_indices[i+1]], facecolor='gray', edgecolor='none', alpha=.25, zorder=-100)
            i += 2

        # format the ticks
        ax.xaxis.set_major_locator(months)
        ax.xaxis.set_major_formatter(monthFmt)
        ax.xaxis.set_minor_locator(days)
        ax.xaxis.set_minor_formatter(dayFmt)
        #ax.format_xdata = md.DateFormatter('%m-%d')
        ax.grid(True, which='major', axis='both')
        ax.set_title('Data: ' + filestem + ' (F=' + str(STEPS) + ')', fontsize=fontS+2)
        ax.tick_params(axis='x', which='both', labelsize=fontS-2)
        ax.set_xlim(xmin=Data.index[mini], xmax=Data.index[maxi-1])

        fig.autofmt_xdate()
        plt.xticks(rotation=35)
        plt.legend()
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.savefig('./Result/Fuzzy/Figures/' + ch + '_XY_CGOFMSM_restored', bbox_inches='tight', dpi=DPI)
        plt.close()


    def PlotTrajectoriesSignal(self, weekend_indices, filestem, STEPS, ch, ch2, Data, E_X, E_R):

        s_Graph = slice(self.__graph_mini, self.__graph_maxi)

        # Dataframe preparation
        #####################################################@
        Data = Data.assign(X_E=E_X)
        Data = Data.assign(R_E=E_R)
        listeHeader = list(Data)

        # Plot of the X data
        #####################################################@
        fig, ax = plt.subplots()

        color = 'tab:red'
        ax.set_ylabel(listeHeader[1], color=color, fontsize=fontS)
        ax.plot(Data.index[s_Graph], Data[listeHeader[1]].iloc[s_Graph], label='True states', color=color)
        ax.plot(Data.index[s_Graph], Data.X_E[s_Graph], label='Estimated states', color=color, dashes=[4,2,4,2])
        ax.tick_params(axis='y', labelcolor=color, labelsize=fontS-2)

        # surlignage des jours de WE
        i = 0
        while i < len(weekend_indices)-1:
            ax.axvspan(Data.index[weekend_indices[i]], Data.index[weekend_indices[i+1]], facecolor='gray', edgecolor='none', alpha=.25, zorder=-100)
            i += 2

        # format the ticks
        ax.xaxis.set_major_locator(months)
        ax.xaxis.set_major_formatter(monthFmt)
        ax.xaxis.set_minor_locator(days)
        ax.xaxis.set_minor_formatter(dayFmt)
        #ax.format_xdata = md.DateFormatter('%m-%d')
        ax.grid(True, which='major', axis='both')
        ax.set_title('Data: ' + filestem + ' (F=' + str(STEPS) + ')', fontsize=fontS+2)
        ax.tick_params(axis='x', which='both', labelsize=fontS-2)
        ax.set_xlim(xmin=Data.index[self.__graph_mini], xmax=Data.index[self.__graph_maxi-1])

        fig.autofmt_xdate()
        plt.xticks(rotation=35)
        plt.legend()
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.savefig('./Result/Fuzzy/Figures/' + ch + '_XY_CGOFMSM_restored', bbox_inches='tight', dpi=DPI)
        plt.close()
        np.savetxt('./Result/Fuzzy/Result_csv/' + ch + '_XY_CGOFMSM_restored.csv', Data.X_E, delimiter=',')

        if len(np.shape(E_R)) != 0:
            fig, ax = plt.subplots()

            if STEPS != 0:
                Rcentres = np.linspace(start=1./(2.*STEPS), stop=1.0-1./(2.*STEPS), num=STEPS, endpoint=True)
            else:
                Rcentres = np.empty(shape=(0,))
            Centres = np.zeros(shape=(STEPS+2))
            Centres[0]         = 0.
            Centres[STEPS+1]   = 1.
            Centres[1:STEPS+1] = Rcentres
            
            color = 'tab:green'
            ax.set_ylabel('Discrete fuzzy jumps', color=color, fontsize=fontS)
            ax.plot(Data.index[s_Graph], Data.R_E[s_Graph], label='Estimated jumps', color=color, dashes=[4,2,4,2])
            if 'R_GT' in Data.columns:
                ax.plot(Data.index[s_Graph], Data.R_GT[s_Graph], label='True jumps', color=color)
            
            ax.tick_params(axis='y', labelcolor=color, labelsize=fontS-2)
            ax.tick_params(axis='x', which='both', labelsize=fontS-2)
            
            ax.set_ylim(ymax=1.05, ymin=-0.05)
            ax.set_xlim(xmin=Data.index[self.__graph_mini], xmax=Data.index[self.__graph_maxi-1])

            ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
            color = 'tab:olive'
            ax2.hlines (Centres, xmin=Data.index[self.__graph_mini], xmax=Data.index[self.__graph_maxi-1], color=color, linestyle='dashed')
            ax2.tick_params(axis='y', labelcolor=color, labelsize=fontS-2)
            ax2.set_yticks(ticks=Centres)

            # format the ticks
            ax.xaxis.set_major_locator(months)
            ax.xaxis.set_major_formatter(monthFmt)
            ax.xaxis.set_minor_locator(days)
            ax.xaxis.set_minor_formatter(dayFmt)
            # ax.format_xdata = md.DateFormatter('%m%Y-%d')

            plt.setp(ax.xaxis.get_minorticklabels(), rotation=90)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=35)
            ax.set_title('Data: ' + filestem, fontsize=fontS+2)
            
            i = 0
            while i < len(weekend_indices)-1:
                ax.axvspan(Data.index[weekend_indices[i]], Data.index[weekend_indices[i+1]], facecolor='gray', edgecolor='none', alpha=.25, zorder=-100)
                i += 2

            fig.autofmt_xdate()
            plt.savefig('./Result/Fuzzy/Figures/' + ch + '_R_CGOFMSM_restored', bbox_inches='tight', dpi=DPI)
            plt.close()
            np.savetxt('./Result/Fuzzy/Result_csv/' + ch + '_R_CGOFMSM_restored.csv', Data.R_E, delimiter=',')

            
    def PlotFuzzyHard(self, ch, ch2, X, E_X_HARD, E_X, bottom=None):

        s_Graph = slice(self.__graph_mini, self.__graph_maxi)
        abscisse= np.linspace(start=self.__graph_mini, stop=self.__graph_maxi-1, num=self.__graph_maxi-self.__graph_mini)

        plt.figure()
        plt.plot(abscisse, X[s_Graph, 0], color='g',                      label='Simulated states')
        if len(np.shape(E_X)) != 0:
            plt.plot(abscisse, E_X[s_Graph, 0], color='r', dashes=[3, 1, 3, 1], label='CGOFMSM')
        if len(np.shape(E_X_HARD)) != 0:
            plt.plot(abscisse, E_X_HARD[s_Graph, 0], color='b', dashes=[5, 2, 5, 2], label='CGOMSM')

        plt.xlabel('n', fontsize=fontS)
        plt.xlim(right=self.__graph_maxi-1, left=self.__graph_mini)
        if bottom != None:
            plt.ylim(bottom=bottom)
        plt.legend()
        plt.title(ch2, fontsize=fontS)
        plt.savefig('./Result/Fuzzy/Figures/' + ch + '_X_CGOFMSM_CGOMSM_restored', bbox_inches='tight', dpi=DPI)
        plt.close()

    def PlotFuzzyHard_R(self, ch, ch2, R, E_R_HARD, E_R, bottom=None):

        s_Graph = slice(self.__graph_mini, self.__graph_maxi)
        abscisse= np.linspace(start=self.__graph_mini, stop=self.__graph_maxi-1, num=self.__graph_maxi-self.__graph_mini)

        plt.figure()
        plt.plot(abscisse, R[s_Graph], color='g',                      label='Simulated states')
        if len(np.shape(E_R)) != 0:
            plt.plot(abscisse, E_R[s_Graph], color='r', dashes=[3, 1, 3, 1], label='CGOFMSM')
        if len(np.shape(E_R_HARD)) != 0:
            plt.plot(abscisse, E_R_HARD[s_Graph], color='b', dashes=[5, 2, 5, 2], label='CGOMSM')

        plt.xlabel('n', fontsize=fontS)
        plt.xlim(right=self.__graph_maxi-1, left=self.__graph_mini)
        if bottom != None:
            plt.ylim(bottom=bottom)
        plt.legend()
        plt.title(ch2, fontsize=fontS)
        plt.savefig('./Result/Fuzzy/Figures/' + ch + '_X_CGOFMSM_CGOMSM_restored', bbox_inches='tight', dpi=DPI)
        plt.close()