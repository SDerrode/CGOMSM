#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import numpy as np
import sys
import copy
import random

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as md

years    = md.YearLocator()   # every year
months   = md.MonthLocator()  # every month
days     = md.DayLocator()    # every day
yearsFmt = md.DateFormatter('%Y      ')
monthFmt = md.DateFormatter('%B  ')
dayFmt   = md.DateFormatter('%d')

import pandas as pd
import datetime as dt

from Fuzzy.InterFuzzy       import simulateFuzzy
from CommonFun.CommonFun    import MSE_PK, Error_Ratio, From_Cov_to_FQ
from CommonFun.CommonFun    import SaveSimulatedFuzzyData, ReadSimulatedFuzzyData
from CommonFun.CommonFun    import Readin_CovMeansProba
from CommonFun.CommonFun    import Test_isCGOMSM_from_Cov, Test_isCGOMSM_from_F
from OFAResto.OFAFuzzyResto import RestorationOFAFuzzy
from OFAResto.OFAResto      import RestorationOFA
from PKFResto.PKFResto      import RestorationPKF
from CGPMSMs.CGPMSMs        import GetParamNearestCGO_cov

i_min = 6      # index min for plot
i_max = 60     # index max for plot
dpi   = 300
fontS = 16     # font size
matplotlib.rc('xtick', labelsize=fontS)
matplotlib.rc('ytick', labelsize=fontS)


def main():

    """
        Programmes pour simuler et restaurer des CGOFMSM.
 
        :Example:

        >> python3 Test_CGOFMSM.py Parameters/Fuzzy/SP2018.param 2:0.07:0.24:0.09 1,1,1 72 3 5 2 1
        >> python3 Test_CGOFMSM.py Parameters/Fuzzy/SP2018.param 2:0.07:0.24:0.09 1,1,0 500 1,2,3,5,7,10 10 1 0 1
        >> nohup python3 Test_CGOFMSM.py Parameters/Fuzzy/SP2018.param 4:0.15:0.15:0.:0.1 1,1,0 1000 1,2,3,5,7,10 10 1 0 1 > serie2.out &

        argv[1] : Nom du fichier de paramètres
        argv[2] : Fuzzy joint law model and parameters. e.g. 2:0.07:0.24:0.09, or 4:0.15:0.15:0.:0.1
        argv[3] : hard Filter & Smoother (0/1), Fuzzy filter(0/1), Fuzzy smoother(0/1), e.g. 1,1,0
        argv[4] : Taille de l'échantillon
        argv[5] : Valeurs de F (une seule, ou plusieurs séparées par des virgules), e.g. 1,3,5
        argv[6] : Nombre d'expériences pour obtenir des résultats moyens
        argv[7] : Debug(3), pipelette (2), normal (1), presque muet (0)
        argv[8] : Plot graphique (0/1)
    """

    print('Ligne de commandes : ', sys.argv, flush=True)

    if len(sys.argv) != 9:
        print('CAUTION : bad number of arguments - see help')
        exit(1)

    # Default value for parameters
    filenameParamCov = 'Parameters/Fuzzy/SP2018.param'
    FSParametersStr  = '2:0.07:0.24:0.09'
    work             = [1, 1, 0]
    N                = 100
    STEPS            = [3, 7]
    NbExp            = 1
    verbose          = 2
    Plot             = True

    # Parameters from argv
    filenameParamCov = sys.argv[1]
    FSParametersStr  = sys.argv[2]
    work             = list(map(int, sys.argv[3].split(',')))
    N                = int(sys.argv[4])
    STEPS            = list(map(int, sys.argv[5].split(',')))
    NbExp            = int(sys.argv[6])
    verbose          = int(sys.argv[7])
    Plot             = True
    if int(sys.argv[8]) == 0: Plot = False

    np.random.seed(None)
    #np.random.seed(1), print('-----------------> ATTENTION : random fixé à 0!!!<------------------')

    readData = False  # Data are read or resimulated ?

    # Moyenne de plusieurs expériences
    aCGOFMSM = CGOFMSM(N, filenameParamCov, verbose, FSParametersStr)
    mean_tab_MSE, mean_tab_MSE_HARD, mean_time = aCGOFMSM.run_several(NbExp, STEPS = STEPS, Plot=Plot, Work=work)


class CGOFMSM:

    def __init__(self, N, filenameParamCov, verbose, FSParametersStr):
        assert N > 1, print('number of samples must be greater than 2')

        self.__n_r = 2 # default value, can be changed by reading of parameters (see below)
        self.__N   = N
        self.__filenameParamCov = filenameParamCov
        self.__verbose          = verbose
        self.__FSParameters     = list(map(str, FSParametersStr.split(':')))

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
            self.__n_r, X, R, Y  = simulateFuzzy(self.__filenameParamCov, self.__FSParameters, self.__N)
            assert self.__n_r == 2, print('number of jumps must be 2!')
            if Plot is True:
                self.PlotSimul(X, R, Y)
        else:
            X, R, Y = ReadSimulatedFuzzyData(filenameSimulatedXRY)
            useless, self.__N = np.shape(X)

        cpt = 0
        for i in range(self.__N):
            if R[0, i] == 0.0 or R[0, i] == 1.0:
                cpt += 1
        pHNum = cpt / self.__N
        #print('pHNum=', pHNum)
        #input('Fin simulation')

        # Save simulated data if required
        if filenameSimulatedXRY is not None and readData == False:
            SaveSimulatedFuzzyData(X, R, Y, filenameSimulatedXRY)

        return X, R, Y, pHNum


    def restore_signal(self, Data, ch, STEPS=[7], hard=True, filt=True, smooth=True, Plot=False):

        elapsed_time = 0
        start_time = time.time()

        # Create main objects
        Resto = RestorationOFAFuzzy(self.__filenameParamCov, STEPS[0], self.__FSParameters, self.__verbose)
        # print(Resto.getFS().getParam())
        # input('pause')

        self.__n_r, F, useless, Q, Cov, Mean_X, Mean_Y = Readin_CovMeansProba(self.__filenameParamCov)
        assert self.__n_r == 2, print('number of jumps must be 2!')
        n_x = np.shape(Mean_X)[1]
        n_y = np.shape(Mean_Y)[1]
        Ok = Test_isCGOMSM_from_Cov(Cov, n_x)
        if Ok == False:
            print('ATTENTION : le modele nest pas un CGO! --> TRANSFORMATION EN CGO')
            Cov = GetParamNearestCGO_cov(Cov, n_x=n_x)
            F, Q = From_Cov_to_FQ(Cov)

        listeHeader=list(Data)

        Y = np.zeros((1, self.__N))
        Y[0, :]=Data['Y'].values

        if hard == True:

            # Approximation des proba hard (par intégration des 4 quadrants)
            MProba, TProba, JProba = Resto.getFS().getTheoriticalHardTransition(self.__n_r)
            # print('Array MProba = ', MProba)
            # print('Array TProba = ', TProba)
            # print('Array JProba = ', JProba)
            # input('pause')

            E_X_OFA_HARD, E2_X_OFA_HARD, Cov_X_OFA_HARD, \
            E_X_OSA_HARD, E2_X_OSA_HARD, Cov_X_OSA_HARD, \
            useless1, useless2, useless3, \
            useless4, E_R_OFA_HARD, E_R_OSA_HARD \
                = RestorationOFA().restore_withoutjumps(Y, F, Q, Resto.getCov(), Resto.getMean_X(), Resto.getMean_Y(), TProba, MProba)

            end_time = time.time()
            elapsed_time += end_time - start_time

            if Plot is True:
                chaine = Resto.getFSText() + '_' + ch + '_FILT_HARD'
                self.PlotTrajectoriesSignal(chaine, 'Hard filter (CGOMSM)', Data, E_X_OFA_HARD, E_R_OFA_HARD)
                if smooth:
                    chaine = Resto.getFSText() + '_' + ch + '_SMOO_HARD'
                    self.PlotTrajectoriesSignal(chaine, 'Hard smoother (CGOMSM)', Data, E_X_OSA_HARD, E_R_OSA_HARD)

        if filt==True or smooth==True:

            # Loop in discrete jumps F
            for i, steps in enumerate(STEPS):
                if self.__verbose >= 1:
                    print('    #####STEPS ', steps)

                Resto.resetSTEPS(STEPS[i])

                # FUZZY: filter (and smooth) with unknown jumps
                E_X_OFA, E_R_OFA, E_R_OFA2, E_X_OSA, E_R_OSA, E_R_OSA2 = Resto.restore_Fuzzy1D(Y, filt=filt, smooth=smooth)
                end_time = time.time()
                elapsed_time += end_time - start_time

                if Plot is True:
                    if filt:
                        chaine = Resto.getFSText() + '_' + ch + '_FILT_FUZZY_STEP_' + str(steps)
                        self.PlotTrajectoriesSignal(chaine, 'Fuzzy filter (CGOMSM)', Data, E_X_OFA, E_R_OFA)
                    if smooth:
                        chaine = Resto.getFSText() + '_' + ch + '_SMOO_FUZZY_STEP_' + str(steps)
                        self.PlotTrajectoriesSignal(chaine, 'Fuzzy smoother (CGOMSM)', Data, E_X_OSA, E_R_OSA)

        return elapsed_time


    def run_several(self, nb_exp, STEPS=[5], Plot=False, Work=[1,1,0]):

        tab_MSE          = np.zeros((nb_exp, len(STEPS), 9))
        tab_MSE_HARD     = np.zeros((nb_exp, len(STEPS), 7))
        tab_elapsed_time = np.zeros((nb_exp))

        hard   = True
        filt   = True
        smooth = True
        if Work[0] == 0: hard   = False
        if Work[1] == 0: filt   = False
        if Work[2] == 0: smooth = False
        if hard==False and filt==False and smooth==False:
            print('work=', work, ' --> Not allowed !')
            exit(1)

        for e in range(nb_exp):
            if self.__verbose >= 1:
                print('\n##########Experiment ', e)
            # result of one experiment
            tab_MSE[e,:], tab_MSE_HARD[e,:], tab_elapsed_time[e], FStext, pHNum = self.run_one("EXP"+ str(e+1), STEPS=STEPS, hard=hard, filt=filt, smooth=smooth, readData=False, Plot=Plot)

        #print('tab_MSE=', tab_MSE)
        #print('tab_MSE_HARD=', tab_MSE_HARD)

        # mean computations for screening
        mean_tab_MSE      = np.mean(tab_MSE,      axis=0)
        mean_tab_MSE_HARD = np.mean(tab_MSE_HARD, axis=0)
        mean_time         = np.mean(tab_elapsed_time)

        if self.__verbose >= 0 and nb_exp > 1:
            self.printResult(mean_tab_MSE, mean_tab_MSE_HARD, STEPS, mean_time, hard=hard, filt=filt, smooth=smooth, ch= '(mean)')
        if len(STEPS)>1:
            self.plotMSE(STEPS, mean_tab_MSE, mean_tab_MSE_HARD, FStext + '_pHNum_' + str(pHNum).replace('.','_'), hard=hard, filt=filt, smooth=smooth)

        return mean_tab_MSE, mean_tab_MSE_HARD, mean_time


    def run_one(self, ch, STEPS=[7], hard=True, filt=True, smooth=True, readData = False, Plot=False):

        # Save of MSE results
        MSE      = np.zeros((len(STEPS), 9))
        MSE_HARD = np.zeros((len(STEPS), 7))

        header = 'Nombre de donnees : N = ' + str(self.__N)

        start_time   = time.time()
        elapsed_time = 0.

        # Create main objects
        Resto = RestorationOFAFuzzy(self.__filenameParamCov, STEPS[0], self.__FSParameters, self.__verbose)

        # Simulaton of a sample
        fname = 'Result/Fuzzy/SimulatedData/' + Resto.getFSText() + '_fuzzy_' + ch + '.txt'
        X, Rfuzzy, Y, pHNum = self.SimulateFuzzy(fname, readData, False)
        Rfuzzy = np.reshape(Rfuzzy, newshape=(self.__N))

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
        if hard:
            start_time = time.time()

            # Hardification des sauts flous
            Rhard = np.around(Rfuzzy).astype(int)

            # Approximation des proba hard (par intégration des 4 quadrants)
            MProba, TProba, JProba = Resto.getFS().getTheoriticalHardTransition(self.__n_r)

            # Filtrage Hard avec les jumps
            E_X_PF_HARD, E2_X_PF_HARD,  E_X_PS_HARD, E2_X_PS_HARD\
                = RestorationPKF().restore_withjump(Y, Rhard, F, Q, Resto.getCov(), Resto.getMean_X(), Resto.getMean_Y(), Likelihood=False)

            MSE_HARD[0, 1] = MSE_PK(E_X_PF_HARD, X)
            MSE_HARD[0, 2] = MSE_PK(E_X_PS_HARD, X)

            # Filtrage Hard sans les jumps
            E_X_OFA_HARD, E2_X_OFA_HARD, Cov_X_OFA_HARD, \
            E_X_OSA_HARD, E2_X_OSA_HARD, Cov_X_OSA_HARD, \
            useless0, useless1, useless2, \
            useless3, E_R_OFA_HARD, E_R_OSA_HARD\
                = RestorationOFA().restore_withoutjumps(Y, F, Q, Resto.getCov(), Resto.getMean_X(), Resto.getMean_Y(), TProba, MProba)
            # print('E_R_OFA_HARD=', E_R_OFA_HARD[0:5])
            # print('E_X_OFA_HARD=', E_X_OFA_HARD[0:5])
            elapsed_time += time.time() - start_time

            MSE_HARD[0, 3] = MSE_PK(E_X_OFA_HARD, X)
            MSE_HARD[0, 5] = MSE_PK(E_R_OFA_HARD, Rfuzzy)
            MSE_HARD[0, 4] = MSE_PK(E_X_OSA_HARD, X)
            MSE_HARD[0, 6] = MSE_PK(E_R_OSA_HARD, Rfuzzy)

            if Plot is True:
                if filt:
                    chaine = Resto.getFSText() + '_pHNum_' + str(pHNum).replace('.','_') + '_' + ch + '_FILT_HARD'
                    self.PlotTrajectories(chaine, 'Hard filter (CGOMSM)', X, Rhard, Y, E_X_PF_HARD, E_X_OFA_HARD, E_R_OFA_HARD, bottom=0.)
                if smooth:
                    chaine = Resto.getFSText() + '_pHNum_' + str(pHNum).replace('.','_') + '_' + ch + '_SMOO_HARD'
                    self.PlotTrajectories(chaine, 'Hard smoother (CGOMSM)', X, Rhard, Y, E_X_PS_HARD, E_X_OSA_HARD, E_R_OSA_HARD, bottom=0.)


        # Fuzzy : Restoration with known and unknown fuzzy jumps
        #####################################################
        if filt==True or smooth==True:
            start_time = time.time()

            # Fuzzy : Restoration with known jumps
            E_X_PF, Cov_X_PF, E_X_PS, Cov_X_PS \
                = RestorationPKF().restore_withfuzzyjump(Y, Rfuzzy, Resto.getCov(), Resto.getMean_X(), Resto.getMean_Y(), Likelihood=False, smooth=smooth)
            MSE[0, 1] = MSE_PK(E_X_PF, X)
            MSE[0, 2] = MSE_PK(E_X_PS, X)
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
                E_X_OFA, E_R_OFA, E_R_OFA2, E_X_OSA, E_R_OSA, E_R_OSA2 = Resto.restore_Fuzzy1D(Y, filt=filt, smooth=smooth)
                elapsed_time += time.time() - start_time

                # MSE
                MSE[i, 3] = MSE_PK(E_X_OFA, X)
                MSE[i, 5] = MSE_PK(E_R_OFA, Rfuzzy)
                MSE[i, 7] = MSE_PK(E_R_OFA2, Rfuzzy)

                if smooth:
                    MSE[i, 4] = MSE_PK(E_X_OSA, X)
                    MSE[i, 6] = MSE_PK(E_R_OSA, Rfuzzy)
                    MSE[i, 8] = MSE_PK(E_R_OSA2, Rfuzzy)

                if Plot is True:
                    if filt:
                        chaine = Resto.getFSText() + '_pHNum_' + str(pHNum).replace('.','_') + '_' + ch + '_STEPS' + str(steps) + '_FILT'
                        self.PlotTrajectories(chaine, 'Fuzzy filter (CGOFMSM)', X, Rfuzzy, Y, E_X_PF, E_X_OFA, E_R_OFA, bottom=0.)
                        chaine = Resto.getFSText() + '_pHNum_' + str(pHNum).replace('.','_') + '_' + ch + '_STEPS' + str(steps) + '_FILT_MPM2'
                        self.PlotTrajectories(chaine, 'Fuzzy filter (CGOFMSM)', X, Rfuzzy, Y, E_X_PF, E_X_OFA, E_R_OFA2, bottom=0.)
                        if hard:
                            chaine = Resto.getFSText() + '_pHNum_' + str(pHNum).replace('.','_') + '_' + ch + '_STEPS' + str(steps) + '_FILT_UJ_X'
                            self.PlotFuzzyHard(chaine, 'CGOFMSM vs CGOMSM filters (UJ)', X, E_X_OFA_HARD, E_X_OFA)
                            chaine = Resto.getFSText() + '_pHNum_' + str(pHNum).replace('.','_') + '_' + ch + '_STEPS' + str(steps) + '_FILT_KJ_X'
                            self.PlotFuzzyHard(chaine, 'CGOFMSM vs CGOMSM filters (KJ)', X, E_X_PF_HARD, E_X_PF)
                            chaine = Resto.getFSText() + '_pHNum_' + str(pHNum).replace('.','_') + '_' + ch + '_STEPS' + str(steps) + '_FILT_UJ_R'
                            self.PlotFuzzyHard_R(chaine, 'CGOFMSM vs CGOMSM filters (UJ)', Rfuzzy, E_R_OFA_HARD, E_R_OFA, bottom=0)
                            chaine = Resto.getFSText() + '_pHNum_' + str(pHNum).replace('.','_') + '_' + ch + '_STEPS' + str(steps) + '_FILT_UJ_R_MPM2'
                            self.PlotFuzzyHard_R(chaine, 'CGOFMSM vs CGOMSM filters (UJ)', Rfuzzy, E_R_OFA_HARD, E_R_OFA2, bottom=0)

                    if smooth:
                        chaine = Resto.getFSText() + '_pHNum_' + str(pHNum).replace('.','_') + '_' + ch + '_STEPS' + str(steps) + '_SMOO'
                        self.PlotTrajectories(chaine, 'Fuzzy smoother (CGOFMSM)', X, Rfuzzy, Y, E_X_PS, E_X_OSA, E_R_OSA, bottom=0.)
                        chaine = Resto.getFSText() + '_pHNum_' + str(pHNum).replace('.','_') + '_' + ch + '_STEPS' + str(steps) + '_SMOO_MPM2'
                        self.PlotTrajectories(chaine, 'Fuzzy smoother (CGOFMSM)', X, Rfuzzy, Y, E_X_PS, E_X_OSA, E_R_OSA2, bottom=0.)
                        if hard:
                            chaine = Resto.getFSText() + '_pHNum_' + str(pHNum).replace('.','_') + '_' + ch + '_STEPS' + str(steps) + '_SMOO_UJ_X'
                            self.PlotFuzzyHard(chaine, 'CGOFMSM vs CGOMSM smoothers (UJ)', X, E_X_OSA_HARD, E_X_OSA)
                            chaine = Resto.getFSText() + '_pHNum_' + str(pHNum).replace('.','_') + '_' + ch + '_STEPS' + str(steps) + '_SMOO_KJ_X'
                            self.PlotFuzzyHard(chaine, 'CGOFMSM vs CGOMSM smoothers (KJ)', X, E_X_PS_HARD, E_X_PS)
                            chaine = Resto.getFSText() + '_pHNum_' + str(pHNum).replace('.','_') + '_' + ch + '_STEPS' + str(steps) + '_SMOO_UJ_R'
                            self.PlotFuzzyHard_R(chaine, 'CGOFMSM vs CGOMSM smoothers (UJ)', Rfuzzy, E_R_OSA_HARD, E_R_OSA, bottom=0)
                            chaine = Resto.getFSText() + '_pHNum_' + str(pHNum).replace('.','_') + '_' + ch + '_STEPS' + str(steps) + '_SMOO_UJ_R_MPM2'
                            self.PlotFuzzyHard_R(chaine, 'CGOFMSM vs CGOMSM smoothers (UJ)', Rfuzzy, E_R_OSA_HARD, E_R_OSA2, bottom=0)

        if self.__verbose >= 1:
            self.printResult(MSE, MSE_HARD, STEPS, elapsed_time, hard=hard, filt=filt, smooth=smooth)

        # SAVE THE RESULT IN CSV FILE
        if filt or smooth:
            fname='Result/Fuzzy/Tab_MSE/' + Resto.getFSText() + '_fuzzy_' + ch + '.txt'
            np.savetxt(fname, MSE, delimiter='\n', newline='\n', header=header, encoding=None)
        if hard:
            fname='Result/Fuzzy/Tab_MSE/' + Resto.getFSText() + '_hard_' + ch + '.txt'
            np.savetxt(fname, MSE_HARD, delimiter='\n', newline='\n', header=header, encoding=None)

        return MSE, MSE_HARD, elapsed_time, Resto.getFSText(), pHNum


    def printResult(self, MSE, MSE_HARD, STEPS, elapsed_time, hard, filt, smooth, ch=''):
        MSE      = np.around(MSE,      decimals=3)
        MSE_HARD = np.around(MSE_HARD, decimals=3)

        print("\n================ Restoration results "+ ch + " ===============")
        # print("  mse_Y\t:", MSE[0, 0])
        print("  STEPS (F)\t\t: ", STEPS, sep='')
        print("")
        if filt:
            print("  FILTER------------------------------------")
            print("    Fuzzy X  (KJ)\t: ", MSE[0, 1], sep='')
            print("    Fuzzy X  (UJ)\t: ", MSE[:, 3], sep='')
            print("    Fuzzy R  (UJ)\t: ", MSE[:, 5], sep='')
            print("    Fuzzy R MPM2 (UJ)\t: ", MSE[:, 7], sep='')
        if hard and filt:
            print("      Hard X (KJ)\t: ", MSE_HARD[0, 1], sep='')
            print("      Hard X (UJ)\t: ", MSE_HARD[0, 3], sep='')
            print("      Hard R (UJ)\t: ", MSE_HARD[0, 5], sep='')
            print("")
        if smooth:
            print("  SMOOTHER------------------------------------")
            print("    Fuzzy X  (KJ)\t: ", MSE[0, 2], sep='')
            print("    Fuzzy X  (UJ)\t: ", MSE[:, 4], sep='')
            print("    Fuzzy R  (UJ)\t: ", MSE[:, 6], sep='')
            print("    Fuzzy R MPM2 (UJ)\t: ", MSE[:, 8], sep='')
        if hard and smooth:
            print("      Hard X  (KJ)\t: ", MSE_HARD[0, 2], sep='')
            print("      Hard X  (UJ)\t: ", MSE_HARD[0, 4], sep='')
            print("      Hard R  (UJ)\t: ", MSE_HARD[0, 6], sep='')
        if hard and filt==False and smooth==False:
            print("  FILTER------------------------------------")
            print("    Hard X    (KJ)\t: ", MSE_HARD[0, 1], sep='')
            print("    Hard X, R (UJ)\t: ", MSE_HARD[0, 3], ", ", MSE_HARD[0, 5], sep='')
            print("  SMOOTHER------------------------------------")
            print("    Hard X    (KJ)\t: ", MSE_HARD[0, 2], sep='')
            print("    Hard X, R (UJ)\t: ", MSE_HARD[0, 4], ", ", MSE_HARD[0, 6], sep='')

        print("--- %s seconds ---" % (int(elapsed_time)), flush=True)

    def plotMSE(self, STEPS, MSE, MSE_HARD, FStext, hard, filt, smooth):
        # Dessin MSE en fonction de F
        fig, axs = plt.subplots(2, 1, sharex=True)
        fig.subplots_adjust(hspace=0.1)

        ListeMaxiJumps = []
        if filt:
            axs[0].plot(STEPS, MSE[:, 5], dashes=[3, 1, 3, 1], color='b', label='Fuzzy filter')
            axs[0].plot(STEPS, MSE[:, 7], dashes=[3, 1, 3, 1], color='r', label='Fuzzy filter - MPM2')
            ListeMaxiJumps.append(max(MSE[:, 5]))
            ListeMaxiJumps.append(max(MSE[:, 7]))
            if hard:
                hard_jump_filter = np.ones(len(MSE_HARD[:, 5])) * MSE_HARD[0, 5]
                axs[0].plot(STEPS, hard_jump_filter, dashes=[5, 2, 5, 2], color='k', label='Hard filter')
                ListeMaxiJumps.append(max(MSE_HARD[:, 5]))
        if smooth:
            axs[0].plot(STEPS, MSE[:, 6], dashes=[6, 6, 6, 6], color='c', label='Fuzzy smoother')
            axs[0].plot(STEPS, MSE[:, 8], dashes=[6, 6, 6, 6], color='r', label='Fuzzy smoother - MPM2')
            ListeMaxiJumps.append(max(MSE[:, 6]))
            ListeMaxiJumps.append(max(MSE[:, 8]))
            if hard:
                hard_jump_smoother = np.ones(len(MSE_HARD[:, 6])) * MSE_HARD[0, 6]
                axs[0].plot(STEPS, hard_jump_smoother, dashes=[1, 3, 1, 3], color='k', label='Hard smoother')
                ListeMaxiJumps.append(max(MSE_HARD[:, 6]))
        axs[0].set_xticks(STEPS)
        axs[0].set_ylabel('MSE (Jumps)', fontsize=fontS)
        maxi = max(ListeMaxiJumps)
        axs[0].set_ylim((0, maxi*1.05))
        axs[0].legend()

        ListeMaxiStates = []
        if filt:
            superv_filter = np.ones(len(MSE[:, 1])) * MSE[0, 1]
            axs[1].plot(STEPS, MSE[:, 3], dashes=[3, 1, 3, 1], color='b', label='Fuzzy filter - UJ')
            axs[1].plot(STEPS, superv_filter, color='b', label='Fuzzy filter - KJ')
            ListeMaxiStates.append(max(MSE[:, 1]))
            ListeMaxiStates.append(max(MSE[:, 3]))
            ListeMaxiStates.append(max(superv_filter))
        if hard:
            hard_filter = np.ones(len(MSE_HARD[:, 3])) * MSE_HARD[0, 3]
            axs[1].plot(STEPS, hard_filter, dashes=[5, 2, 5, 2], color='k', label='Hard filter - UJ')
            ListeMaxiStates.append(max(MSE_HARD[:, 3]))
            ListeMaxiStates.append(max(hard_filter))
        if smooth:
            superv_smoother = np.ones(len(MSE[:, 2])) * MSE[0, 2]
            axs[1].plot(STEPS, MSE[:, 4], dashes=[6, 6, 6, 6], color='c', label='Fuzzy smoother - UJ')
            axs[1].plot(STEPS, superv_smoother, color='c', label='Fuzzy smoother - KJ')
            ListeMaxiStates.append(max(MSE[:, 2]))
            ListeMaxiStates.append(max(MSE[:, 4]))
            ListeMaxiStates.append(max(superv_smoother))
            if hard:
                hard_smoother = np.ones(len(MSE_HARD[:, 4])) * MSE_HARD[0, 4]
                axs[1].plot(STEPS, hard_smoother, dashes=[1, 3, 1, 3], color='k', label='Hard smoother - UJ')
                ListeMaxiStates.append(max(MSE_HARD[:, 4]))
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
        plt.savefig(fname, bbox_inches='tight', dpi=dpi)

        plt.close()

    def PlotSimul(self, X, Rfuzzy, Y):
        abscisse= np.linspace(start=i_min, stop=i_max-1, num=i_max-i_min)
        plt.figure()
        plt.plot(abscisse, X[0, i_min:i_max], color='b', label='Simulated states')
        plt.plot(abscisse, Y[0, i_min:i_max], color='k', label='Observations')
        plt.xlabel('n', fontsize=fontS)
        plt.xlim(right=i_max-1, left=i_min)
        plt.legend()
        plt.savefig('./Result/Fuzzy/Figures/SimuXY_SerieX_Y', bbox_inches='tight', dpi=dpi)
        plt.close()

        plt.figure()
        plt.plot(abscisse, Rfuzzy[0, i_min:i_max], color='c', label='Simulated jumps')
        plt.xlabel('n', fontsize=fontS)
        plt.xlim(right=i_max-1, left=i_min)
        plt.legend()
        plt.savefig('./Result/Fuzzy/Figures/SimuR_SerieX_Y', bbox_inches='tight', dpi=dpi)
        plt.close()

    def PlotTrajectories(self, ch, ch2, X, Rfuzzy, Y, E_X, E_X_O, E_R, bottom=None):
        abscisse= np.linspace(start=i_min, stop=i_max-1, num=i_max-i_min)
        plt.figure()
        plt.plot(abscisse, X[0, i_min:i_max], color='g', label='Simulated states')
        #plt.plot(abscisse, Y[0, i_min:i_max], color='k', label='Observations')
        plt.plot(abscisse, E_X[0, i_min:i_max], color='b', dashes=[5, 2, 5, 2], label='Restored (KJ)')
        plt.plot(abscisse, E_X_O[0, i_min:i_max], color='r', dashes=[3, 1, 3, 1], label='Restored (UJ)')
        plt.xlabel('n', fontsize=fontS)
        plt.xlim(right=i_max-1, left=i_min)
        plt.legend()
        plt.title(ch2, fontsize=fontS)
        #plt.title('Observations and states (simulated and restored using CGOFMSM)')
        #plt.title('States : simulated and restored using CGOFMSM')
        plt.savefig('./Result/Fuzzy/Figures/' + ch + '_XY_CGOFMSM_restored', bbox_inches='tight', dpi=dpi)
        plt.close()

        plt.figure()
        plt.plot(abscisse, Rfuzzy[i_min:i_max], color='g', label='Simulated jumps')
        plt.plot(abscisse, E_R[i_min:i_max], color='r', dashes=[3, 1, 3, 1], label='Restored')
        plt.xlabel('n', fontsize=fontS)
        plt.xlim(right=i_max-1, left=i_min)
        if bottom != None:
            plt.ylim(bottom=bottom)
        plt.legend()
        plt.title(ch2, fontsize=fontS)
        plt.savefig('./Result/Fuzzy/Figures/' + ch + '_R_CGOFMSM_restored', bbox_inches='tight', dpi=dpi)
        plt.close()

    def PlotTrajectoriesSignal(self, ch, ch2, Data, E_X, E_R):

        #print('shapeX=', E_X.reshape((self.__N)).shape)
        Data1 = Data.assign (e=E_X.reshape((self.__N)))
        #Data2 = Data1.assign(f=E_R.reshape((self.__N)))
        Data2 = Data1.assign(f=E_R)
        Data2.rename(columns={'e' : 'X_E'}, inplace=True)
        Data2.rename(columns={'f' : 'R_E'}, inplace=True)
        # listeHeader = list(Data2)
        # print(listeHeader)
        # print(Data2.head(35))
        # print(E_X[0,0:10])
        # input('pause')

        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(Data2.index, Data2.X,   label='True states', color='r', dashes=[2,2,2,2])
        ax.plot(Data2.index, Data2.X_E, label='Estimated states', color='r')
        
        # format the ticks
        # format the ticks
        # ax.xaxis.set_major_locator(years)
        # ax.xaxis.set_major_formatter(yearsFmt)
        # ax.xaxis.set_minor_locator(months)
        # ax.xaxis.set_minor_formatter(monthFmt)
        ax.xaxis.set_major_locator(months)
        ax.xaxis.set_major_formatter(monthFmt)
        ax.xaxis.set_minor_locator(days)
        ax.xaxis.set_minor_formatter(dayFmt)

        # datemin = np.datetime64(absci[0], 'Y')
        # datemax = np.datetime64(absci[-1], 'Y') + np.timedelta64(1, 'Y')
        #ax.set_xlim(datemin, datemax)
        ax.format_xdata = md.DateFormatter('%Y-%m-%d')
        #ax.format_ydata = Y
        ax.grid(True, which='minor', axis='both')
        ax.set_title('Power consumption (kW)')
      
        # rotates and right aligns the x labels, and moves the bottom of the
        # axes up to make room for them
        fig.autofmt_xdate()
        plt.legend()
        plt.xticks(rotation=35)
        plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.15)
        fig.tight_layout()          # otherwise the right y-label is slightly clipped
        plt.savefig('./Result/Fuzzy/Figures/' + ch + '_XY_CGOFMSM_restored', bbox_inches='tight', dpi=dpi)
        plt.close()
        np.savetxt('./Result/Fuzzy/Result_csv/' + ch + '_XY_CGOFMSM_restored.csv', Data2.X_E, delimiter=',')



        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(Data2.index, Data2.R_GT, label='True jumps', color='g', dashes=[2,2,2,2])
        ax.plot(Data2.index, Data2.R_E, label='Estimated jumps', color='g')

        # format the ticks
        # format the ticks
        # ax.xaxis.set_major_locator(years)
        # ax.xaxis.set_major_formatter(yearsFmt)
        # ax.xaxis.set_minor_locator(months)
        # ax.xaxis.set_minor_formatter(monthFmt)
        ax.xaxis.set_major_locator(months)
        ax.xaxis.set_major_formatter(monthFmt)
        ax.xaxis.set_minor_locator(days)
        ax.xaxis.set_minor_formatter(dayFmt)

        # datemin = np.datetime64(absci[0], 'Y')
        # datemax = np.datetime64(absci[-1], 'Y') + np.timedelta64(1, 'Y')
        #ax.set_xlim(datemin, datemax)
        ax.format_xdata = md.DateFormatter('%Y-%m-%d')
        #ax.format_ydata = Y
        ax.grid(True, which='minor', axis='both')
        ax.set_title('Jumps')
      
        # rotates and right aligns the x labels, and moves the bottom of the
        # axes up to make room for them
        fig.autofmt_xdate()
        plt.legend()
        plt.xticks(rotation=35)
        plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.15)
        fig.tight_layout()          # otherwise the right y-label is slightly clipped
        plt.savefig('./Result/Fuzzy/Figures/' + ch + '_R_CGOFMSM_restored', bbox_inches='tight', dpi=dpi)
        plt.close()
        np.savetxt('./Result/Fuzzy/Result_csv/' + ch + '_R_CGOFMSM_restored.csv', Data2.R_E, delimiter=',')


    def PlotFuzzyHard(self, ch, ch2, X, E_X_HARD, E_X, bottom=None):

        abscisse= np.linspace(start=i_min, stop=i_max-1, num=i_max-i_min)
        plt.figure()
        plt.plot(abscisse, X       [0, i_min:i_max], color='g',                      label='Simulated states')
        plt.plot(abscisse, E_X     [0, i_min:i_max], color='r', dashes=[3, 1, 3, 1], label='CGOFMSM')
        plt.plot(abscisse, E_X_HARD[0, i_min:i_max], color='b', dashes=[5, 2, 5, 2], label='CGOMSM')

        plt.xlabel('n', fontsize=fontS)
        plt.xlim(right=i_max-1, left=i_min)
        if bottom != None:
            plt.ylim(bottom=bottom)
        plt.legend()
        plt.title(ch2, fontsize=fontS)
        plt.savefig('./Result/Fuzzy/Figures/' + ch + '_X_CGOFMSM_CGOMSM_restored', bbox_inches='tight', dpi=dpi)
        plt.close()

    def PlotFuzzyHard_R(self, ch, ch2, R, E_R_HARD, E_R, bottom=None):

        abscisse= np.linspace(start=i_min, stop=i_max-1, num=i_max-i_min)
        plt.figure()
        plt.plot(abscisse, R       [i_min:i_max], color='g',                      label='Simulated states')
        plt.plot(abscisse, E_R     [i_min:i_max], color='r', dashes=[3, 1, 3, 1], label='CGOFMSM')
        plt.plot(abscisse, E_R_HARD[i_min:i_max], color='b', dashes=[5, 2, 5, 2], label='CGOMSM')

        plt.xlabel('n', fontsize=fontS)
        plt.xlim(right=i_max-1, left=i_min)
        if bottom != None:
            plt.ylim(bottom=bottom)
        plt.legend()
        plt.title(ch2, fontsize=fontS)
        plt.savefig('./Result/Fuzzy/Figures/' + ch + '_X_CGOFMSM_CGOMSM_restored', bbox_inches='tight', dpi=dpi)
        plt.close()


if __name__ == '__main__':
    main()
