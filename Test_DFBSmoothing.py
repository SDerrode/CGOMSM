#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# python3 -m compileall .
# pyclean est un alias qui permet de supprimer tous les fichiers compilés (à partir du répertoire d'où est lancé cette commande)
# alias pyclean="find . -name \*.pyc -o -name \*.pyo -o -name __pycache__ -exec rm -f -r "{}" /Users/MacBook_Derrode/Documents/temp  \;"

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os, sys
import datetime
from shutil import copy2
#sys.path.insert(0, os.path.abspath("."))

from CommonFun.CommonFun import Readin_CovMeansProba, Readin_data, MSE_PK, SE_PKn, MSE_PK_MARG, SE_PKn_MARG
from CommonFun.CommonFun import Test_isCGOMSM_from_Cov, Error_Ratio, From_Cov_to_FQ, getprobamarkov, ImpressionMatLatex
from PKFResto.PKFResto   import RestorationPKF
from OFAResto.OFAResto   import RestorationOFA
from CGPMSMs.CGPMSMs     import simuCGPMSM, GetParamNearestCGO_cov, GetBackCov_CGPMSM

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

mpl.rc('text', usetex=True)
mpl.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

import time
plt.close('all')
dpi = 100 #300


def main():

    """
        Programmes pour restaurer par lissage "Double-Filtering Based Smoothing".

        :Example:

        >> python3 Test_DFBSmoothing.py ./Parameters/DFBSmoothing/Cov_CGPMSM_0.0.param 300 30 1 0

        argv[1] : Nom du fichier de paramètres
        argv[2] : Taille de l'échantillon
        argv[3] : Nombre d'expériences pour obtenir des résultats moyens
        argv[4] : Debug(3), pipelette (2), normal (1), presque muet (0)
        argv[5] : Plot graphique d'extraits (0/1)
    """

    if len(sys.argv) == 6:
        # Parameters from argv
        filenameParam   = sys.argv[1]
        N               = int(sys.argv[2])
        NbExp           = int(sys.argv[3])
        Verbose         = int(sys.argv[4])
        PlotExtraitSimu = False
        if int(sys.argv[5]) != 0 and NbExp <=3:
            PlotExtraitSimu = True
    else:
        # Default value for parameters
        filenameParam   = './Parameters/DFBSmoothing/Cov_CGPMSM.param'
        N               = 100
        NbExp           = 10
        Verbose         = 1
        PlotExtraitSimu = False

    print('Command line : python3', filenameParam, N, NbExp, Verbose, PlotExtraitSimu)

    base      = os.path.basename(filenameParam)
    paramfile = os.path.splitext(base)[0]

    np.random.seed(None)
    #np.random.seed(1) # ---> 0 0 0 0 0 --> non identiques !
    # np.random.seed(2) # ---> 1 1 1 1 1 --> non identiques !
    # np.random.seed(4), # ---> 0 0 0 1 1
    # np.random.seed(15), print('-----------------> ATTENTION : random fixé à 15!!!<------------------')

    plotMax       = min(200, N)
    # listePlotProg = [999, 1999, 2999, 3999]
    # listePlotProg = [99, 199, 299, 399]
    listePlotProg = []
    rangeN   = range(N)

    start_time = time.time()

    # MODELE GENERAL CGPMSM
    n_r, F, B, Q, Cov_CGPMSM, Mean_X, Mean_Y, JProba, IProba, TProba, SProba = Readin_CovMeansProba(filenameParam)
    n_x = np.shape(Mean_X)[1]
    n_y = np.shape(Mean_Y)[1]
    n_z = n_x + n_y
    # print('Array Cov_CGPMSM = ', Cov_CGPMSM)
    Ok = Test_isCGOMSM_from_Cov(Cov_CGPMSM, n_x, verbose=Verbose>1)
    if Ok == False:
        print('ATTENTION: Le modele original n''est pas un CGOMSM!!!')
    else:
        print('Le modele original est un CGOMSM.')

    # Majorant aveugle de l'erreur quadratique
    Tab_MajMSE_Aveugle = RestorationOFA().Calcul_MajMSE_Aveugle(Cov_CGPMSM, Mean_X, Mean_Y, IProba, N, n_x, n_z, n_r)
    # print('shape of Tab_MajMSE_Aveugle = ', np.shape(Tab_MajMSE_Aveugle))
    # print('Array Tab_MajMSE_Aveugle[:, 0, 0] = ', Tab_MajMSE_Aveugle[:, 0, 0])

    # MODELE DU FILTRE CGPMSM REVERSE
    Cov_CGPMSM_REV = GetBackCov_CGPMSM(Cov_CGPMSM, n_x=n_x)
    F_REV, Q_REV = From_Cov_to_FQ(Cov_CGPMSM_REV)
    #print('Array Cov_CGPMSM_REV = ', Cov_CGPMSM_REV)

    # Tab_MajMSE_Aveugle = RestorationOFA().Calcul_MajMSE_Aveugle(Cov_CGPMSM_REV, Mean_X, Mean_Y, IProba, N, n_x, n_z, n_r)
    # print('Array Tab_MajMSE_Aveugle[:, 0, 0] = ', Tab_MajMSE_Aveugle[:, 0, 0])
    ##### On obtient bien les mêmes résultats
    #input('pause')

    Ok = Test_isCGOMSM_from_Cov(Cov_CGPMSM_REV, n_x)
    if Ok == False:
        print('ATTENTION: Le modele reverse n''est pas un CGOMSM!!!')
        # print('Cov_CGPMSM_REV=\n', Cov_CGPMSM_REV)
        # input('pause')
    else:
        print('Le modele reverse est un CGOMSM.')

    # 1- FILTRAGE ET LISSAGE OPTIMAUX DIRECTS: find the nearest CGOMSM paremeters
    Cov_CGOMSM_DIR       = GetParamNearestCGO_cov(Cov_CGPMSM, n_x=n_x)
    F_CGO_DIR, Q_CGO_DIR = From_Cov_to_FQ(Cov_CGOMSM_DIR)
    # print('Array F_CGO_DIR = ', F_CGO_DIR)

    # 2- FILTRAGE ET LISSAGE OPTIMAUX REVERSES: find the nearest CGOMSM paremeters
    Cov_CGOMSM_REV       = GetParamNearestCGO_cov(Cov_CGPMSM_REV, n_x=n_x)
    F_CGO_REV, Q_CGO_REV = From_Cov_to_FQ(Cov_CGOMSM_REV)
    # print('Array F_CGO_REV = ', F_CGO_REV)
    # input('pause')

    if np.allclose(Cov_CGOMSM_DIR, Cov_CGOMSM_REV) == False:
        print('Les 2 filtres ne sont pas identiques')
    else:
        print('Les 2 filtres sont identiques')

    # =========== Loop for averaging results ============ #
    TabMSEn    = np.zeros(shape=(12, NbExp, n_x, N))
    TabERR     = np.zeros(shape=(4,  NbExp))
    TabCovEtat = np.zeros(shape=(n_x, n_x, 11, N))

    for exp in range(NbExp):

        # Une nouvelle simulation
        X, R, Y = simuCGPMSM(F, B, Cov_CGPMSM, Mean_X, Mean_Y, SProba, TProba, N)
        R = np.reshape(R.astype(int), newshape=(N)) # conversion de R en entier
        Y_REV = Y[:, ::-1] # Y in reverse order
        R_REV = R[::-1]    # R in reverse order

        # =========== Supervised Restoration ============ #
        # 1- RESTAURATION A SAUTS CONNUS DIRECTE
        E_X_F_CGP_SUPERV, COV_X_F_CGP_SUPERV, E_X_S_CGP_SUPERV, COV_X_S_CGP_SUPERV \
          = RestorationPKF().restore_withjump(Y, R, F, Q, Cov_CGPMSM, Mean_X, Mean_Y, Likelihood=False)
        
        # L'algo suivant utilise un smoother sans recursivité (seulement pour N petit)
        # E_X_F_CGP_SUPERV2, COV_X_F_CGP_SUPERV2, E_X_S_CGP_SUPERV2, COV_X_S_CGP_SUPERV2 \
        #     = RestorationPKF().restore_withjump2(Y, R, F, Q, Cov_CGPMSM, Mean_X, Mean_Y)
        # print('Diff E_X_S = \n', np.around(E_X_S_CGP_SUPERV - E_X_S_CGP_SUPERV2, decimals=4) )
        E_X_F_CGP_SUPERV2, COV_X_F_CGP_SUPERV2, E_X_S_CGP_SUPERV2, COV_X_S_CGP_SUPERV2 = E_X_F_CGP_SUPERV, COV_X_F_CGP_SUPERV, E_X_S_CGP_SUPERV, COV_X_S_CGP_SUPERV
        # print('Array E_X_S_CGP_SUPERV = \n', E_X_S_CGP_SUPERV)
        # print('Array E_X_S_CGP_SUPERV2 = \n', E_X_S_CGP_SUPERV2)
        # print('Array COV_X_S_CGP_SUPERV = \n', COV_X_S_CGP_SUPERV)
        # print('Array COV_X_S_CGP_SUPERV2 = \n', COV_X_S_CGP_SUPERV2)
        # input('pause')
        # if Verbose >= 2:
        #     print('X[0,0]                      =', X[0,0])
        #     print('E_X_F_CGP_SUPERV[0,0]       =', E_X_F_CGP_SUPERV[0,0])
        #     print('E_X_S_CGP_SUPERV[0,0]       =', E_X_S_CGP_SUPERV[0,0])
        #     print('X[0,0]-E_X_F_CGP_SUPERV[0,0]=', X[0,0]-E_X_F_CGP_SUPERV[0,0])
        #     print('X[0,0]-E_X_S_CGP_SUPERV[0,0]=', X[0,0]-E_X_S_CGP_SUPERV[0,0])
        #     if Verbose == 3:
        #         input('pause')

        # 2- RESTAURATION A SAUTS CONNUS REVERSE
        E_X_F_CGP_SUPERV_REV_ORDER, COV_X_F_CGP_SUPERV_REV_ORDER, E_X_S_CGP_SUPERV_REV_ORDER, COV_X_S_CGP_SUPERV_REV_ORDER \
          = RestorationPKF().restore_withjump(Y_REV, R_REV, F_REV, Q_REV, Cov_CGPMSM_REV, Mean_X, Mean_Y, Likelihood=False)
        E_X_F_CGP_SUPERV_REV   = E_X_F_CGP_SUPERV_REV_ORDER   [:,       ::-1]
        COV_X_F_CGP_SUPERV_REV = COV_X_F_CGP_SUPERV_REV_ORDER [:, :,    ::-1]
        E_X_S_CGP_SUPERV_REV   = E_X_S_CGP_SUPERV_REV_ORDER   [:,       ::-1]
        COV_X_S_CGP_SUPERV_REV = COV_X_S_CGP_SUPERV_REV_ORDER [:, :,    ::-1]

        # L'algo suivant utilise un smoother sans recursivité (seulement pour N petit)
        # E_X_F_CGP_SUPERV2_REV_ORDER, COV_X_F_CGP_SUPERV2_REV_ORDER, E_X_S_CGP_SUPERV2_REV_ORDER, COV_X_S_CGP_SUPERV2_REV_ORDER \
        #     = RestorationPKF().restore_withjump2(Y_REV, R_REV, F_REV, Q_REV, Cov_CGPMSM_REV, Mean_X, Mean_Y)
        # E_X_F_CGP_SUPERV2_REV   = E_X_F_CGP_SUPERV2_REV_ORDER   [:,       ::-1]
        # COV_X_F_CGP_SUPERV2_REV = COV_X_F_CGP_SUPERV2_REV_ORDER [:, :,    ::-1]
        # E_X_S_CGP_SUPERV2_REV   = E_X_S_CGP_SUPERV2_REV_ORDER   [:,       ::-1]
        # COV_X_S_CGP_SUPERV2_REV = COV_X_S_CGP_SUPERV2_REV_ORDER [:, :,    ::-1]
        # print('Diff E_X_S = \n', np.around(E_X_F_CGP_SUPERV_REV - E_X_F_CGP_SUPERV2_REV, decimals=4) )
        # E_X_F_CGP_SUPERV2, COV_X_F_CGP_SUPERV2, E_X_F_CGP_SUPERV2_REV, COV_X_S_CGP_SUPERV2 = E_X_F_CGP_SUPERV, COV_X_F_CGP_SUPERV, E_X_F_CGP_SUPERV_REV, COV_X_S_CGP_SUPERV
        # print('Array E_X_F_CGP_SUPERV_REV = \n', E_X_F_CGP_SUPERV_REV)
        # print('Array E_X_F_CGP_SUPERV2_REV = \n', E_X_F_CGP_SUPERV2_REV)
        # print('Array COV_X_S_CGP_SUPERV = \n', COV_X_S_CGP_SUPERV)
        # print('Array COV_X_S_CGP_SUPERV2 = \n', COV_X_S_CGP_SUPERV2)
        # print('Diff E_X_S = \n', np.around(E_X_S_CGP_SUPERV2 - E_X_S_CGP_SUPERV2_REV, decimals=4) )
        # input('pause')
        if Verbose >= 2:
            print('X[0,0]                      =', X[0,0])
            print('E_X_F_CGP_SUPERV[0,0]       =', E_X_F_CGP_SUPERV[0,0])
            print('E_X_S_CGP_SUPERV[0,0]       =', E_X_S_CGP_SUPERV[0,0])
            print('X[0,0]-E_X_F_CGP_SUPERV[0,0]=', X[0,0]-E_X_F_CGP_SUPERV[0,0])
            print('X[0,0]-E_X_S_CGP_SUPERV[0,0]=', X[0,0]-E_X_S_CGP_SUPERV[0,0])
            if Verbose == 3:
                input('pause')

        # =========== Optimal smoothing approximation unknown switches ============ #
        # 1- FILTRAGE ET LISSAGE OPTIMAL DIRECTS
        E_X_OFA_DIR, E2_X_OFA_DIR, COV_X_OFA_DIR, E_X_OSA_DIR, E2_X_OSA_DIR, COV_X_OSA_DIR, \
        E_Xn_dp_rn_y1_to_yn_DIR, E2_Xn_dp_rn_y1_to_yn_DIR, COV_Xn_dp_rn_y1_to_yn_DIR, p_rn_dp_y1_to_yN_DIR, E_R_OFA_DIR, E_R_OSA_DIR \
            = RestorationOFA().restore_withoutjumps(Y, F_CGO_DIR, Q_CGO_DIR, Cov_CGOMSM_DIR, Mean_X, Mean_Y, TProba, IProba)

        E_Xn_dp_yn_rn_DIR, E2_Xn_dp_yn_rn_DIR, COV_Xn_dp_yn_rn_DIR = \
            RestorationOFA().Compute_E_E2_COV_Xn_dp_yn_rn(Y, Cov_CGOMSM_DIR, Mean_X, Mean_Y)
        if Verbose >= 2:
            print('X[0,0]                      =', X[0,0])
            print('E_X_OFA_DIR[0,0]            =', E_X_OFA_DIR[0,0])
            print('E_X_OSA_DIR[0,0]            =', E_X_OSA_DIR[0,0])
            print('(X[0,0]-E_X_OFA_DIR[0,0])**2=', (X[0,0]-E_X_OFA_DIR[0,0])**2)
            print('(X[0,0]-E_X_OSA_DIR[0,0])**2=', (X[0,0]-E_X_OSA_DIR[0,0])**2)
            if Verbose == 3:
                input('pause')

        # 2- FILTRAGE ET LISSAGE OPTIMAL REVERSES
        E_X_OFA_REV_order, E2_X_OFA_REV_order, COV_X_OFA_REV_order, E_X_OSA_REV_order, E2_X_OSA_REV_order, COV_X_OSA_REV_order, \
        E_Xn_dp_rn_y1_to_yn_REV_order, E2_Xn_dp_rn_y1_to_yn_REV_order, COV_Xn_dp_rn_y1_to_yn_REV_order, p_rn_dp_y1_to_yN_REV_order, E_R_OFA_REV_order, E_R_OSA_REV_order \
            = RestorationOFA().restore_withoutjumps(Y_REV, F_CGO_REV, Q_CGO_REV, Cov_CGOMSM_REV, Mean_X, Mean_Y, TProba, IProba)

        # retourne pour mettre dans le bon sens
        E_X_OFA_REV               = E_X_OFA_REV_order              [:,       ::-1]
        E2_X_OFA_REV              = E2_X_OFA_REV_order             [:, :,    ::-1]
        COV_X_OFA_REV             = COV_X_OFA_REV_order            [:, :,    ::-1]
        E_X_OSA_REV               = E_X_OSA_REV_order              [:,       ::-1]
        E2_X_OSA_REV              = E2_X_OSA_REV_order             [:, :,    ::-1]
        COV_X_OSA_REV             = COV_X_OSA_REV_order            [:, :,    ::-1]
        E_Xn_dp_rn_y1_to_yn_REV   = E_Xn_dp_rn_y1_to_yn_REV_order  [:, :,    ::-1]
        E2_Xn_dp_rn_y1_to_yn_REV  = E2_Xn_dp_rn_y1_to_yn_REV_order [:, :, :, ::-1]
        COV_Xn_dp_rn_y1_to_yn_REV = COV_Xn_dp_rn_y1_to_yn_REV_order[:, :, :, ::-1]
        p_rn_dp_y1_to_yN_REV      = p_rn_dp_y1_to_yN_REV_order     [:,       ::-1]
        E_R_OFA_REV               = E_R_OFA_REV_order              [::-1]
        E_R_OSA_REV               = E_R_OSA_REV_order              [::-1]

        E_Xn_dp_yn_rn_REV_order, E2_Xn_dp_yn_rn_REV_order, COV_Xn_dp_yn_rn_REV_order = \
            RestorationOFA().Compute_E_E2_COV_Xn_dp_yn_rn(Y_REV, Cov_CGOMSM_REV, Mean_X, Mean_Y)
        E_Xn_dp_yn_rn_REV   = E_Xn_dp_yn_rn_REV_order  [:, :,    ::-1]
        E2_Xn_dp_yn_rn_REV  = E2_Xn_dp_yn_rn_REV_order [:, :, :, ::-1]
        COV_Xn_dp_yn_rn_REV = COV_Xn_dp_yn_rn_REV_order[:, :, :, ::-1]

        # 3- NEW LISSAGE BY DOUBLE FILTERING BASED SMOOTHING
        E_X_OSA_NEW, COV_X_OSA_NEW = RestorationOFA().DFBSmoothing(\
            E_Xn_dp_rn_y1_to_yn_DIR, E2_Xn_dp_rn_y1_to_yn_DIR, E_Xn_dp_rn_y1_to_yn_REV, E2_Xn_dp_rn_y1_to_yn_REV, \
            E_Xn_dp_yn_rn_DIR, E2_Xn_dp_yn_rn_DIR, p_rn_dp_y1_to_yN_DIR, n_x)

        # Save the results - Square Error
        TabMSEn[0,  exp, :, :] = SE_PKn_MARG(X, E_X_F_CGP_SUPERV)
        TabMSEn[1,  exp, :, :] = SE_PKn(X, E_X_F_CGP_SUPERV)
        TabMSEn[2,  exp, :, :] = SE_PKn(X, E_X_S_CGP_SUPERV)
        TabMSEn[3,  exp, :, :] = SE_PKn(X, E_X_F_CGP_SUPERV2)
        TabMSEn[4,  exp, :, :] = SE_PKn(X, E_X_S_CGP_SUPERV2)
        TabMSEn[5,  exp, :, :] = SE_PKn(X, E_X_OFA_DIR)
        TabMSEn[6,  exp, :, :] = SE_PKn(X, E_X_OSA_DIR)
        TabMSEn[7,  exp, :, :] = SE_PKn(X, E_X_OFA_REV)
        TabMSEn[8,  exp, :, :] = SE_PKn(X, E_X_OSA_REV)
        TabMSEn[9,  exp, :, :] = SE_PKn(X, E_X_OSA_NEW)
        TabMSEn[10, exp, :, :] = SE_PKn(X, E_X_F_CGP_SUPERV_REV)
        TabMSEn[11, exp, :, :] = SE_PKn(X, E_X_S_CGP_SUPERV_REV)

        # Erreur sur le sauts
        TabERR[0, exp] = Error_Ratio(R, E_R_OFA_DIR)
        TabERR[1, exp] = Error_Ratio(R, E_R_OSA_DIR)
        TabERR[2, exp] = Error_Ratio(R, E_R_OFA_REV)
        TabERR[3, exp] = Error_Ratio(R, E_R_OSA_REV)

         # Comparaison du nouveau et de l'ancien lissage
        TabCovEtat += SaveCovarianceEtats(rangeN, COV_X_F_CGP_SUPERV, COV_X_S_CGP_SUPERV, COV_X_F_CGP_SUPERV2, COV_X_S_CGP_SUPERV2, COV_X_F_CGP_SUPERV_REV, COV_X_S_CGP_SUPERV_REV, COV_X_OFA_DIR, COV_X_OSA_DIR, COV_X_OFA_REV, COV_X_OSA_REV, COV_X_OSA_NEW)

        if Verbose == 1 :
            if NbExp<100 :
                print("#%d/%d"%(exp+1, NbExp), end='; ', flush=True)
            elif exp%100 == 0:
                print("#%d/%d"%(exp+1, NbExp), end='; ', flush=True)

        if PlotExtraitSimu == True and (N>1):
            PlotFigures(plotMax, exp, X, E_X_OFA_DIR, E_X_OSA_DIR, R, E_R_OFA_DIR, E_R_OSA_DIR, \
                E_R_OFA_REV, E_R_OSA_REV, E_X_OFA_REV, E_X_OSA_REV, E_Xn_dp_yn_rn_DIR, E_Xn_dp_yn_rn_DIR, E_X_OSA_NEW)

    # Mean Square Error for printing results
    TabMSE_Mean = np.zeros(shape=(12, n_x))
    TabMSE_Var  = np.zeros(shape=(12, n_x))
    for i in range(12):
        TabMSE_Mean[i, :] = np.mean( np.mean(TabMSEn[i, :, :, :], axis=2), axis=0)
        TabMSE_Var [i, :] = np.var ( np.mean(TabMSEn[i, :, :, :], axis=2), axis=0)
    TabERR_Mean = np.mean(TabERR, axis=1)

    # Mean Square Error for plotting results
    TabMSEn_Mean = np.zeros(shape=(12, n_x, N))
    TabMSEn_Var  = np.zeros(shape=(12, n_x, N))
    for i in range(12):
        TabMSEn_Mean[i, :, :] = np.mean(TabMSEn[i, :, :, :], axis=0)
        TabMSEn_Var [i, :, :] = np.var (TabMSEn[i, :, :, :], axis=0)

    print("\n============ Supervised Restoration (AVERAGE of %d exp., sample size N=%d) ==========="%(NbExp, N))
    printMSE(TabMSE_Mean, TabMSE_Var, TabERR_Mean, Tab_MajMSE_Aveugle, f=sys.stdout, latex=True)
    print("\n--- %d seconds (AVERAGE) ---" % float((time.time() - start_time) / float(NbExp)))

    if N>1:
        # detect the current working directory and print it
        chemin = os.getcwd()
        date = str(datetime.datetime.now())
        chemin = chemin + '/Result/DFBSmoothing/Figures/' + date
        try:
            os.mkdir(chemin)
        except OSError:
            print ("Creation of the directory %s failed" % chemin)
        
        # Dessin des covariances des états filtrés et lissés
        figname = chemin + '/CovEtat_' + paramfile + '_' + str(NbExp)
        PlotCovariancesEtat(figname, rangeN, NbExp, TabCovEtat / NbExp)
        # Dessin de la maoyenne des Square Error en chaque point
        figname = chemin + '/MeanMSEn_' + paramfile + '_' + str(NbExp)
        PlotMSEn(figname, rangeN, NbExp, TabMSEn_Mean, Tab_MajMSE_Aveugle)

        # copie du fichier de paramètres
        copy2(filenameParam, chemin)

        # Copie des résultat de MSE
        txtName = chemin + '/MSE_result.txt'
        f = open(txtName, 'wt')
        print("\n============ Supervised Restoration (AVERAGE of %d exp., sample size N=%d) ==========="%(NbExp, N), file=f)
        printMSE(TabMSE_Mean, TabMSE_Var, TabERR_Mean, Tab_MajMSE_Aveugle, f=f, latex=True)
        print("\n--- %d seconds (AVERAGE) ---" % float((time.time() - start_time) / float(NbExp)), file=f)
        f.close()

        # copy des matrices latex
        decim = 3
        txtName = chemin + '/Matrix.txt'
        f = open(txtName, 'wt')
        print('# matrice Cov', file=f)
        ImpressionMatLatex(Cov_CGPMSM, '\\Gamma', n_r, n_z*2, decim=decim, file=f)
        print('\n# matrice A', file=f)
        ImpressionMatLatex(F, 'A', n_r, n_z, decim=decim, file=f)
        print('\n# matrice B', file=f)
        ImpressionMatLatex(B, 'B', n_r, n_z, decim=decim, file=f)
        print('\n# matrice CGOMSM Direct', file=f)
        ImpressionMatLatex(Cov_CGOMSM_DIR, '\\Gamma', n_r, n_z*2, file=f)
        print('\n# matrice CGOMSM Reverse', file=f)
        ImpressionMatLatex(Cov_CGOMSM_REV, '\\Gamma', n_r, n_z*2, file=f)
        f.close()


def SaveCovarianceEtats(rangeN, COV_X_F_CGP_SUPERV, COV_X_S_CGP_SUPERV, COV_X_F_CGP_SUPERV2, COV_X_S_CGP_SUPERV2, COV_X_F_CGP_SUPERV_REV, COV_X_S_CGP_SUPERV_REV, COV_X_OFA_DIR, COV_X_OSA_DIR, COV_X_OFA_REV, COV_X_OSA_REV, COV_X_OSA_NEW):
    n_x, N = np.shape(COV_X_OFA_DIR)[1:]
    TabCovEtat = np.zeros(shape=(n_x, n_x, 11, N))

    for n in rangeN:
        TabCovEtat[:, :, 0, n]  = COV_X_F_CGP_SUPERV     [:, :, n]
        TabCovEtat[:, :, 1, n]  = COV_X_S_CGP_SUPERV     [:, :, n]
        TabCovEtat[:, :, 2, n]  = COV_X_F_CGP_SUPERV2    [:, :, n]
        TabCovEtat[:, :, 3, n]  = COV_X_S_CGP_SUPERV2    [:, :, n]
        TabCovEtat[:, :, 4, n]  = COV_X_OFA_DIR          [:, :, n]
        TabCovEtat[:, :, 5, n]  = COV_X_OSA_DIR          [:, :, n]
        TabCovEtat[:, :, 6, n]  = COV_X_OFA_REV          [:, :, n]
        TabCovEtat[:, :, 7, n]  = COV_X_OSA_REV          [:, :, n]
        TabCovEtat[:, :, 8, n]  = COV_X_OSA_NEW          [:, :, n]
        TabCovEtat[:, :, 9, n]  = COV_X_F_CGP_SUPERV_REV [:, :, n]
        TabCovEtat[:, :, 10, n] = COV_X_S_CGP_SUPERV_REV [:, :, n]

    return TabCovEtat


def PlotMSEn(figname, rangeN, Nb, TabMSEn, Tab_MajMSE_Aveugle):

    n_x = np.shape(TabMSEn)[1]

    for dimx in range(n_x):
        fig, ax = plt.subplots(figsize=(8,5))

        # plt.plot(rangeN, TabMSEn[1, dimx, :], 'g',                      label='Recursive filt.   (Fei)')
        # plt.plot(rangeN, TabMSEn[2, dimx, :], 'y',                      label='Recursive smooth. (Fei)')
        # plt.plot(rangeN, TabMSEn[3, dimx, :], 'b', dashes=[3, 6, 3, 6], label='Recursive filt.   (Wojciech)', alpha=0.6)
        # plt.plot(rangeN, TabMSEn[4, dimx, :], 'm', dashes=[3, 6, 3, 6], label='Non-recursive smooth. (Frédéric)', alpha=0.6)

        plt.plot(rangeN, TabMSEn[1, dimx, :],             'g', label='Dir. filt. (CGP-KJ)', linestyle='--')
        plt.plot(rangeN, TabMSEn[2, dimx, :],             'g', label='Dir. smoot. (CGP-KJ)', alpha=0.5)
        plt.plot(rangeN, TabMSEn[10, dimx, :],            'y', label='Rev. filt. (CGP-KJ)', linestyle='--')
        plt.plot(rangeN, TabMSEn[11, dimx, :],            'y', label='Rev. smoot. (CGP-KJ)', alpha=0.5)
        #plt.plot(rangeN, TabMSEn[3, dimx, :],             'y', dashes=[3, 1, 3, 1], label='Filt.   (CGP-KJ)2')
        #plt.plot(rangeN, TabMSEn[4, dimx, :],             'y', dashes=[3, 1, 3, 1], label='Smooth. (CGP-KJ)2')
        plt.plot(rangeN, TabMSEn[5, dimx, :],             'b', label='Dir. filt. (CGO-UJ)', linestyle='--')
        plt.plot(rangeN, TabMSEn[6, dimx, :],             'b', label='Dir. smoot. (CGO-UJ)')
        plt.plot(rangeN, TabMSEn[7, dimx, :],             'c', label='Rev. filt. (CGO-UJ)', linestyle='--')
        plt.plot(rangeN, TabMSEn[8, dimx, :],             'c', label='Rev. smoot. (CGO-UJ)')
        plt.plot(rangeN, Tab_MajMSE_Aveugle[0, dimx, :],  'k', label='Upper bound (KJ)', alpha=0.5)
        plt.plot(rangeN, Tab_MajMSE_Aveugle[1, dimx, :],  'k', label='Upper bound (UJ)', linestyle=':', alpha=0.5)
        plt.plot(rangeN, TabMSEn[9, dimx, :],             'r', label='DFB smoot. (CGO-UJ)')

        ax.grid(True, which='major', axis='both')
        leg = plt.legend(bbox_to_anchor=(0.1, 0.1, 0.8, -0.25), loc="upper center", ncol=3, shadow=False, fancybox=False, fontsize=11)

        #title = 'CGPMSM known jumps - Individual MSE - mean of %d exp.'%(Nb)
        title = 'Individual Square Error - mean of %d exp.'%(Nb)
        if n_x>1:
            title += ' - Dim of x: %d/%d'%(dimx+1, n_x)
        plt.title(title, fontsize=16)
        fig.tight_layout()
        name = figname + '_n_x_'+str(dimx)+'.png'
        plt.savefig(name, dpi=dpi)

    plt.close('all')



def PlotCovariancesEtat(figname, rangeN, Nb, TabCovEtat):

    n_x = np.shape(TabCovEtat)[0]

    for dimx in range(n_x):
        # plot de X
        fig, ax = plt.subplots(figsize=(8,5))
        # print(fig.get_size_inches())

        # plt.plot(rangeN, TabCovEtat[dimx, dimx, 0, :], 'g',                      label='Recursive filt.   (Fei)')
        # plt.plot(rangeN, TabCovEtat[dimx, dimx, 1, :], 'y',                      label='Recursive smooth. (Fei)')
        # plt.plot(rangeN, TabCovEtat[dimx, dimx, 2, :], 'b', dashes=[3, 6, 3, 6], label='Recursive filt.   (Wojciech), alpha=0.6')
        # plt.plot(rangeN, TabCovEtat[dimx, dimx, 3, :], 'm', dashes=[3, 6, 3, 6], label='Non-recursive smooth. (Frédéric), alpha=0.6')

        plt.plot(rangeN, TabCovEtat[dimx, dimx, 0,  :], 'g', label='Dir. filt. (CGP-KJ)', linestyle='--')
        plt.plot(rangeN, TabCovEtat[dimx, dimx, 1,  :], 'g', label='Dir. smoot. (CGP-KJ)')
        plt.plot(rangeN, TabCovEtat[dimx, dimx, 9,  :], 'y', label='Rev. filt. (CGP-KJ)', linestyle='--')
        plt.plot(rangeN, TabCovEtat[dimx, dimx, 10, :], 'y', label='Rev. smoot. (CGP-KJ)')
        plt.plot(rangeN, TabCovEtat[dimx, dimx, 4,  :], 'b', label='Dir. filt. (CGO-UJ)', linestyle='--')
        plt.plot(rangeN, TabCovEtat[dimx, dimx, 5,  :], 'b', label='Dir. smoot. (CGO-UJ)')
        plt.plot(rangeN, TabCovEtat[dimx, dimx, 6,  :], 'c', label='Rev. filt. (CGO-UJ)', linestyle='--')
        plt.plot(rangeN, TabCovEtat[dimx, dimx, 7,  :], 'c', label='Rev. smoot. (CGO-UJ)')
        plt.plot(rangeN, TabCovEtat[dimx, dimx, 8,  :], 'r', label='DFB smoot. (CGO-UJ)')

        ax.grid(True, which='major', axis='both')

        #leg = plt.legend(loc="best", ncol=4, shadow=False, fancybox=False, fontsize=11)
        leg = plt.legend(bbox_to_anchor=(0.1, 0.1, 0.8, -0.25), loc="upper center", ncol=3, shadow=False, fancybox=False, fontsize=10)
        # leg.get_frame().set_alpha(0.3)

        # title = 'CGPMSM known jumps - State estimate variance - mean of %d exp.'%(Nb)
        title = 'Individual state variance - mean of %d exp.'%(Nb)
        if n_x>1:
            title += ' - Dim of x: %d/%d'%(dimx+1, n_x)
        plt.title(title, fontsize=18)
        fig.tight_layout()          # otherwise the right y-label is slightly clipped
        name = figname + '_n_x_'+str(dimx)+'.png'
        plt.savefig(name, dpi=dpi)

    plt.close('all')


def printMSE(TabMSE, TabMSE_Var, TabERR, Tab_MajMSE_Aveugle, f=sys.stdout, latex=False):

    TabMSE = np.around(TabMSE, decimals=3)
    TabERR = np.around(TabERR*100., decimals=3)

    if TabMSE_Var.any() == True:
        TabMSE_Var = np.around(TabMSE_Var, decimals=2)

        # print("  MSE (X,Y)\t\t\t: ", TabMSE[0, :], sep='')
        print("  Blind upper bound (KJ) \t: %.3f"        %(Tab_MajMSE_Aveugle[0, :, 0]),       sep='', file=f)
        print("  Blind upper bound (UJ) \t: %.3f"        %(Tab_MajMSE_Aveugle[1, :, 0]),       sep='', file=f)
        print("  Dir. Filter   (CGP - KJ)\t: %.3f (%.2f)" %(TabMSE[1, :], TabMSE_Var[1, :]), sep='', file=f)
        print("  Dir. Smoother (CGP - KJ)\t: %.3f (%.2f)" %(TabMSE[2, :], TabMSE_Var[2, :]), sep='', file=f)
        print("  Rev. Filter   (CGP - KJ)\t: %.3f (%.2f)" %(TabMSE[10, :], TabMSE_Var[10, :]), sep='', file=f)
        print("  Rev. Smoother (CGP - KJ)\t: %.3f (%.2f)" %(TabMSE[11, :], TabMSE_Var[11, :]), sep='', file=f)
        # print("  Filter   (CGP - KJ) 2\t\t: %.3f (%.2f)" %(TabMSE[3, :], TabMSE_Var[3, :]), sep='', file=f)
        # print("  Smoother (CGP - KJ) 2\t\t: %.3f (%.2f)" %(TabMSE[4, :], TabMSE_Var[4, :]), sep='', file=f)
        print("")
        print("  Filter   X, R (CGO - DIRECT)\t: %.3f (%.2f), %.1f%%" % (TabMSE[5, :], TabMSE_Var[5, :], TabERR[0]), sep='', file=f)
        print("  Smoother X, R (CGO - DIRECT)\t: %.3f (%.2f), %.1f%%" % (TabMSE[6, :], TabMSE_Var[6, :], TabERR[1]), sep='', file=f)
        print("")
        print("  Filter   X, R (CGO - REVERSE)\t: %.3f (%.2f), %.1f%%" % (TabMSE[7, :], TabMSE_Var[7, :], TabERR[2]), sep='', file=f)
        print("  Smoother X, R (CGO - REVERSE)\t: %.3f (%.2f), %.1f%%" % (TabMSE[8, :], TabMSE_Var[8, :], TabERR[3]), sep='', file=f)
        print("")
        print("  New DF based Smoother X (CGO)\t: %.3f (%.2f)" % (TabMSE[9, :], TabMSE_Var[9, :]), sep='', file=f)

        if latex==True:
            print("")
            print("    -> LATEX DIRECT  : %.3f (%.2f) & %.1f%% & %.3f (%.2f) & %.1f%%" % (TabMSE[5, :], TabMSE_Var[5, :], TabERR[0], TabMSE[6, :], TabMSE_Var[6, :], TabERR[1]), sep='', file=f)
            print("    -> LATEX REVERSE : %.3f (%.2f) & %.1f%% & %.3f (%.2f) & %.1f%%" % (TabMSE[7, :], TabMSE_Var[7, :], TabERR[2], TabMSE[8, :], TabMSE_Var[8, :], TabERR[3]), sep='', file=f)

    else:
        # print("  MSE (X,Y)\t\t\t: ", TabMSE[0, :], sep='', file=f)
        print("  Blind upper bound (KJ) \t: %.3f"        %(Tab_MajMSE_Aveugle[0, :, 0]),       sep='', file=f)
        print("  Blind upper bound (UJ) \t: %.3f"        %(Tab_MajMSE_Aveugle[1, :, 0]),       sep='', file=f)

        print("  Dir. Filter   (CGP - KJ)\t\t: %.3f" %(TabMSE[1, :]), sep='', file=f)
        print("  Dir. Smoother (CGP - KJ)\t\t: %.3f" %(TabMSE[2, :]), sep='', file=f)
        print("  Rev. Filter   (CGP - KJ)\t\t: %.3f" %(TabMSE[10, :]), sep='', file=f)
        print("  Rev. Smoother (CGP - KJ)\t\t: %.3f" %(TabMSE[11, :]), sep='', file=f)
        print("  Filter   (CGP - KJ) 2\t\t: %.3f" %(TabMSE[3, :]), sep='', file=f)
        print("  Smoother (CGP - KJ) 2\t\t: %.3f" %(TabMSE[4, :]), sep='', file=f)
        print("")
        print("  Filter   X, R (CGO - DIRECT)\t: %.3f, %.1f%%" % (TabMSE[5, :], TabERR[0]), sep='', file=f)
        print("  Smoother X, R (CGO - DIRECT)\t: %.3f, %.1f%%" % (TabMSE[6, :], TabERR[1]), sep='', file=f)
        print("")
        print("  Filter   X, R (CGO - REVERSE)\t: %.3f, %.1f%%" % (TabMSE[7, :], TabERR[2]), sep='', file=f)
        print("  Smoother X, R (CGO - REVERSE)\t: %.3f, %.1f%%" % (TabMSE[8, :], TabERR[3]), sep='', file=f)
        print("")
        print("  New DF based Smoother X (CGO)\t: %.3f" % (TabMSE[9, :]), sep='', file=f)

        if latex==True:
            print("")
            print("    -> LATEX DIRECT  : %.3f & %.1f%% & %.3f & %.1f%%" % (TabMSE[5, :], TabERR[0], TabMSE[6, :], TabERR[1]), sep='', file=f)
            print("    -> LATEX REVERSE : %.3f & %.1f%% & %.3f & %.1f%%" % (TabMSE[7, :], TabERR[2], TabMSE[8, :], TabERR[3]), sep='', file=f)


def PlotFigures(plotMax, a, X, E_X_OFA_DIR, E_X_OSA_DIR, R, E_R_OFA_DIR, E_R_OSA_DIR, E_R_OFA_REV, E_R_OSA_REV, E_X_OFA_REV, E_X_OSA_REV, E_Xn_dp_yn_rn_DIR, E_Xn_dp_yn_rn_REV, E_X_OSA_NEW):

    n_x = np.shape(X)[0]
    for x in range(n_x):

        plt.figure()
        plt.plot(X[x, 0:plotMax],           'b', label='Hidden states')
        plt.plot(E_X_OFA_DIR[x, 0:plotMax], 'r', label=r'Filtering - $\widehat{\boldsymbol{x}_1^N}$')
        plt.plot(E_X_OSA_DIR[x, 0:plotMax], 'g', label='Smoothing')
        plt.legend()
        plt.title('ORIGINAL FILTERING AND SMOOTHING (dim X = ' + str(x) + ')')
        figname = './Result/DFBSmoothing/Figures/ORIGINAL_FILTERING_AND_SMOOTHING_' + str(a) + '_n_x_' + str(x) + '.png'
        plt.savefig(figname, dpi=dpi)

        plt.figure()
        plt.plot(R[0:plotMax],           'b', label='Jumps')
        plt.plot(E_R_OFA_DIR[0:plotMax], 'r', label='Filtering jumps')
        # plt.plot(E_R_OSA_DIR[0:plotMax], 'g', label='Smoothing jumps')
        plt.legend()
        plt.title('JUMPS ORIGINAL (dim X = ' + str(x) + ')')
        figname = './Result/DFBSmoothing/Figures/UMPS_DIR_' + str(a) + '_n_x_' + str(x) + '.png'
        plt.savefig(figname, dpi=dpi)

        plt.figure()
        plt.plot(R[0:plotMax],           'b', label='Jumps')
        plt.plot(E_R_OFA_REV[0:plotMax], 'r', label='Filtering jumps')
        # plt.plot(E_R_OSA_REV[0:plotMax], 'g', label='Smoothing jumps')
        plt.legend()
        plt.title('JUMPS REVERSE (dim X = ' + str(x) + ')')
        figname = './Result/DFBSmoothing/Figures/JUMPS_REV_' + str(a) + '_n_x_' + str(x) + '.png'
        plt.savefig(figname, dpi=dpi)

        plt.figure()
        plt.plot(X[x, 0:plotMax],           'b', label='Hidden states')
        plt.plot(E_X_OFA_REV[x, 0:plotMax], 'r', label=r'Filtering - $\widetilde{\boldsymbol{x}_1^N}$')
        plt.plot(E_X_OSA_REV[x, 0:plotMax], 'g', label='Smoothing')
        plt.legend()
        plt.title('REVERSE FILTERING AND SMOOTHING (dim X = ' + str(x) + ')')
        figname = './Result/DFBSmoothing/Figures/REVERSE_FILTERING_AND_SMOOTHING_' + str(a) + '_n_x_' + str(x) + '.png'
        plt.savefig(figname, dpi=dpi)

        plt.figure()
        plt.plot(X[x, 0:plotMax],                 'b', label='Hidden states')
        plt.plot(E_Xn_dp_yn_rn_DIR[x, 0:plotMax], 'r', label=r'$\widehat{E}\left[\boldsymbol{X}_n \left| \boldsymbol{y}_n \right. \right]$')
        # plt.plot(E_Xn_dp_yn_rn_REV[x, 0:plotMax], 'g', label=r'Reverse : $\widetilde{E}\left[\boldsymbol{X}_n \left| \boldsymbol{y}_n \right. \right]$')
        plt.legend()
        plt.title('The THIRD TERM (dim X = ' + str(x) + ')')
        figname = './Result/DFBSmoothing/Figures/THIRD_TERM_' + str(a) + '_n_x_' + str(x) + '.png'
        plt.savefig(figname, dpi=dpi)

        plt.figure()
        plt.plot(X[x, 0:plotMax],               'b', label='Hidden states')
        plt.plot(E_X_OSA_NEW[x, 0:plotMax],     'g', label=r'$E\left[\boldsymbol{X}_n \left| \boldsymbol{y}_1^N \right. \right]$')
        # plt.plot(E_X_NEW_OSA_REV[x, 0:plotMax], 'r', label=r'Reverse : $\widetilde{E}\left[\boldsymbol{X}_n \left| \boldsymbol{y}_n \right. \right]$')
        plt.legend()
        plt.title('NEW SMOOTHER (dim X = ' + str(x) + ')')
        figname = './Result/DFBSmoothing/Figures/NEW_SMOOTHER_' + str(a) + '_n_x_' + str(x) + '.png'
        plt.savefig(figname, dpi=dpi)

        plt.figure()
        plt.plot(X[x, 0:plotMax],           'b', label='Hidden states')
        plt.plot(E_X_OSA_REV[x, 0:plotMax], 'g', label='Smoothing direct')
        plt.plot(E_X_OSA_NEW[x, 0:plotMax], 'r', label='New smoothing')
        plt.legend()
        # plt.title('NEW SMOOTHER (dim X = ' + str(x) + ')')
        figname = './Result/DFBSmoothing/Figures/NEW_SMOOTHER_VS_SMOOTHER_DIRECT_' + str(a) + '_n_x_' + str(x) + '.png'
        plt.savefig(figname, dpi=dpi)

        plt.figure()
        plt.plot(X[x, 0:plotMax],           'b', label='Hidden states')
        plt.plot(E_X_OSA_REV[x, 0:plotMax], 'g', label='Smoothing reverse')
        plt.plot(E_X_OSA_NEW[x, 0:plotMax], 'r', label='New smoothing')
        plt.legend()
        # plt.title('NEW SMOOTHER (dim X = ' + str(x) + ')')
        figname = './Result/DFBSmoothing/Figures/NEW_SMOOTHER_VS_SMOOTHER_REVERSE_' + str(a) + '_n_x_' + str(x) + '.png'
        plt.savefig(figname, dpi=dpi)

    plt.close('all')

if __name__ == '__main__':
    main()
