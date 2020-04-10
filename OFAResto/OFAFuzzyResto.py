#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 11:35:45 2017

@author: Fay
"""

import numpy as np
import scipy as sc
import math
import copy
import warnings
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform

from OFAResto.LoiDiscreteFuzzy import Loi1DDiscreteFuzzy, Loi2DDiscreteFuzzy
from CommonFun.CommonFun       import From_Cov_to_FQ_bis, Readin_CovMeansProba, Test_isCGOMSM_from_Cov, is_pos_def
from CGPMSMs.CGPMSMs           import GetParamNearestCGO_cov, From_Cov_to_FQ

from Fuzzy.InterFuzzy import InterBiLineaire_Matrix, InterLineaire_Vector

from Fuzzy.APrioriFuzzyLaw_Series1    import LoiAPrioriSeries1
from Fuzzy.APrioriFuzzyLaw_Series2    import LoiAPrioriSeries2
from Fuzzy.APrioriFuzzyLaw_Series2bis import LoiAPrioriSeries2bis
from Fuzzy.APrioriFuzzyLaw_Series2ter import LoiAPrioriSeries2ter
from Fuzzy.APrioriFuzzyLaw_Series3    import LoiAPrioriSeries3
from Fuzzy.APrioriFuzzyLaw_Series4    import LoiAPrioriSeries4
from Fuzzy.APrioriFuzzyLaw_Series4bis import LoiAPrioriSeries4bis

def getrnFromindrn(Rcentres, indrn):
    if indrn == 0:               return 0.
    if indrn == len(Rcentres)+1: return 1.
    return Rcentres[indrn-1]

def getindrnFromrn(STEPS, rn):
    if rn == 0.: return 0
    if rn == 1.: return STEPS+1
    return int(math.floor(rn*STEPS)+1)

                   
def loijointeAP1(rn, rnp1, proba, probaR2CondR1, Cov, yn, ynp1, Mean_Y, np1, interpolation, STEPS):

    n_z = int(np.shape(Cov)[1]/2)

    if interpolation == True:
        A_rn_rnp1, Q_rn_rnp1 = From_Cov_to_FQ_bis(InterBiLineaire_Matrix(Cov, rn, rnp1), n_z)
        moyrn   = float(InterLineaire_Vector(Mean_Y, rn))
        moyrnp1 = float(InterLineaire_Vector(Mean_Y, rnp1))
    else:
        indrn   = getindrnFromrn(STEPS, rn)
        indrnp1 = getindrnFromrn(STEPS, rnp1)
        A_rn_rnp1, Q_rn_rnp1 = From_Cov_to_FQ_bis(Cov[indrn*STEPS+indrnp1], n_z)
        moyrn   = Mean_Y[indrn]
        moyrnp1 = Mean_Y[indrnp1]

    # pdf de la gaussienne
    Gauss = norm.pdf(ynp1, loc=(yn - moyrn)*float(A_rn_rnp1[1, 1]) + moyrnp1, scale=np.sqrt(Q_rn_rnp1[1, 1])).item()

    result = proba.get(rn)*probaR2CondR1(rn, rnp1)*Gauss
    if not np.isfinite(result):
        print('proba.get(rn)=', proba.get(rn))
        print('probaR2CondR1(rn, rnp1)=', probaR2CondR1(rn, rnp1))
        print('Gauss=', Gauss)
        input('Attente loijointeAP1')

    return result


def loijointeAP2(rnp1, rn, proba, probaR2CondR1, Cov, yn, ynp1, Mean_Y, np1, interpolation, STEPS):

    n_z = int(np.shape(Cov)[1]/2)

    if interpolation == True:
        A_rn_rnp1, Q_rn_rnp1 = From_Cov_to_FQ_bis(InterBiLineaire_Matrix(Cov, rn, rnp1), n_z)
        moyrn   = float(InterLineaire_Vector(Mean_Y, rn))
        moyrnp1 = float(InterLineaire_Vector(Mean_Y, rnp1))
    else:
        indrn   = getindrnFromrn(STEPS, rn)
        inrnp1  = getindrnFromrn(STEPS, rnp1)
        A_rn_rnp1, Q_rn_rnp1 = From_Cov_to_FQ_bis(Cov[indrn*STEPS+indrnp1], n_z)
        moyrn   = Mean_Y[indrn]
        moyrnp1 = Mean_Y[indrnp1]

    # pdf de la gaussienne
    Gauss = norm.pdf(ynp1, loc=(yn - moyrn)*float(A_rn_rnp1[1, 1]) + moyrnp1, scale=np.sqrt(Q_rn_rnp1[1, 1])).item()

    result = proba.get(rnp1)*probaR2CondR1(rn, rnp1)*Gauss
    if not np.isfinite(result):
        print('proba.get(rnp1)=', proba.get(rnp1))
        print('probaR2CondR1(rn, rnp1)=', probaR2CondR1(rn, rnp1))
        print('Gauss=', Gauss)
        input('Attente loijointeAP2')

    return result

def calcMarg(r, interpolation, EPS, STEPS, LJ_AP, ProbaFB, probaR2CondR1, Cov, yn, ynp1, Mean_Y, np1):

    # Atemp, errtemp = 0., 0.
    # if STEPS != 0:
    #      Atemp, errtemp = sc.integrate.quad(func=LJ_AP, a=0.+EPS, b=1.-EPS, args=argument, epsabs=1E-3, epsrel=1E-3, limit=100)
    
    argument = (r, ProbaFB, probaR2CondR1, Cov, yn, ynp1, Mean_Y, np1, interpolation, STEPS)
    A        = 0.
    # err1     = 0.
    if STEPS != 0:
        val = 0.
        for i in range(STEPS):
            # A NE PAS REMPLACER PAR UNE SIMPLE SOMME!!!!!
            ATemp, errTemp = sc.integrate.quad(func=LJ_AP, a=i*1./STEPS+EPS, b=(i+1)*1./STEPS-EPS, args=argument, epsabs=1E-3, epsrel=1E-3, limit=100)
            A    += ATemp
            # err1 += errTemp
        # if (abs(A-Atemp)/A>0.10):
        #     print('\nA orig=', Atemp, ', err=', errtemp)
        #     print('A new =', A,     ', err=', err1)
        #     print('   pourcent', abs(A-Atemp)/A*100.)
        #     # for i, e in enumerate(argument):
        #     #     print('arg[', i, '] = ', e)
        #     absci = np.linspace(0.+EPS, 1.-EPS, 50)
        #     dessin = np.zeros(shape=(len(absci)))
        #     for i, rnp1 in enumerate(absci):
        #         dessin[i] = LJ_AP(rnp1, r, ProbaFB, probaR2CondR1, Cov, yn, ynp1, Mean_Y, np1)
        #     plt.figure()
        #     plt.plot(absci, dessin, color='b', marker='P', label='Erreur integration par sc.integrate.quad - calcMarg')
        #     plt.legend()
        #     plt.show()
        #     #plt.savefig('./Result/Fuzzy/Figures/SimuXY_SerieX_Y', bbox_inches='tight', dpi=dpi)
        #     plt.close()
        #     input('attente - calcMarg')

    A0 = LJ_AP(0., r, ProbaFB, probaR2CondR1, Cov, yn, ynp1, Mean_Y, np1, interpolation, STEPS)
    A1 = LJ_AP(1., r, ProbaFB, probaR2CondR1, Cov, yn, ynp1, Mean_Y, np1, interpolation, STEPS)
    if np.isnan(A + A0 + A1):
        print('np1= ', np1)
        print('A  = ', A)
        print('A0 = ', A0)
        print('A1 = ', A1)
        input('Nan!!')

    return A+A0+A1

def CalcE_X_np1(rnp1, proba, tab_E):
    return proba.get(rnp1) * tab_E.get(rnp1)

def IntegDouble_CalcE_Z_np1(EPS, STEPS, proba, tab_E, n_z, Rcentres):
    
    if STEPS == 0:
        return proba.get(0., 0.)*tab_E.get(0., 0.) + proba.get(0., 1.)*tab_E.get(0., 1.) + proba.get(1., 0.)*tab_E.get(1., 0.) + proba.get(1., 1.)*tab_E.get(1., 1.)
    
    pR = np.zeros(shape=(STEPS, n_z))

    #### pour r1==0.
    rn, indrn = 0., 0
    for indrnp1, rnp1 in enumerate(Rcentres):
        pR[indrnp1, :] = proba.get(rn, rnp1)* np.reshape(tab_E.get(rn, rnp1), newshape=(n_z))
    integ = np.mean(pR) + proba.get(0., 0.)*tab_E.get(0., 0.) + proba.get(0., 1.)*tab_E.get(0., 1.)

    #### pour r1==1.
    rn, indrn = 1., STEPS+1
    for indrnp1, rnp1 in enumerate(Rcentres):
        pR[indrnp1, :] = proba.get(rn, rnp1) * np.reshape(tab_E.get(rn, rnp1), newshape=(n_z))
    integ += np.mean(pR) + proba.get(1., 0.)*tab_E.get(1., 0.) + proba.get(1., 1.)*tab_E.get(1., 1.)

    # #### La surface à l'intérieur
    for indrn, rn in enumerate(Rcentres):
        for indrnp1, rnp1 in enumerate(Rcentres):
            pR[indrnp1, :] = proba.get(rn, rnp1) * np.reshape(tab_E.get(rn, rnp1), newshape=(n_z))
        integ += np.mean(pR) + proba.get(rn, 0.)*tab_E.get(rn, 0.) + proba.get(rn, 1.)*tab_E.get(rn, 1.)

    return integ


def Integ_CalcE_X_np1(EPS, STEPS, proba, tab_E, np1, Rcentres):

    argument = (proba, tab_E)
    A = 0.
    for c in Rcentres:
        A += CalcE_X_np1(c, proba, tab_E)/STEPS

    A0 = CalcE_X_np1(0., proba, tab_E)
    A1 = CalcE_X_np1(1., proba, tab_E)
    if np.isnan(A + A0 + A1):
        print('np1= ', np1)
        print('A  = ', A)
        print('A0 = ', A0)
        print('A1 = ', A1)
        input('Nan!!')

    return A+A0+A1

def CalcE_X_np1_dp_rnpun(r, rnp1, proba, tab_E, normalisationNum):
    
    result = proba.get(r, rnp1) * tab_E.get(r, rnp1) / normalisationNum
    if np.isnan(result):
        print('proba.get(r, rnp1)=', proba.get(r, rnp1))
        print('tab_E.get(r, rnp1)=', tab_E.get(r, rnp1))
        print('=',result)

    return result

def Integ_CalcE_X_np1_dp_rnpun(EPS, STEPS, Rcentres, rnp1, proba, tab_E, np1):

    normalisationNum = TestIntegMarg(proba, rnp1, EPS, STEPS, Rcentres)
    if normalisationNum == 0.:
        # print('normalisationNum=', normalisationNum)
        normalisationNum = 1. # lorsque toutes les proba sont nulles, alors on s'en fout.
        # print('rnp1=', rnp1)
        # proba.print();
        # input('normalisation num')

    argument = (rnp1, proba, tab_E, normalisationNum)

    # A        = 0.
    # err1     = 0.
    # 
    # if STEPS != 0:
    #     val = 0.
    #     for i in range(STEPS):
    #         ATemp, errTemp = sc.integrate.quad(func=CalcE_X_np1_dp_rnpun, args=argument, a=i*1./STEPS+EPS, b=(i+1)*1./STEPS-EPS, limit=200, epsabs=1E-3, epsrel=1E-3)
    #         A    += ATemp
    #         err1 += errTemp
    #     if (err1>1E-8):
    #         print('\nA=', A)
    #         print('err1=', err1)
    #         input('pause Integ_CalcE_X_np1_dp_rnpun')
        
    A = 0.
    for c in Rcentres:
        A += CalcE_X_np1_dp_rnpun(c, rnp1, proba, tab_E, normalisationNum)[0]/STEPS

    A0 = CalcE_X_np1_dp_rnpun(0., rnp1, proba, tab_E, normalisationNum)
    A1 = CalcE_X_np1_dp_rnpun(1., rnp1, proba, tab_E, normalisationNum)
    if np.isnan(A + A0 + A1):
        print('np1=', np1, ', rnp1=', rnp1)
        print('normalisationNum=', normalisationNum)
        print('A  = ', A)
        print('A0 = ', A0)
        print('A1 = ', A1)
        input('Nan!!')

    return A + A0 + A1

def IntegMarg(rn, rnp1, p_rn_d_rnpun_yun_ynpun):
    return p_rn_d_rnpun_yun_ynpun.get(rn, rnp1)

def TestIntegMarg(p_rn_d_rnpun_yun_ynpun, rnp1, EPS, STEPS, Rcentres):

    argument = (rnp1, p_rn_d_rnpun_yun_ynpun)

    # err1     = 0.
    # A        = 0.
    # if STEPS != 0:
    #     val = 0.
    #     for i in range(STEPS):
    #         ATemp, errTemp = sc.integrate.quad(limit=200, func=IntegMarg, a=i*1./STEPS+EPS, b=(i+1)*1./STEPS-EPS, epsabs=1E-3, epsrel=1E-3, args=argument)
    #         A    += ATemp
    #         err1 += errTemp
    #     if (err1>1E-8):
    #         print('\nA=', A)
    #         print('err1=', err1)
    #         for i, e in enumerate(argument):
    #             print('arg[', i, '] = ', e)
    #         absci = np.linspace(0.+EPS, 1.-EPS, 50)
    #         dessin = np.zeros(shape=(len(absci)))
    #         for i, rnp1 in enumerate(absci):
    #             dessin[i] = CalcE_X_np1(rnp1, proba, tab_E)
    #         plt.figure()
    #         plt.plot(absci, dessin, color='b', marker='P', label='Erreur integration par sc.integrate.quad - TestIntegMarg')
    #         plt.legend()
    #         plt.show()
    #         #plt.savefig('./Result/Fuzzy/Figures/SimuXY_SerieX_Y', bbox_inches='tight', dpi=dpi)
    #         plt.close()
    #         input('attente - TestIntegMarg')

    A = 0.
    for c in Rcentres:
        A += IntegMarg(c, rnp1, p_rn_d_rnpun_yun_ynpun)/STEPS

    A0 = IntegMarg(0., rnp1, p_rn_d_rnpun_yun_ynpun)
    A1 = IntegMarg(1., rnp1, p_rn_d_rnpun_yun_ynpun)
    if np.isnan(A + A0 + A1):
        print('rnp1=', rnp1)
        print('A  = ', A)
        print('A0 = ', A0)
        print('A1 = ', A1)
        input('Nan!!')

    return A + A0 + A1



#############################################################################################
class RestorationOFAFuzzy:
    """
    Optimal filtering approximation
    """
    def __init__(self, filenameParam, STEPS, FSParameters, interpolation, verbose):
        
        self.__interpolation = interpolation
        self.__verbose       = verbose

        if interpolation == False:
            self.__n_r, self.__STEPS, self.__A, self.__B, self.__Q, self.__Cov, self.__Mean_X, self.__Mean_Y = Readin_CovMeansProba(filenameParam)
        else:
            self.__STEPS = STEPS
            self.__n_r = 2
            self.__A, self.__B, self.__Q, self.__Cov, self.__Mean_X, self.__Mean_Y = Readin_CovMeansProba(filenameParam, interpolation)
            # print('n_r=', self.__n_r)
            # print('self.__STEPS=', self.__STEPS)
            # input('interpolation == True')

        self.__n_x = np.shape(self.__Mean_X)[1]
        self.__n_y = np.shape(self.__Mean_Y)[1]
        self.__n_z = self.__n_x + self.__n_y

        if Test_isCGOMSM_from_Cov(self.__Cov, self.__n_x, verbose=True) == False:
            print('self.__Cov=', self.__Cov)
            input('ATTENTION (class RestorationOFAFuzzy): ce nest pas un CGOMSM!!! --> IMPOSSIBLE')
            # self.__Cov = GetParamNearestCGO_cov(self.__Cov, n_x=self.__n_x)

        if self.__STEPS != 0:
            self.__Rcentres = np.linspace(start=1./(2.*self.__STEPS), stop=1.0-1./(2.*self.__STEPS), num=self.__STEPS, endpoint=True)
        else:
            self.__Rcentres = np.empty(shape=(0,))

        self.__FS = None
        if FSParameters[0] == '1':
            self.__FS = LoiAPrioriSeries1(alpha=float(FSParameters[1]), gamma=float(FSParameters[2]))
        elif FSParameters[0] == '2':
            self.__FS = LoiAPrioriSeries2(alpha=float(FSParameters[1]), eta=float(FSParameters[2]), delta=float(FSParameters[3]))
        elif FSParameters[0] == '2bis':
            self.__FS = LoiAPrioriSeries2bis(alpha=float(FSParameters[1]), eta=float(FSParameters[2]), delta=float(FSParameters[3]), lamb=float(FSParameters[4]))
        elif FSParameters[0] == '2ter':
            self.__FS = LoiAPrioriSeries2ter(alpha0=float(FSParameters[1]), alpha1=float(FSParameters[2]), beta=float(FSParameters[3]))
        elif FSParameters[0] == '3':
            self.__FS = LoiAPrioriSeries3(alpha=float(FSParameters[1]), delta=float(FSParameters[2]))
        elif FSParameters[0] == '4':
            self.__FS = LoiAPrioriSeries4(alpha=float(FSParameters[1]), gamma=float(FSParameters[2]), delta_d=float(FSParameters[3]), delta_u=float(FSParameters[4]))
        elif FSParameters[0] == '4bis':
            self.__FS = LoiAPrioriSeries4bis(alpha=float(FSParameters[1]), gamma=float(FSParameters[2]), delta_d=float(FSParameters[3]), delta_u=float(FSParameters[4]), lamb=float(FSParameters[5]))
        else:
            input('Impossible')

        self.__FSText = 'FS' + FSParameters[0] + '_pH_' + str(self.__FS.maxiHardJump()).replace('.','_')
        # print('self.__FSText=', self.__FSText)
        # input('fin du constructeur')


    def getParams(self):
        return self.__n_r, self.__n_x, self.__n_y, self.__n_z, np.array([self.__STEPS]), self.__A, self.__B, self.__Q, self.__Cov, self.__Mean_X, self.__Mean_Y

    def resetSTEPS(self, steps):
        self.__STEPS = steps
        if self.__STEPS != 0:
            self.__Rcentres = np.linspace(start=1./(2.*self.__STEPS), stop=1.0-1./(2.*self.__STEPS), num=self.__STEPS, endpoint=True)
        else:
            self.__Rcentres = np.empty(shape=(0,))

    def getFSText(self):
        return self.__FSText
    def getFS(self):
        return self.__FS
    def getCov(self):
        return self.__Cov
    def getMean_X(self):
        return self.__Mean_X
    def getMean_Y(self):
        return self.__Mean_Y

    def compute_jumps_forward(self, EPS, Y, Rcentres):

        n_y, N = np.shape(Y)
        
        ProbaForward = []
        p_rn_d_rnpun_yun_ynpun = []
        tab_normalis = []

        ######################
        # Initialisation
        np1 = 0
        ynp1 = Y[0, np1]
        ProbaForward.append(Loi1DDiscreteFuzzy(EPS, self.__interpolation, self.__STEPS, Rcentres))
        ProbaForward[np1].set1_1D(self.__FS.probaR, self.__Cov, ynp1, self.__Mean_Y)

        tab_normalis.append(ProbaForward[np1].Integ1D())
        ProbaForward[np1].normalisation(tab_normalis[np1])
        # ProbaForward[np1].print()
        # print('Integ=', ProbaForward[np1].Integ1D())
        # input('Attente')
        #ProbaForward[np1].plot('$p(r_n | y_1^n)$')

        ###############################
        # Boucle
        for np1 in range(1, N):
            if self.__verbose >= 2:
                print('\r         forward np1=', np1, ' sur N=', N, end='', flush = True)

            yn   = ynp1
            ynp1 = Y[0, np1]

            ProbaForward.append(Loi1DDiscreteFuzzy(EPS, self.__interpolation, self.__STEPS, Rcentres))
            ProbaForward[np1].set2_1D(calcMarg, loijointeAP1, ProbaForward[np1-1], self.__FS.probaR2CondR1, self.__Cov, yn, ynp1, self.__Mean_Y, np1)
            ProbaForward[np1].nextAfterZeros() # on evite des proba de zero

            ### AVANT normalisation de ProbaForward
            p_rn_d_rnpun_yun_ynpun.append(Loi2DDiscreteFuzzy(EPS, self.__interpolation, self.__STEPS, self.__Rcentres, self.__Mean_X, self.__Mean_Y, self.__Cov, (1, 1)))
            p_rn_d_rnpun_yun_ynpun[np1-1].set1b_2D(ProbaForward[np1], ProbaForward[np1-1], loijointeAP1, self.__FS.probaR2CondR1, yn, ynp1, np1)

            tab_normalis.append(ProbaForward[np1].Integ1D())
            ProbaForward[np1].normalisation(tab_normalis[np1])
            # ProbaForward[np1].print()
            # print('Integ=', ProbaForward[np1].Integ1D())
            # input('Attente')

        if self.__verbose >= 2:
            print(' ')

        return ProbaForward, tab_normalis, p_rn_d_rnpun_yun_ynpun


    def compute_jumps_predict(self, EPS, Rcentres, ProbaForward):

        N = len(ProbaForward)
        
        ProbaPredict = []

        # Ne sert à rien mais sert pour aligner les algo
        ProbaPredict.append(Loi2DDiscreteFuzzy(EPS, self.__interpolation, self.__STEPS, self.__Rcentres, self.__Mean_X, self.__Mean_Y, self.__Cov, (1, 1)))

        for n in range(0, N-1):
            if self.__verbose >= 2:
                print('\r         predict n=', n, ' sur N=', N, end='', flush = True)
       
            ProbaPredict.append(Loi2DDiscreteFuzzy(EPS, self.__interpolation, self.__STEPS, self.__Rcentres, self.__Mean_X, self.__Mean_Y, self.__Cov, (1, 1)))
            ProbaPredict[n].set1c_2D(self.__FS.probaR2CondR1, ProbaForward[n])
            # print('n=', n, ', Integ=', ProbaPredict[n].Integ1D())
            # input('attente')

            # normalisation in order to compensate for the fuzzy discretization
            ProbaPredict[n].normalisation(ProbaPredict[n].Integ2D())
    
        if self.__verbose >= 2:
            print(' ')

        return ProbaPredict


    def compute_jumps_backward(self, EPS, Y, tab_normalis, Rcentres):

        n_y, N = np.shape(Y)

        # attention, on va stocker dans l'ordre inverse (et à la fin on fait un reverse)
        ProbaBackward = []

        ######################
        # initialisation de beta
        n = N-1
        indice = N-n-1
        yn = Y[0, n]
        
        ProbaBackward.append(Loi1DDiscreteFuzzy(EPS, self.__interpolation, self.__STEPS, Rcentres))
        ProbaBackward[indice].setone_1D()
        #ProbaBackward[indice].print()
        #print(ProbaBackward[indice].sum())
        #input('pause')

        ###############################
        # Boucle pour backward
        for n in range(N-2, -1, -1):
            if self.__verbose >= 2:
                print('\r         backward n=', n, ' sur N=', N, end='             ', flush = True)

            indice = N-n-1
            ynp1 = yn
            yn = Y[0, n]

            ProbaBackward.append(Loi1DDiscreteFuzzy(EPS, self.__interpolation, self.__STEPS, Rcentres))
            ProbaBackward[indice].set2_1D(calcMarg, loijointeAP2, ProbaBackward[indice-1], self.__FS.probaR2CondR1, self.__Cov, yn, ynp1, self.__Mean_Y, n)
            #ProbaBackward[indice].nextAfterZeros() # on evite des proba de zero
            ProbaBackward[indice].normalisation(tab_normalis[n+1])
            #ProbaBackward[indice].plot('$p(r_{n+1} | y_1^{n+1})$')

        # reversing de la liste pour la remettre dans le bon ordre
        ProbaBackward.reverse()

        if self.__verbose >= 2:
            print(' ')
        return ProbaBackward

    def compute_jumps_smooth(self, N, EPS, ProbaForward, ProbaBackward, Rcentres):

        tab_p_rn_dp_y1_to_yN = []

        ###############################
        # Boucle sur backward
        for n in range(N):
            if self.__verbose >= 2:
                print('\r         proba lissage n=', n, ' sur N=', N, end='   ', flush = True)

            # calcul du produit forward * backward
            tab_p_rn_dp_y1_to_yN.append(Loi1DDiscreteFuzzy(EPS, self.__interpolation, self.__STEPS, Rcentres))
            tab_p_rn_dp_y1_to_yN[n].set6_1D(ProbaForward[n], ProbaBackward[n])

            # normalisation : uniquement due pour compenser des pb liés aux approximations numeriques de forward et de backward
            # Si F= 20, on voit que la normalisation n'est pas necessaire (deja la somme == 1.)
            tab_p_rn_dp_y1_to_yN[n].normalisation(tab_p_rn_dp_y1_to_yN[n].sum())
            #tab_p_rn_dp_y1_to_yN[n].print()
            #print('sum=', tab_p_rn_dp_y1_to_yN[n].sum())
            #input('Attente')

        if self.__verbose >= 2:
            print(' ')

        return tab_p_rn_dp_y1_to_yN


    def restore_Fuzzy1D(self, Y, filt=True, smooth=False, predic=False):
        """
        1 dimension should be faster than multiD
        """
        # Taille des échantillons à restaurer
        N = np.shape(Y)[1]

        # Les constantes
        EPS = 1E-15

        # les tableaux a remplir...
        E_R_n    = np.zeros((N))
        E_R_np1  = np.zeros((N))
        E_R_N    = np.zeros((N))
        E_R_n2   = np.zeros((N))
        E_R_np12 = np.zeros((N))
        E_R_N2   = np.zeros((N))
        E_X_n    = np.zeros((1, N))
        E2_X_n   = np.zeros((1, N))
        E_Z_np1  = np.zeros((self.__n_z, N))
        E2_Z_np1 = np.zeros((self.__n_z, self.__n_z, N))
        E_X_N    = np.zeros((1, N))
        E2_X_N   = np.zeros((1, N))

        ########################
        # Preparations des X
        tab_E_Znp1      = Loi2DDiscreteFuzzy(EPS, self.__interpolation, self.__STEPS, self.__Rcentres, self.__Mean_X, self.__Mean_Y, self.__Cov, (self.__n_z, 1))
        tab_VAR_Znp1    = Loi2DDiscreteFuzzy(EPS, self.__interpolation, self.__STEPS, self.__Rcentres, self.__Mean_X, self.__Mean_Y, self.__Cov, (self.__n_z, self.__n_z))
        tab_E_Xnp1_dp2  = Loi2DDiscreteFuzzy(EPS, self.__interpolation, self.__STEPS, self.__Rcentres, self.__Mean_X, self.__Mean_Y, self.__Cov, (self.__n_x, 1))
        tab_E2_Xnp1_dp2 = Loi2DDiscreteFuzzy(EPS, self.__interpolation, self.__STEPS, self.__Rcentres, self.__Mean_X, self.__Mean_Y, self.__Cov, (self.__n_x, self.__n_x))
        tab_E_Xnp1_dp1  = Loi1DDiscreteFuzzy(EPS, self.__interpolation, self.__STEPS, self.__Rcentres)
        tab_E2_Xnp1_dp1 = Loi1DDiscreteFuzzy(EPS, self.__interpolation, self.__STEPS, self.__Rcentres)

        ########################
        # Proba sauts
        tab_p_rn_dp_y1_to_yn, tab_normalis, p_rn_d_rnpun_yun_ynpun = self.compute_jumps_forward(EPS, Y, self.__Rcentres)
        if predic:
            tab_p_rnnp1_dp_y1_to_yn = self.compute_jumps_predict(EPS, self.__Rcentres, tab_p_rn_dp_y1_to_yn)

        if smooth:
            ProbaBackward        = self.compute_jumps_backward(EPS, Y, tab_normalis, self.__Rcentres)
            tab_p_rn_dp_y1_to_yN = self.compute_jumps_smooth  (N, EPS, tab_p_rn_dp_y1_to_yn, ProbaBackward, self.__Rcentres)

        np1  = 0
        ynp1 = Y[0, np1]

        #########################
        # MPM filtrage et lissage
        if filt:
            flevel_max_filt, proba_max_filt = tab_p_rn_dp_y1_to_yn[np1].fuzzyMPM_1D()
            E_R_n[np1] = flevel_max_filt
            flevel_max_filt2, proba_max_filt2 = tab_p_rn_dp_y1_to_yn[np1].fuzzyMPM2_1D()
            E_R_n2[np1] = flevel_max_filt2
        if predic:
            E_R_np1[np1]  = -1
            E_R_np12[np1] = -1
        if smooth:
            flevel_max_smoo, proba_max_smoo = tab_p_rn_dp_y1_to_yN[np1].fuzzyMPM_1D()
            E_R_N[np1] = flevel_max_smoo
            flevel_max_smoo2, proba_max_smoo2 = tab_p_rn_dp_y1_to_yN[np1].fuzzyMPM2_1D()
            E_R_N2[np1] = flevel_max_smoo2

        ######################
        # initialisation des X filtrés et lissés
        tab_E_Xnp1_dp1.set3a_1D (self.__Mean_X, self.__Mean_Y, self.__Cov, ynp1)
        tab_E2_Xnp1_dp1.set3b_1D(self.__Mean_X, self.__Mean_Y, self.__Cov, ynp1, tab_E_Xnp1_dp1)
        if self.__verbose >= 3:
            if tab_E2_Xnp1_dp1.test_VarianceNeg_1D(tab_E_Xnp1_dp1) == False:
                print('np1=', np1)
                tab_E_Xnp1_dp1.print()
                tab_E2_Xnp1_dp1.print()
                input('pause - initialisation des X filtrés et lissés')

        if filt:
            E_X_n [0, np1] = Integ_CalcE_X_np1(EPS, self.__STEPS, tab_p_rn_dp_y1_to_yn[np1], tab_E_Xnp1_dp1,  np1, self.__Rcentres)
            E2_X_n[0, np1] = Integ_CalcE_X_np1(EPS, self.__STEPS, tab_p_rn_dp_y1_to_yn[np1], tab_E2_Xnp1_dp1, np1, self.__Rcentres)
            if self.__verbose >= 3:
                if (E2_X_n[0, np1] - E_X_n[0, np1]*E_X_n[0, np1]) < 0.:
                    print('np1=', np1)
                    print('E2_X_n! np1=', np1, ', ', E2_X_n[0, np1] - E_X_n[0, np1]*E_X_n[0, np1])
                    print('E2_X_n! ', E2_X_n[0, np1], ', ', E_X_n[0, np1]*E_X_n[0, np1])
                    input('pause - if filt:')

        if predic:
            E_Z_np1 [:, np1]    = np.zeros(shape=(self.__n_z))
            E2_Z_np1[:, :, np1] = np.zeros(shape=(self.__n_z, self.__n_z))

        if smooth:
            E_X_N [0, np1] = Integ_CalcE_X_np1(EPS, self.__STEPS, tab_p_rn_dp_y1_to_yN[np1], tab_E_Xnp1_dp1,  np1, self.__Rcentres)
            E2_X_N[0, np1] = Integ_CalcE_X_np1(EPS, self.__STEPS, tab_p_rn_dp_y1_to_yN[np1], tab_E2_Xnp1_dp1, np1, self.__Rcentres)
            if self.__verbose >= 3:
                print('np1=', np1)
                if (E2_X_N[0, np1] - E_X_N[0, np1]*E_X_N[0, np1]) < 0.:
                    print('E2_X_N! np1=', np1, ', ', E2_X_N[0, np1] - E_X_N[0, np1]*E_X_N[0, np1])
                    print('E2_X_N! ', E2_X_N[0, np1], ', ', E_X_N[0, np1]*E_X_N[0, np1])
                    input('pause - if smooth:')

        ###############################
        # Boucle sur les données
        for np1 in range(1, N):
            if self.__verbose >= 2:
                print('\r         filter and/or smoother np1=', np1, ' sur N=', N, end='    ', flush = True)

            yn   = ynp1
            ynp1 = Y[0, np1]

            ##################################
            ###### PARTIE CONCERNANT LES SAUTS
            ##################################

            #########################
            # stockage de la proba des sauts pour le filtre (et pour le lisseur) la plus élevée
            if filt:
                flevel_max_filt, proba_max_filt = tab_p_rn_dp_y1_to_yn[np1].fuzzyMPM_1D()
                E_R_n[np1] = flevel_max_filt
                flevel_max_filt2, proba_max_filt2 = tab_p_rn_dp_y1_to_yn[np1].fuzzyMPM2_1D()
                E_R_n2[np1] = flevel_max_filt2
            if predic:
                flevel_max_filt, proba_max_filt = tab_p_rnnp1_dp_y1_to_yn[np1].fuzzyMPM_2D()
                E_R_np1[np1] = flevel_max_filt
                flevel_max_filt2, proba_max_filt2 = tab_p_rnnp1_dp_y1_to_yn[np1].fuzzyMPM2_2D()
                E_R_np12[np1] = flevel_max_filt2
            if smooth:
                flevel_max_smoo, proba_max_smoo = tab_p_rn_dp_y1_to_yN[np1].fuzzyMPM_1D()
                E_R_N[np1] = flevel_max_smoo
                flevel_max_smoo2, proba_max_smoo2 = tab_p_rn_dp_y1_to_yN[np1].fuzzyMPM2_1D()
                E_R_N2[np1] = flevel_max_smoo2

            ##################################
            ###### PARTIE CONCERNANT LES X / FILTER
            ##################################

            # SOUS PARTIE CONCERNANT les calculs sur X
            ##################################
            # 1. calcul de E_Znp1 = E[Zn+1 | r_n^{n+1}, ...] et covariance associée
            tab_E_Znp1.set1_2D(yn, tab_E_Xnp1_dp1)
            tab_VAR_Znp1.set33_2D(yn, tab_E_Xnp1_dp1, tab_E2_Xnp1_dp1)
            if self.__verbose >= 3:
                OK = tab_VAR_Znp1.test_VarianceNeg_2D_b()
                if OK == False:
                    print('PB Variance phase 1')
                    input('pause')

            # 2. calcul de E[X_{n+1} | r_n^{n+1}, ...] et la covariance associée
            tab_E_Xnp1_dp2.set4_2D (tab_E_Znp1,     tab_VAR_Znp1, ynp1)
            tab_E2_Xnp1_dp2.set5_2D(tab_E_Xnp1_dp2, tab_VAR_Znp1)
            if self.__verbose >= 3:
                OK = tab_E2_Xnp1_dp2.test_VarianceNeg_2D(tab_E_Xnp1_dp2)
                if OK == False:
                    print('PB Variance phase 2')
                    input('pause')

            # 3. Calcul de E[X_{n+1} | r_{n+1}, ...] et la covariance associée
            tab_E_Xnp1_dp1.set4_1D (Integ_CalcE_X_np1_dp_rnpun, p_rn_d_rnpun_yun_ynpun[np1-1], tab_E_Xnp1_dp2,  np1)
            tab_E2_Xnp1_dp1.set4_1D(Integ_CalcE_X_np1_dp_rnpun, p_rn_d_rnpun_yun_ynpun[np1-1], tab_E2_Xnp1_dp2, np1)
            if self.__verbose >= 3:
                OK = tab_E2_Xnp1_dp1.test_VarianceNeg_1D(tab_E_Xnp1_dp1)
                if OK == False:
                    print('PB Variance phase 3')
                    input('pause')

            # 4a. Calcul du filtre :  E[X_{n+1} | ...] et la covariance associée
            if filt:
                E_X_n[0, np1]  = Integ_CalcE_X_np1(EPS, self.__STEPS, tab_p_rn_dp_y1_to_yn[np1], tab_E_Xnp1_dp1,  np1, self.__Rcentres)
                E2_X_n[0, np1] = Integ_CalcE_X_np1(EPS, self.__STEPS, tab_p_rn_dp_y1_to_yn[np1], tab_E2_Xnp1_dp1, np1, self.__Rcentres)
                if self.__verbose >= 3:
                    if (E2_X_n[0, np1] - E_X_n[0, np1]*E_X_n[0, np1]) <0.:
                        print('-->PB Variance Neg FILTRAGE E2_X_n! np1=', np1, ', ', E2_X_n[0, np1] - E_X_n[0, np1]*E_X_n[0, np1])
                        input('Variance Neg!')

            # 4b. Calcul du lisseur :  E[X_{n+1} | ...] et la covariance associée
            if smooth:
                E_X_N[0, np1]  = Integ_CalcE_X_np1(EPS, self.__STEPS, tab_p_rn_dp_y1_to_yN[np1], tab_E_Xnp1_dp1,  np1, self.__Rcentres)
                E2_X_N[0, np1] = Integ_CalcE_X_np1(EPS, self.__STEPS, tab_p_rn_dp_y1_to_yN[np1], tab_E2_Xnp1_dp1, np1, self.__Rcentres)
                if self.__verbose >= 3:
                    if (E2_X_N[0, np1] - E_X_N[0, np1]*E_X_N[0, np1]) <0.:
                        print('-->PB Variance Neg LISSAGE E2_X_N! np1=', np1, ', ', E2_X_N[0, np1] - E_X_N[0, np1]*E_X_N[0, np1])
                        input('Variance Neg!')

            # 4c. Calcul du predicteur :  E[Z_{n+1} | ...] et la covariance associée
            if predic:
                E_Z_np1 [:, np1]    = np.reshape(IntegDouble_CalcE_Z_np1(EPS, self.__STEPS, tab_p_rnnp1_dp_y1_to_yn[np1], tab_E_Znp1,   self.__n_z, self.__Rcentres), newshape=(self.__n_z))
                # E2_Z_np1[:, :, np1] = np.reshape(IntegDouble_CalcE_Z_np1(EPS, self.__STEPS, tab_p_rnnp1_dp_y1_to_yn[np1], tab_VAR_Znp1, self.__n_z, self.__Rcentres)), newshape=(self.__n_z))
                # if self.__verbose >= 3:
                #     A = E2_Z_np1[:, :, np1] - np.outer(E_Z_np1[:, np1], E_Z_np1[:, np1])
                #     if is_pos_def(A) == False:
                #         print('np1=', np1)
                #         print('E2_Z_np1! np1=', np1, ', ', A)
                #         print('E2_Z_np1! ', E2_Z_np1[:, :, np1], ', ', np.outer(E_Z_np1[:, np1], E_Z_np1[:, np1]))
                #         input('pause - if predic:')


        if self.__verbose >= 2: print(' ')

        return (E_X_n, E_R_n, E_R_n2, E_X_N, E_R_N, E_R_N2, E_Z_np1, E_R_np1, E_R_np12)
