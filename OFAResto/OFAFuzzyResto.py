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

from CommonFun.CommonFun              import From_Cov_to_FQ_bis, Readin_CovMeansProba, Test_isCGOMSM_from_Cov, is_pos_def
from CGPMSMs.CGPMSMs                  import GetParamNearestCGO_cov, From_Cov_to_FQ

from OFAResto.LoiDiscreteFuzzy_TMC    import Loi1DDiscreteFuzzy_TMC, Loi2DDiscreteFuzzy_TMC
from OFAResto.TabDiscreteFuzzy        import Tab1DDiscreteFuzzy, Tab2DDiscreteFuzzy, getindrnFromrn

from Fuzzy.InterFuzzy                 import InterBiLineaire_Matrix, InterLineaire_Vector
from Fuzzy.APrioriFuzzyLaw_Series1    import LoiAPrioriSeries1
from Fuzzy.APrioriFuzzyLaw_Series2    import LoiAPrioriSeries2
from Fuzzy.APrioriFuzzyLaw_Series2bis import LoiAPrioriSeries2bis
from Fuzzy.APrioriFuzzyLaw_Series2ter import LoiAPrioriSeries2ter
from Fuzzy.APrioriFuzzyLaw_Series3    import LoiAPrioriSeries3
from Fuzzy.APrioriFuzzyLaw_Series4    import LoiAPrioriSeries4
from Fuzzy.APrioriFuzzyLaw_Series4bis import LoiAPrioriSeries4bis



def loijointeAP1(rn, rnp1, indrnp1, proba, probaR2CondR1, Cov, Mean_Y, yn, ynp1, np1, interpolation, STEPS):
    n_z = int(np.shape(Cov)[1]/2)

    if interpolation == True:
        A_rn_rnp1, Q_rn_rnp1 = From_Cov_to_FQ_bis(InterBiLineaire_Matrix(Cov, rn, rnp1), n_z)
        moyrn   = float(InterLineaire_Vector(Mean_Y, rn))
        moyrnp1 = float(InterLineaire_Vector(Mean_Y, rnp1))
    else:
        indrn   = getindrnFromrn(STEPS, rn)
        A_rn_rnp1, Q_rn_rnp1 = From_Cov_to_FQ_bis(Cov[indrn*(STEPS+2)+indrnp1], n_z)
        moyrn   = Mean_Y[indrn]
        moyrnp1 = Mean_Y[indrnp1]

    # pdf de la gaussienne
    Gauss = norm.pdf(ynp1, loc=(yn - moyrn)*float(A_rn_rnp1[1, 1]) + moyrnp1, scale=np.sqrt(Q_rn_rnp1[1, 1])).item()

    result = proba.getr(rn)*probaR2CondR1(rn, rnp1)*Gauss
    if not np.isfinite(result):
        print('proba.getr(rn)=', proba.getr(rn))
        print('probaR2CondR1(rn, rnp1)=', probaR2CondR1(rn, rnp1))
        print('Gauss=', Gauss)
        input('Attente loijointeAP1')

    return result


def loijointeAP2(rnp1, rn, indrn, proba, probaR2CondR1, Cov, Mean_Y, yn, ynp1, np1, interpolation, STEPS):

    n_z = int(np.shape(Cov)[1]/2)

    if interpolation == True:
        A_rn_rnp1, Q_rn_rnp1 = From_Cov_to_FQ_bis(InterBiLineaire_Matrix(Cov, rn, rnp1), n_z)
        moyrn   = float(InterLineaire_Vector(Mean_Y, rn))
        moyrnp1 = float(InterLineaire_Vector(Mean_Y, rnp1))
    else:
        indrnp1  = getindrnFromrn(STEPS, rnp1)
        A_rn_rnp1, Q_rn_rnp1 = From_Cov_to_FQ_bis(Cov[indrn*(STEPS+2)+indrnp1], n_z)
        moyrn   = Mean_Y[indrn]
        moyrnp1 = Mean_Y[indrnp1]

    # pdf de la gaussienne
    Gauss = norm.pdf(ynp1, loc=(yn - moyrn)*float(A_rn_rnp1[1, 1]) + moyrnp1, scale=np.sqrt(Q_rn_rnp1[1, 1])).item()

    result = proba.getr(rnp1)*probaR2CondR1(rn, rnp1)*Gauss
    if not np.isfinite(result):
        print('proba.getr(rnp1)=', proba.getr(rnp1))
        print('probaR2CondR1(rn, rnp1)=', probaR2CondR1(rn, rnp1))
        print('Gauss=', Gauss)
        input('Attente loijointeAP2')

    return result

def calcMarg(r, indr, interpolation, EPS, STEPS, LJ_AP, ProbaFB, probaR2CondR1, Cov, Mean_Y, yn, ynp1, np1):

    # Atemp, errtemp = 0., 0.
    # if STEPS != 0:
    #      Atemp, errtemp = sc.integrate.quad(func=LJ_AP, a=0.+EPS, b=1.-EPS, args=argument, epsabs=1E-3, epsrel=1E-3, limit=100)
    
    argument = (r, indr, ProbaFB, probaR2CondR1, Cov, Mean_Y, yn, ynp1, np1, interpolation, STEPS)
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

    A0 = LJ_AP(0., r, indr, ProbaFB, probaR2CondR1, Cov, Mean_Y, yn, ynp1, np1, interpolation, STEPS)
    A1 = LJ_AP(1., r, indr, ProbaFB, probaR2CondR1, Cov, Mean_Y, yn, ynp1, np1, interpolation, STEPS)
    if np.isnan(A + A0 + A1):
        print('np1= ', np1)
        print('A  = ', A)
        print('A0 = ', A0)
        print('A1 = ', A1)
        input('Nan!!')

    return A+A0+A1

def CalcE_X_np1(indrnp1, proba, tab_E):
    return proba.getindr(indrnp1) * tab_E.getindr(indrnp1)

def Integ_CalcE_X_np1(EPS, STEPS, proba, tab_E, np1, Rcentres):

    A = 0.
    for indr in range(STEPS):
        A += CalcE_X_np1(indr+1, proba, tab_E)/STEPS

    A0 = CalcE_X_np1(0,       proba, tab_E)
    A1 = CalcE_X_np1(STEPS+1, proba, tab_E)
    if np.isnan(A + A0 + A1):
        print('np1= ', np1)
        print('A  = ', A)
        print('A0 = ', A0)
        print('A1 = ', A1)
        input('Nan!!')

    # print('A  = ', A)
    # print('A0 = ', A0)
    # print('A1 = ', A1)
    # print('A+A0+A1=', A+A0+A1)
    # input('attente Integ_CalcE_X_np1')

    return A+A0+A1

def CalcE_X_np1_dp_rnpun(indr, indrnp1, proba, tab_E, normalisationNum):
    result = proba.getindr(indr, indrnp1) * tab_E.getindr(indr, indrnp1) / normalisationNum
    if np.isnan(result):
        print('proba.getindr(indr, indrnp1)=', proba.getindr(indr, indrnp1))
        print('tab_E.getindr(indr, indrnp1)=', tab_E.getindr(indr, indrnp1))
        print('=',result)

    return result

def Integ_CalcE_X_np1_dp_rnpun(EPS, STEPS, Rcentres, indrnp1, proba, tab_E, np1):

    normalisationNum = TestIntegMarg(proba, indrnp1, EPS, STEPS, Rcentres)
    if normalisationNum == 0.:
        # print('normalisationNum=', normalisationNum)
        normalisationNum = 1. # lorsque toutes les proba sont nulles, alors on s'en fout.
        # print('rnp1=', rnp1)
        # proba.print();
        # input('normalisation num')

    # argument = (indrnp1, proba, tab_E, normalisationNum)

    A = 0.
    for indrn in range(STEPS):
        A += CalcE_X_np1_dp_rnpun(indrn+1, indrnp1, proba, tab_E, normalisationNum)[0]/STEPS
    
    # print('indrnp1=', indrnp1)
    A0 = CalcE_X_np1_dp_rnpun(0,       indrnp1, proba, tab_E, normalisationNum)
    A1 = CalcE_X_np1_dp_rnpun(STEPS+1, indrnp1, proba, tab_E, normalisationNum)
    if np.isnan(A + A0 + A1):
        print('np1=', np1, ', indrnp1=', indrnp1)
        print('normalisationNum=', normalisationNum)
        print('A  = ', A)
        print('A0 = ', A0)
        print('A1 = ', A1)
        input('Nan!!')

    # print('A  = ', A)
    # print('CalcE_X_np1_dp_rnpun(1, indrnp1, proba, tab_E, normalisationNum)=', CalcE_X_np1_dp_rnpun(1, indrnp1, proba, tab_E, normalisationNum))
    # print('A0 = ', A0)
    # print('A1 = ', A1)
    # print('A+A0+A1=', A+A0+A1)
    # input('attente Integ_CalcE_X_np1')

    return A + A0 + A1

def IntegMarg(indrn, indrnp1, p_rn_d_rnpun_yun_ynpun):
    return p_rn_d_rnpun_yun_ynpun.getindr(indrn, indrnp1)

def TestIntegMarg(p_rn_d_rnpun_yun_ynpun, indrnp1, EPS, STEPS, Rcentres):

    argument = (indrnp1, p_rn_d_rnpun_yun_ynpun)

    A = 0.
    for indr in range(STEPS):
        A += IntegMarg(indr+1, indrnp1, p_rn_d_rnpun_yun_ynpun)/STEPS

    A0 = IntegMarg(0,       indrnp1, p_rn_d_rnpun_yun_ynpun)
    A1 = IntegMarg(STEPS+1, indrnp1, p_rn_d_rnpun_yun_ynpun)
    if np.isnan(A + A0 + A1):
        print('indrnp1=', indrnp1)
        print('A  = ', A)
        print('A0 = ', A0)
        print('A1 = ', A1)
        input('Nan!!')

    return A + A0 + A1


def IntegDouble_Predictor(EPS, STEPS, proba, tab_E, n_z, Rcentres):
    
    # if STEPS == 0:
    #     return proba.getindr(0, 0)*tab_E.getindr(0, 0)              + proba.getindr(0, STEPS+1)*tab_E.getindr(0, STEPS+1) + \
    #            proba.getindr(STEPS+1, 0.)*tab_E.getindr(STEPS+1, 0) + proba.getindr(STEPS+1, STEPS+1)*tab_E.getindr(STEPS+1, STEPS+1)
    
    tab_E_pondere = Tab2DDiscreteFuzzy(EPS, STEPS, False, Rcentres, dim=tab_E.getDim())
    tab_E_pondere.Prod(proba, tab_E)
    return tab_E_pondere.Integ()


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

        #self.__FSText = 'FS' + FSParameters[0] + '_pH_' + str(self.__FS.maxiHardJump()).replace('.','_')
        self.__FSText = 'FS' + FSParameters[0]
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

        N, n_y = np.shape(Y)
        
        ProbaForward = []
        p_rn_d_rnpun_yun_ynpun = []
        tab_normalis = []

        ######################
        # Initialisation
        np1 = 0
        ynp1 = Y[np1, :]
        ProbaForward.append(Loi1DDiscreteFuzzy_TMC(EPS, self.__STEPS, self.__interpolation, Rcentres))
        ProbaForward[np1].setForward_1(self.__FS.probaR, self.__Cov, ynp1, self.__Mean_Y)

        tab_normalis.append(ProbaForward[np1].Integ())
        ProbaForward[np1].normalisation(tab_normalis[np1])
        # ProbaForward[np1].print()
        # print('Integ=', ProbaForward[np1].Integ())
        # input('Attente')
        #ProbaForward[np1].plot('$p(r_n | y_1^n)$')

        ###############################
        # Boucle
        for np1 in range(1, N):
            if self.__verbose >= 2:
                print('\r         forward np1=', np1, ' sur N=', N, end='', flush=True)

            yn   = ynp1
            ynp1 = Y[np1, :]

            ProbaForward.append(Loi1DDiscreteFuzzy_TMC(EPS, self.__STEPS, self.__interpolation, Rcentres))
            ProbaForward[np1].setForwBack(calcMarg, loijointeAP1, ProbaForward[np1-1], self.__FS.probaR2CondR1, self.__Cov, self.__Mean_Y, yn, ynp1, np1)
            #ProbaForward[np1].nextAfterZeros() # on evite des proba de zero

            ### AVANT normalisation de ProbaForward
            p_rn_d_rnpun_yun_ynpun.append(Loi2DDiscreteFuzzy_TMC(EPS, self.__STEPS, self.__interpolation, self.__Rcentres))
            p_rn_d_rnpun_yun_ynpun[np1-1].CalcProbaTransAposteriori(ProbaForward[np1], ProbaForward[np1-1], loijointeAP1, self.__FS.probaR2CondR1, self.__Cov, self.__Mean_Y, yn, ynp1, np1)

            tab_normalis.append(ProbaForward[np1].Integ())
            ProbaForward[np1].normalisation(tab_normalis[np1])
            # ProbaForward[np1].print()
            # print('Integ=', ProbaForward[np1].Integ())
            # input('Attente')

        if self.__verbose >= 2:
            print(' ')

        return ProbaForward, tab_normalis, p_rn_d_rnpun_yun_ynpun


    def compute_jumps_predict(self, EPS, Rcentres, ProbaForward):

        N = len(ProbaForward)
        
        tab_p_rnnp1_dp_y1_to_yn = []
        tab_p_rnp1_dp_y1_to_yn  = []

        np1 = 0

        # Ne sert à rien mais sert sauf aligner les algo
        tab_p_rnnp1_dp_y1_to_yn.append(Loi2DDiscreteFuzzy_TMC(EPS, self.__STEPS, self.__interpolation, self.__Rcentres))
        tab_p_rnp1_dp_y1_to_yn.append (Loi1DDiscreteFuzzy_TMC(EPS, self.__STEPS, self.__interpolation, self.__Rcentres))
        tab_p_rnp1_dp_y1_to_yn[np1].setValCste(1.)
        tab_p_rnp1_dp_y1_to_yn[np1].normalisation(tab_p_rnp1_dp_y1_to_yn[np1].Integ())

        for np1 in range(1, N):
            if self.__verbose >= 2:
                print('\r         predict np1=', np1, ' sur N=', N, end='', flush=True)
       
            tab_p_rnnp1_dp_y1_to_yn.append(Loi2DDiscreteFuzzy_TMC(EPS, self.__STEPS, self.__interpolation, self.__Rcentres))
            tab_p_rnnp1_dp_y1_to_yn[np1].predicSauts(self.__FS.probaR2CondR1, ProbaForward[np1-1])
            # print('np1=', np1, ', Integ=', tab_p_rnnp1_dp_y1_to_yn[np1].Integ())

            # normalisation in order to compensate for the fuzzy discretization
            tab_p_rnnp1_dp_y1_to_yn[np1].normalisation(tab_p_rnnp1_dp_y1_to_yn[np1].Integ())

            # marginalisation
            tab_p_rnp1_dp_y1_to_yn.append(tab_p_rnnp1_dp_y1_to_yn[np1].getMarginal_r2())


        if self.__verbose >= 2:
            print(' ')

        return tab_p_rnnp1_dp_y1_to_yn, tab_p_rnp1_dp_y1_to_yn


    def compute_jumps_backward(self, EPS, Y, tab_normalis, Rcentres):

        N, n_y = np.shape(Y)

        # attention, on va stocker dans l'ordre inverse (et à la fin on fait un reverse)
        ProbaBackward = []

        ######################
        # initialisation de beta
        n = N-1
        indice = N-n-1
        yn = Y[n, :]
        
        ProbaBackward.append(Loi1DDiscreteFuzzy_TMC(EPS, self.__STEPS, self.__interpolation, Rcentres))
        ProbaBackward[indice].setValCste(1.)
        #ProbaBackward[indice].print()
        #print(ProbaBackward[indice].Integ())
        #input('pause')

        ###############################
        # Boucle pour backward
        for n in range(N-2, -1, -1):
            if self.__verbose >= 2:
                print('\r         backward n=', n, ' sur N=', N, end='             ', flush=True)

            indice = N-n-1
            ynp1 = yn
            yn = Y[n, :]

            ProbaBackward.append(Loi1DDiscreteFuzzy_TMC(EPS, self.__STEPS, self.__interpolation, Rcentres))
            ProbaBackward[indice].setForwBack(calcMarg, loijointeAP2, ProbaBackward[indice-1], self.__FS.probaR2CondR1, self.__Cov, self.__Mean_Y, yn, ynp1, n)
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
                print('\r         proba lissage n=', n, ' sur N=', N, end='   ', flush=True)

            # calcul du produit forward * backward
            tab_p_rn_dp_y1_to_yN.append(Loi1DDiscreteFuzzy_TMC(EPS, self.__STEPS, self.__interpolation, Rcentres))
            tab_p_rn_dp_y1_to_yN[n].ProductFB(ProbaForward[n], ProbaBackward[n])

            # normalisation : uniquement due pour compenser des pb liés aux approximations numeriques de forward et de backward
            # Si F= 20, on voit que la normalisation n'est pas necessaire (deja la somme == 1.)
            tab_p_rn_dp_y1_to_yN[n].normalisation(tab_p_rn_dp_y1_to_yN[n].Integ())
            #tab_p_rn_dp_y1_to_yN[n].print()
            #print('sum=', tab_p_rn_dp_y1_to_yN[n].Integ())
            #input('Attente')

        if self.__verbose >= 2:
            print(' ')

        return tab_p_rn_dp_y1_to_yN


    def restore_Fuzzy1D(self, Y, filt=True, smooth=False, predic=0):
        """
        1 dimension should be faster than multiD
        """
        # Taille des échantillons à restaurer
        N = np.shape(Y)[0]

        # Les constantes
        EPS = 1E-15

        # les tableaux a remplir...
        E_R_n     = np.zeros((N))
        E_R_np1   = np.zeros((N))
        E_R_N     = np.zeros((N))
        E_X_n     = np.zeros((N, self.__n_x))
        E2_X_n    = np.zeros((N, self.__n_x, self.__n_x))
        E_Z_np1   = np.zeros((N, self.__n_z))
        VAR_Z_np1 = np.zeros((N, self.__n_z, self.__n_z))
        E_X_N     = np.zeros((N, self.__n_x))
        E2_X_N    = np.zeros((N, self.__n_x, self.__n_x))

        ########################
        # Preparations des X
        E_Znp1_dp2   = Tab2DDiscreteFuzzy(EPS, self.__STEPS, self.__interpolation, self.__Rcentres, dim=(self.__n_z, 1))
        VAR_Znp1_dp2 = Tab2DDiscreteFuzzy(EPS, self.__STEPS, self.__interpolation, self.__Rcentres, dim=(self.__n_z, self.__n_z))
        E_Xnp1_dp2   = Tab2DDiscreteFuzzy(EPS, self.__STEPS, self.__interpolation, self.__Rcentres, dim=(self.__n_x, 1))
        E2_Xnp1_dp2  = Tab2DDiscreteFuzzy(EPS, self.__STEPS, self.__interpolation, self.__Rcentres, dim=(self.__n_x, self.__n_x))
        E_Xnp1_dp1   = Tab1DDiscreteFuzzy(EPS, self.__STEPS, self.__interpolation, self.__Rcentres, dim=(self.__n_x, 1))
        E2_Xnp1_dp1  = Tab1DDiscreteFuzzy(EPS, self.__STEPS, self.__interpolation, self.__Rcentres, dim=(self.__n_x, self.__n_x))

        ########################
        # Proba sauts
        tab_p_rn_dp_y1_to_yn, tab_normalis, p_rn_d_rnpun_yun_ynpun = self.compute_jumps_forward(EPS, Y, self.__Rcentres)
        if predic>0:
            tab_p_rnnp1_dp_y1_to_yn, tab_p_rnp1_dp_y1_to_yn = self.compute_jumps_predict(EPS, self.__Rcentres, tab_p_rn_dp_y1_to_yn)
        if smooth:
            ProbaBackward        = self.compute_jumps_backward(EPS, Y, tab_normalis, self.__Rcentres)
            tab_p_rn_dp_y1_to_yN = self.compute_jumps_smooth  (N, EPS, tab_p_rn_dp_y1_to_yn, ProbaBackward, self.__Rcentres)

        np1  = 0
        ynp1 = Y[np1, :]

        #########################
        # MPM filtrage et lissage
        if filt:
            maxi, r, indr = tab_p_rn_dp_y1_to_yn[np1].fuzzyMPM_1D()
            E_R_n[np1] = r
        if predic>0:
            maxi, r, indr = tab_p_rnp1_dp_y1_to_yn[np1].fuzzyMPM_1D()
            E_R_np1[np1] = r
        if smooth:
            maxi, r, indr = tab_p_rn_dp_y1_to_yN[np1].fuzzyMPM_1D()
            E_R_N[np1] = r

        ######################
        # initialisation des X filtrés et lissés
        E_Xnp1_dp1.set3a_1D (self.__Mean_X, self.__Mean_Y, self.__Cov, ynp1)
        E2_Xnp1_dp1.set3b_1D(self.__Mean_X, self.__Mean_Y, self.__Cov, ynp1, E_Xnp1_dp1)
        if self.__verbose >= 3:
            if E2_Xnp1_dp1.test_VarianceNeg_1D(E_Xnp1_dp1) == False:
                print('np1=', np1)
                E_Xnp1_dp1.print()
                E2_Xnp1_dp1.print()
                input('pause - initialisation des X filtrés et lissés')

        if filt:
            E_X_n [np1, :]    = Integ_CalcE_X_np1(EPS, self.__STEPS, tab_p_rn_dp_y1_to_yn[np1], E_Xnp1_dp1,  np1, self.__Rcentres)
            E2_X_n[np1, :, :] = Integ_CalcE_X_np1(EPS, self.__STEPS, tab_p_rn_dp_y1_to_yn[np1], E2_Xnp1_dp1, np1, self.__Rcentres)
            if self.__verbose >= 3:
                if (E2_X_n[np1, :, :] - np.outer(E_X_n[np1, :], E_X_n[np1, :])) < 0.:
                    print('np1=', np1)
                    print('E2_X_n! np1=', np1, ', ', E2_X_n[np1, :, :] - np.outer(E_X_n[np1, :], E_X_n[np1, :]))
                    print('E2_X_n! ', E2_X_n[np1, :, :], ', ', np.outer(E_X_n[np1, :], E_X_n[np1, :]))
                    input('pause - if filt:')

        if predic>0:
            E_Z_np1 [np1, 0:self.__n_x] = self.__Mean_X[getindrnFromrn(self.__STEPS, E_R_np1[np1]), :]
            E_Z_np1 [np1, self.__n_x:]  = self.__Mean_Y[getindrnFromrn(self.__STEPS, E_R_np1[np1]), :]
            VAR_Z_np1[np1, :, :] = 0. # pas de prediction pour le premier

        if smooth:
            E_X_N [np1, :]    = Integ_CalcE_X_np1(EPS, self.__STEPS, tab_p_rn_dp_y1_to_yN[np1], E_Xnp1_dp1,  np1, self.__Rcentres)
            E2_X_N[np1, :, :] = Integ_CalcE_X_np1(EPS, self.__STEPS, tab_p_rn_dp_y1_to_yN[np1], E2_Xnp1_dp1, np1, self.__Rcentres)
            if self.__verbose >= 3:
                print('np1=', np1)
                if (E2_X_N[np1, :, :] - np.outer(E_X_N[np1, :], E_X_N[np1, :])) < 0.:
                    print('E2_X_N! np1=', np1, ', ', E2_X_N[np1, :, :] - np.outer(E_X_N[np1, :], E_X_N[np1, :]))
                    print('E2_X_N! ', E2_X_N[np1, :, :], ', ', np.outer(E_X_N[np1, :], E_X_N[np1, :]))
                    input('pause - if smooth:')

        ###############################
        # Boucle sur les données
        for np1 in range(1, N):
            if self.__verbose >= 2:
                print('\r         filter and/or smoother np1=', np1, ' sur N=', N, end='    ', flush=True)

            yn   = ynp1
            ynp1 = Y[np1, :]

            #######################################
            ###### PARTIE CONCERNANT LES SAUTS
            #######################################

            ############################################
            # stockage de la proba des sauts pour le filtre (et pour le lisseur) la plus élevée
            if filt:
                maxi, r, indr = tab_p_rn_dp_y1_to_yn[np1].fuzzyMPM_1D()
                E_R_n[np1] = r
            if predic>0:
                maxi, r, indr = tab_p_rnp1_dp_y1_to_yn[np1].fuzzyMPM_1D()
                E_R_np1[np1] = r
            if smooth:
                maxi, r, indr = tab_p_rn_dp_y1_to_yN[np1].fuzzyMPM_1D()
                E_R_N[np1] = r

            ########################################
            ###### PARTIE CONCERNANT LES X / FILTER
            ########################################

            # SOUS PARTIE CONCERNANT les calculs sur X
            ############################################
            # 1. calcul de E_Znp1_ = E[Zn+1 | r_n^{n+1}, ...] et covariance associée
            E_Znp1_dp2.set1_2D(self.__Cov, self.__Mean_X, self.__Mean_Y, yn, E_Xnp1_dp1)
            VAR_Znp1_dp2.set33_2D(self.__Cov, yn, E_Xnp1_dp1, E2_Xnp1_dp1)
            if self.__verbose >= 3:
                OK = VAR_Znp1_dp2.test_VarianceNeg_2D_b()
                if OK == False:
                    print('PB Variance phase 1')
                    input('pause')

            # 2. calcul de E[X_{n+1} | r_n^{n+1}, ...] et la covariance associée
            E_Xnp1_dp2.set4_2D (E_Znp1_dp2, VAR_Znp1_dp2, ynp1)
            E2_Xnp1_dp2.set5_2D(E_Xnp1_dp2, VAR_Znp1_dp2)
            if self.__verbose >= 3:
                OK = E2_Xnp1_dp2.test_VarianceNeg_2D(E_Xnp1_dp2)
                if OK == False:
                    print('PB Variance phase 2')
                    input('pause')

            # 3. Calcul de E[X_{n+1} | r_{n+1}, ...] et la covariance associée
            E_Xnp1_dp1.set4_1D (Integ_CalcE_X_np1_dp_rnpun, p_rn_d_rnpun_yun_ynpun[np1-1], E_Xnp1_dp2,  np1)
            E2_Xnp1_dp1.set4_1D(Integ_CalcE_X_np1_dp_rnpun, p_rn_d_rnpun_yun_ynpun[np1-1], E2_Xnp1_dp2, np1)
            if self.__verbose >= 3:
                OK = E2_Xnp1_dp1.test_VarianceNeg_1D(E_Xnp1_dp1)
                if OK == False:
                    print('PB Variance phase 3')
                    input('pause')

            # 4a. Calcul du filtre :  E[X_{n+1} | ...] et la covariance associée
            if filt:
                E_X_n [np1, :]    = Integ_CalcE_X_np1(EPS, self.__STEPS, tab_p_rn_dp_y1_to_yn[np1], E_Xnp1_dp1,  np1, self.__Rcentres)
                E2_X_n[np1, :, :] = Integ_CalcE_X_np1(EPS, self.__STEPS, tab_p_rn_dp_y1_to_yn[np1], E2_Xnp1_dp1, np1, self.__Rcentres)
                if self.__verbose >= 3:
                    if (E2_X_n[np1, :, :] - np.outer(E_X_n[np1, :], E_X_n[np1, :])) <0.:
                        print('-->PB Variance Neg FILTRAGE E2_X_n! np1=', np1, ', ', np.outer(E_X_n[np1, :], E_X_n[np1, :]))
                        input('Variance Neg!')

            # 4b. Calcul du lisseur :  E[X_{n+1} | ...] et la covariance associée
            if smooth:
                E_X_N [np1, :]    = Integ_CalcE_X_np1(EPS, self.__STEPS, tab_p_rn_dp_y1_to_yN[np1], E_Xnp1_dp1,  np1, self.__Rcentres)
                E2_X_N[np1, :, :] = Integ_CalcE_X_np1(EPS, self.__STEPS, tab_p_rn_dp_y1_to_yN[np1], E2_Xnp1_dp1, np1, self.__Rcentres)
                if self.__verbose >= 3:
                    if (E2_X_N[np1, :, :] - np.outer(E_X_N[np1, :], E_X_N[np1, :])) <0.:
                        print('-->PB Variance Neg LISSAGE E2_X_N! np1=', np1, ', ', E2_X_N[np1, :, :] - np.outer(E_X_N[np1, :], E_X_N[np1, :]))
                        input('Variance Neg!')

            # 4c. Calcul du predicteur :  E[Z_{n+1} | ...] et la covariance associée
            if predic>0:
                A = IntegDouble_Predictor(EPS, self.__STEPS, tab_p_rnnp1_dp_y1_to_yn[np1], E_Znp1_dp2, self.__n_z, self.__Rcentres)
                E_Z_np1 [np1, :] = np.reshape(A, newshape=(self.__n_z))
                # print('E_Z_np1 [np1, :]=', E_Z_np1 [np1, :])
                # input('pause')
                VAR_Z_np1[np1, :, :] = IntegDouble_Predictor(EPS, self.__STEPS, tab_p_rnnp1_dp_y1_to_yn[np1], VAR_Znp1_dp2, self.__n_z, self.__Rcentres)
                if self.__verbose >= 3:
                    if is_pos_def(VAR_Z_np1[np1, :, :]) == False:
                        print('np1=', np1)
                        print('E_Z_np1 [np1, :] = ', E_Z_np1 [np1, :])
                        print('VAR_Z_np1[np1, :, :] = ', VAR_Z_np1[np1, :, :])
                        input('pause - if predic>0:')


        if self.__verbose >= 2: print(' ')

        return E_X_n, E_R_n, E_X_N, E_R_N, E_Z_np1, E_R_np1
