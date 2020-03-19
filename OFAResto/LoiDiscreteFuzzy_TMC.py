#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 11:35:45 2017

@author: Stephane
"""
import numpy as np
import math
import random
import matplotlib.pyplot as plt
import scipy as sc
from scipy.stats import norm
from scipy.stats import multivariate_normal

from Fuzzy.InterFuzzy import InterLineaire_Matrix, InterLineaire_Vector
from CommonFun.CommonFun import From_Cov_to_FQ_bis

def getGaussXY(M, Lambda2, P, Pi2, xn, yn, xnpun, ynpun, verb=False):
    
    #print('Lambda2=', Lambda2)
    MeanX = M[0, 0] * xn + M[0, 1] * yn + M[0, 2] * ynpun + M[0, 3]
    if verb == True:
        print('\n  MeanX  =', MeanX)
        print('  M  =', M[0, :])
        print('  Lambda2=', Lambda2)
        print('  xnpun=', xnpun)
    if Lambda2 != 0.:
        GaussX = multivariate_normal.pdf(xnpun, mean=MeanX, cov=Lambda2)
    else:
        return 0.

    #print('Pi2=', Pi2)
    MeanY = P[0, 0] * yn + P[0, 1]
    if verb == True:
        print('  MeanY  =', MeanY)
        print('  Pi2    =', Pi2)
        print('  ynpun=',   ynpun)
    if Pi2 != 0.:
        GaussY = multivariate_normal.pdf(ynpun, mean=MeanY, cov=Pi2)
    else:
        return 0.

    return GaussX * GaussY

def loiForw(rn, rnp1, probaR2CondR1):

    result = probaR2CondR1(rn, rnp1)
    if not np.isfinite(result):
        print('probaR2CondR1(rn, rnp1)=', probaR2CondR1(rn, rnp1))
        input('Attente loiForw')

    return result

def calcF(indrnp1, EPS, STEPS, Rcentres, ProbaF, FS, M, Lambda2, P, Pi2, zn, znp1):

    xn, yn, xnpun, ynpun = zn[0:1].item(), zn[1:2].item(), znp1[0:1].item(), znp1[1:2].item()

    if   indrnp1 == 0:       rnp1 = 0.
    elif indrnp1 == STEPS+1: rnp1 = 1.
    else:                    rnp1 = Rcentres[indrnp1-1]

    argument = (rnp1, FS.probaR2CondR1)
    A        = 0.
    verb     = False
    for indrn, rn in enumerate(Rcentres):
        # ON NE PEUT PAS REMPLACER PAR UNE SIMPLE SOMME POUR GAGNER DU TEMPS!!!! 
        #   Parce que probaR2CondR1 n'est pas constant sur l'instervalle d'intégration
        #   Par contre le reste l'est, donc on peut le sortir de l'intégration numérique
    
        # Cette solution est la plus rapide car plein de choses sont constantes sur le petit interval à intégrer
        # if rnp1 == 1.  and xnpun >600:
        #     verb = True
        GaussXY = getGaussXY(M[indrn+1, indrnp1], Lambda2[indrn+1, indrnp1], P[indrn+1, indrnp1], Pi2[indrn+1, indrnp1], xn, yn, xnpun, ynpun, verb)
        # if rnp1 == 1. and xnpun >600:
        #     print('  rnp1 == 1., rn = ', rn, ', GaussXY=', GaussXY)
        #     input('temp')
        if GaussXY > 0.:
            ATemp, errTemp = sc.integrate.quad(func=loiForw, a=float(indrn)/STEPS+EPS, b=float(indrn+1)/STEPS-EPS, args=argument, epsabs=1E-2, epsrel=1E-2, limit=50)
            A += ATemp * GaussXY * ProbaF.get(rn)
            # input('temp')
    
    rn, indrn = 0., 0
    GaussXY   = getGaussXY(M[indrn, indrnp1], Lambda2[indrn, indrnp1], P[indrn, indrnp1], Pi2[indrn, indrnp1], xn, yn, xnpun, ynpun)
    A0        = FS.probaR2CondR1(rn, rnp1) * GaussXY * ProbaF.get(rn)
    
    rn, indrn = 1., STEPS+1
    # if rnp1 == 1. and xnpun >600:
    #     verb = True
    GaussXY   = getGaussXY(M[indrn, indrnp1], Lambda2[indrn, indrnp1], P[indrn, indrnp1], Pi2[indrn, indrnp1], xn, yn, xnpun, ynpun, verb)
    A1        = FS.probaR2CondR1(rn, rnp1) * GaussXY * ProbaF.get(rn)
    # if rnp1 == 1. and xnpun >600:
    #     print('  rnp1 == 1., rn = ', rn, ', GaussXY=', GaussXY)
    #     print('FS.probaR2CondR1(rn, rnp1) =', FS.probaR2CondR1(rn, rnp1))
    #     print('ProbaF.get(rn)             =', ProbaF.get(rn))
    #     print('ProbaF.Integ()             =', ProbaF.Integ())
    #     ProbaF.print()
    #     print('A1                         =', A1)
    #     print('A + A0 + A1                =', A + A0 + A1)
    #     input('temp')
    
    if not np.isfinite(A + A0 + A1):
        print('  A  = ', A)
        print('  A0 = ', A0)
        print('  A1 = ', A1)
        input('Nan!!')

    return A+A0+A1


def loiBackw(rnp1, rn, probaR2CondR1):

    result = probaR2CondR1(rn, rnp1)
    if not np.isfinite(result):
        print('probaR2CondR1(rn, rnp1)=', probaR2CondR1(rn, rnp1))
        input('Attente loiForw')

    return result


def calcB(indrn, EPS, STEPS, Rcentres, ProbaB, FS, M, Lambda2, P, Pi2, zn, znp1):

    xn, yn, xnpun, ynpun = zn[0:1], zn[1:2], znp1[0:1], znp1[1:2]

    if   indrn == 0:       rn = 0.
    elif indrn == STEPS+1: rn = 1.
    else:                  rn = Rcentres[indrn-1]

    argument = (rn, FS.probaR2CondR1)
    A        = 0.
   
    for indrnp1, rnp1 in enumerate(Rcentres):
        # ON NE PEUT PAS REMPLACER PAR UNE SIMPLE SOMME POUR GAGNER DU TEMPS!!!! 
        #   Parce que probaR2CondR1 n'est pas constant sur l'instervalle d'intégration
        #   Par contre le reste l'est, donc on peut le sortir de l'intégration numérique
    
        # Cette solution est la plus rapide car plein de choses sont constantes sur le petit interval à intégrer
        GaussXY = getGaussXY(M[indrn, indrnp1+1], Lambda2[indrn, indrnp1+1], P[indrn, indrnp1+1], Pi2[indrn, indrnp1+1], xn, yn, xnpun, ynpun)
        if GaussXY > 0.:
            ATemp, errTemp = sc.integrate.quad(func=loiBackw, a=float(indrnp1)/STEPS+EPS, b=float(indrnp1+1)/STEPS-EPS, args=argument, epsabs=1E-2, epsrel=1E-2, limit=50)
            A += ATemp * GaussXY * ProbaB.get(rnp1)
            #input('temp')
    
    rnp1, indrnp1 = 0., 0
    GaussXY       = getGaussXY(M[indrn, indrnp1], Lambda2[indrn, indrnp1], P[indrn, indrnp1], Pi2[indrn, indrnp1], xn, yn, xnpun, ynpun)
    A0            = FS.probaR2CondR1(rn, rnp1) * GaussXY * ProbaB.get(rnp1)
    
    rnp1, indrnp1 = 1., STEPS+1
    GaussXY       = getGaussXY(M[indrn, indrnp1], Lambda2[indrn, indrnp1], P[indrn, indrnp1], Pi2[indrn, indrnp1], xn, yn, xnpun, ynpun)
    A1            = FS.probaR2CondR1(rn, rnp1) * GaussXY * ProbaB.get(rnp1)

    if not np.isfinite(A + A0 + A1):
        print('  A  = ', A)
        print('  A0 = ', A0)
        print('  A1 = ', A1)
        input('Nan!!')

    return A+A0+A1


############################################################################################################
class Loi2DDiscreteFuzzy_TMC():

    def __init__(self, EPS, STEPS, Rcentres):
        self.__EPS      = EPS
        self.__STEPS    = STEPS
        self.__Rcentres = Rcentres
        if len(Rcentres) != self.__STEPS:
            input('PB constructeur Loi1DDiscreteFuzzy_TMC')

        self.__p00 = 0.
        self.__p10 = 0.
        self.__p01 = 0.
        self.__p11 = 0.
        
        if self.__STEPS != 0:
            self.__p00_01 = np.zeros(shape=(self.__STEPS))
            self.__p01_11 = np.zeros(shape=(self.__STEPS))
            self.__p11_10 = np.zeros(shape=(self.__STEPS))
            self.__p10_00 = np.zeros(shape=(self.__STEPS))
        else:
            self.__p00_01 = np.empty(shape=(0))
            self.__p01_11 = np.empty(shape=(0))
            self.__p11_10 = np.empty(shape=(0))
            self.__p10_00 = np.empty(shape=(0))

        if self.__STEPS != 0:
            self.__p = np.zeros(shape=(self.__STEPS, self.__STEPS))
        else:
            self.__p = np.empty(shape=(0))

        # print(np.shape(self.__p00))
        # print(np.shape(self.__p00_01))
        # print(np.shape(self.__p))
        # input('attente 2D')

    def get(self, r1, r2):
        
        if r1==1.:
            if r2==1.:
                return self.__p11
            elif r2>0.:
                indice = math.floor(r2*self.__STEPS)
                return self.__p01_11[indice]
            else:
                return self.__p01

        if r1==0.:
            if r2==1.:
                return self.__p10
            elif r2>0.:
                indice = math.floor(r2*self.__STEPS)
                return self.__p10_00[indice]
            else:
                return self.__p00

        if r2==1.:
            indr1 = math.floor(r1*self.__STEPS)
            return self.__p11_10[indr1]
        elif r2>0.:
            indr1 = math.floor(r1*self.__STEPS)
            indr2 = math.floor(r2*self.__STEPS)
            return self.__p[indr1, indr2]
        else:
            indr1 = math.floor(r1*self.__STEPS)
            return self.__p00_01[indr1]


    def CalcPsi(self, PForward_n, PBackward_np1, FS, M, Lambda2, P, Pi2, zn, znp1):

        xn, yn, xnpun, ynpun = zn[0:1], zn[1:2], znp1[0:1], znp1[1:2]

        # Pour les masses
        rn, rnp1       = 0., 0.
        indrn, indrnp1 = 0, 0
        GaussXY        = getGaussXY(M[indrn, indrnp1], Lambda2[indrn, indrnp1], P[indrn, indrnp1], Pi2[indrn, indrnp1], xn, yn, xnpun, ynpun)
        self.__p00     = PForward_n.get(rn) * PBackward_np1.get(rnp1) * GaussXY * FS.probaR2CondR1(rn, rnp1)
        
        rn, rnp1       = 1., 0.
        indrn, indrnp1 = self.__STEPS+1, 0
        GaussXY        = getGaussXY(M[indrn, indrnp1], Lambda2[indrn, indrnp1], P[indrn, indrnp1], Pi2[indrn, indrnp1], xn, yn, xnpun, ynpun)
        self.__p01     = PForward_n.get(rn) * PBackward_np1.get(rnp1) * GaussXY * FS.probaR2CondR1(rn, rnp1)

        rn, rnp1       = 1., 1.
        indrn, indrnp1 = self.__STEPS+1, self.__STEPS+1
        GaussXY        = getGaussXY(M[indrn, indrnp1], Lambda2[indrn, indrnp1], P[indrn, indrnp1], Pi2[indrn, indrnp1], xn, yn, xnpun, ynpun)
        self.__p11     = PForward_n.get(rn) * PBackward_np1.get(rnp1) * GaussXY * FS.probaR2CondR1(rn, rnp1)
        
        rn, rnp1       = 0., 1.
        indrn, indrnp1 = 0, self.__STEPS+1
        GaussXY        = getGaussXY(M[indrn, indrnp1], Lambda2[indrn, indrnp1], P[indrn, indrnp1], Pi2[indrn, indrnp1], xn, yn, xnpun, ynpun)
        self.__p10     = PForward_n.get(rn) * PBackward_np1.get(rnp1) * GaussXY * FS.probaR2CondR1(rn, rnp1)

        
        if self.__STEPS != 0:

            # Pour les arètes
            for ind, r in enumerate(self.__Rcentres):
                
                # self.__p00_01
                rnp1    = 0.
                indrnp1 = 0
                GaussXY = getGaussXY(M[ind+1, indrnp1], Lambda2[ind+1, indrnp1], P[ind+1, indrnp1], Pi2[ind+1, indrnp1], xn, yn, xnpun, ynpun)
                self.__p00_01[ind] = PForward_n.get(r) * PBackward_np1.get(rnp1) * GaussXY * FS.probaR2CondR1(r, rnp1)

                # self.__p01_11
                rn      = 1.
                indrn   = self.__STEPS+1
                GaussXY = getGaussXY(M[indrn, ind+1], Lambda2[indrn, ind+1], P[indrn, ind+1], Pi2[indrn, ind+1], xn, yn, xnpun, ynpun)
                self.__p01_11[ind] = PForward_n.get(rn) * PBackward_np1.get(r) * GaussXY * FS.probaR2CondR1(rn, r)

                # self.__p11_10
                rnp1    = 1.
                indrnp1 = self.__STEPS+1
                GaussXY = getGaussXY(M[ind+1, indrnp1], Lambda2[ind+1, indrnp1], P[ind+1, indrnp1], Pi2[ind+1, indrnp1], xn, yn, xnpun, ynpun)
                self.__p11_10[ind] = PForward_n.get(r) * PBackward_np1.get(rnp1) * GaussXY * FS.probaR2CondR1(r, rnp1)

                # self.__p10_00
                rn      = 0.
                indrn   = 0
                GaussXY = getGaussXY(M[indrn, ind+1], Lambda2[indrn, ind+1], P[indrn, ind+1], Pi2[indrn, ind+1], xn, yn, xnpun, ynpun)
                self.__p10_00[ind] = PForward_n.get(rn) * PBackward_np1.get(r) * GaussXY * FS.probaR2CondR1(rn, r)

            # Pour l'intérieur
            for indrn, rn in enumerate(self.__Rcentres):
                for indrnp1, rnp1 in enumerate(self.__Rcentres):
                    GaussXY = getGaussXY(M[indrn+1, indrnp1+1], Lambda2[indrn+1, indrnp1+1], P[indrn+1, indrnp1+1], Pi2[indrn+1, indrnp1+1], xn, yn, xnpun, ynpun)
                    self.__p[indrn, indrnp1] = PForward_n.get(rn) * PBackward_np1.get(rnp1) * GaussXY * FS.probaR2CondR1(rn, rnp1)


    def Integ(self):

        integ = self.__p00 + self.__p10 + self.__p01 + self.__p11
        if self.__STEPS == 0:
            return integ

        #### pour r1==0.
        integ = np.mean(self.__p10_00) + self.__p00 + self.__p10

        #### pour r1==1.
        integ += np.mean(self.__p01_11) + self.__p01 + self.__p11

        #### La surface à l'intérieur
        pR = np.ndarray(shape=(self.__STEPS))
        for j in range(self.__STEPS):
            pR[j] = np.mean(self.__p[j, :]) + self.__p00_01[j] + self.__p11_10[j]
        integ += np.mean(pR)

        return integ


    def normalisation(self, norm):

        if norm != 0.:
            self.__p00    /= norm
            self.__p10    /= norm
            self.__p01    /= norm
            self.__p11    /= norm    
            
            self.__p00_01 /= norm
            self.__p01_11 /= norm
            self.__p11_10 /= norm
            self.__p10_00 /= norm 
            
            self.__p      /= norm
        else:
            input('pb if norm == 0.')


    def print(self):

        print("Les coins:")
        print(self.__p00, self.__p01, self.__p11, self.__p01)

        print("Les bords:")
        for i, rnp1 in enumerate(self.__Rcentres):
            print(self.__p00_01[i], end=' ')
        print("\n")
        for i, rnp1 in enumerate(self.__Rcentres):
            print(self.__p01_11[i], end=' ')
        print("\n")
        for i, rnp1 in enumerate(self.__Rcentres):
            print(self.__p11_10[i], end=' ')
        print("\n")
        for i, rnp1 in enumerate(self.__Rcentres):
            print(self.__p10_00[i], end=' ')

        print("\nLe coeur:")
        for indrn, rn in enumerate(self.__Rcentres):
            for indrnp1, rnp1 in enumerate(self.__Rcentres):
                print(self.__p[indrn, indrnp1], end=' ')
            print(" ")


############################################################################################################
class Loi1DDiscreteFuzzy_TMC():

    def __init__(self, EPS, STEPS, Rcentres):
        self.__EPS      = EPS
        self.__STEPS    = STEPS
        self.__Rcentres = Rcentres
        if len(Rcentres) != self.__STEPS:
            input('PB constructeur Loi1DDiscreteFuzzy_TMC')

        self.__p0 = 0.
        self.__p1 = 0.
        if self.__STEPS != 0:
            self.__p01 = np.zeros(shape=(self.__STEPS))
        else:
            self.__p01 = np.empty(shape=(0,))
    
    def ProductFB(self, loi1, loi2):
        for i in range(self.__STEPS):
            self.__p01[i] = loi1.__p01[i] * loi2.__p01[i]
        self.__p0 = loi1.__p0 * loi2.__p0
        self.__p1 = loi1.__p1 * loi2.__p1

    def getRcentre(self):
        return self.__Rcentres

    def getSTEPS(self):
        return self.__STEPS

    def getEPS(self):
        return self.__EPS

    def get(self, r):
        if r==1.:
            return self.__p1
        elif r>0.:
            indice = math.floor(r*self.__STEPS)
            return self.__p01[indice]
        else:
            return self.__p0

    def set(self, r, val):
        if r==1.:
            self.__p1=val
        elif r>0.:
            indice = math.floor(r*self.__STEPS)
            self.__p01[indice]=val
        else:
            self.__p0 = val

    def print(self):
        print('__p0 = ', self.__p0)
        for i, rnp1 in enumerate(self.__Rcentres):
            print('  __p01[',rnp1, ']=', self.__p01[i])
        print('__p1 = ', self.__p1)

    def test(self, E2, E):
        A = E2 - np.dot(E, np.transpose(E))
        return A


    def CalcForw1(self, FS, z, MeanCovFuzzy):

        alpha, ind = 0., 0 # le premier
        self.__p0 = FS.probaR(alpha) * multivariate_normal.pdf(z, mean=MeanCovFuzzy.getMean(ind), cov=MeanCovFuzzy.getCov(ind))

        alpha, ind = 1., self.__STEPS+1 # le dernier
        self.__p1   = FS.probaR(alpha) * multivariate_normal.pdf(z, mean=MeanCovFuzzy.getMean(ind), cov=MeanCovFuzzy.getCov(ind))

        for ind, alpha in enumerate(self.__Rcentres):
            self.__p01[ind] = FS.probaR(alpha) * multivariate_normal.pdf(z, mean=MeanCovFuzzy.getMean(ind+1), cov=MeanCovFuzzy.getCov(ind+1))
        
        self.normalisation(self.Integ())
        # self.print()
        # print(self.Integ())
        # input('pause - set1_1D')

    def setValCste(self, val):
        for i in range(self.__STEPS):
            self.__p01[i] = val
        self.__p0 = val
        self.__p1 = val

    def CalcForB(self, FctCalculForB, probaForB, FS, M, Lambda2, P, Pi2, zn, znp1):

        # les proba sont renvoyées non normalisées
        self.__p0 = FctCalculForB(0, self.__EPS, self.__STEPS, self.__Rcentres, probaForB, FS, M, Lambda2, P, Pi2, zn, znp1)
        for i, r in enumerate(self.__Rcentres):
            self.__p01[i] = FctCalculForB(i+1, self.__EPS, self.__STEPS, self.__Rcentres, probaForB, FS, M, Lambda2, P, Pi2, zn, znp1)
        self.__p1 = FctCalculForB(self.__STEPS+1, self.__EPS, self.__STEPS, self.__Rcentres, probaForB, FS, M, Lambda2, P, Pi2, zn, znp1)
        # if znp1[0:1] > 600:
        #     self.print()
        #     input('ATTENTE')

    def nextAfterZeros(self):
        if self.__p0 < 1e-300:
            self.__p0 = 1e-300 #np.nextafter(0, 1)*10

        if self.__p1 < 1e-300:
            self.__p1 = 1e-300 #np.nextafter(0, 1)*10

        for i in range(self.__STEPS):
            if self.__p01[i] < 1e-300:
                self.__p01[i] = 1e-300 #np.nextafter(0, 1)*10

    def normalisation(self, norm):
        if norm != 0.:
            self.__p01 /= norm
            self.__p0  /= norm
            self.__p1  /= norm
        else:
            input('ATTENTION : norm == 0.')

    def CalcCond(self, rn, ProbaGamma_n_rn, ProbaPsi_n):

        #print('  ProbaGamma_n_rn=', ProbaGamma_n_rn)
        self.__p0 = ProbaPsi_n.get(rn, 0.) / ProbaGamma_n_rn
        # print('  ProbaPsi_n.get(rn, 0.)=', ProbaPsi_n.get(rn, 0.))
        # print('ProbaPsi_n.get(rn, 0.)=', ProbaPsi_n.get(rn, 0.))
        # print('self.__p0=', self.__p0)
        for ind_rnp1, rnp1 in enumerate(self.__Rcentres):
            self.__p01[ind_rnp1] = ProbaPsi_n.get(rn, rnp1) / ProbaGamma_n_rn
        self.__p1 = ProbaPsi_n.get(rn, 1.) / ProbaGamma_n_rn
        #self.print()

        # This integral is converging to 1 if F growth. In case F small (i.e<10) it is better to normalise
        integ = self.Integ()
        # self.print()
        # print('integ', integ)

        # test if all 0 
        if self.TestIsAllZero():
            # print('integ', integ)
            # print('integ ProbaGamma_n', ProbaGamma_n.Integ())
            # print('integ ProbaPsi_n', ProbaPsi_n.Integ())
            # self.print()
            # ProbaGamma_n.print()
            # ProbaPsi_n.print()
            # print('p0 : ', ProbaPsi_n.get(rn, 0.), ProbaGamma_n_rn)
            # print('p1 : ', ProbaPsi_n.get(rn, 1.), ProbaGamma_n_rn)
            # input('pause')
            input('\nWarnng :all the proba cond is 0. when rn=' + str(rn))
        else:
            self.normalisation(integ)
            if abs(1.-self.Integ()) > 1E-3:
                print('  Integ R2 Cond R1:', self.Integ())
                input('PB PB PB proba cond')


    def TestIsAllZero(self):

        if self.__p0 != 0.: return False
        if self.__p1 != 0.: return False
        for ind in range(self.__STEPS):
            if self.__p01[ind] != 0.: return False
        return True

    def getSample(self):

        proba = np.zeros(shape=(3))
        proba[0] = self.__p0
        proba[1] = self.__p1
        proba[2] = 1. - (proba[0]+proba[1])
        typeSample = random.choices(population=[0, self.__STEPS+1, 2], weights=proba)[0]
        if typeSample != 2:
            r1 = int(typeSample)
            #print('tirage saut dur')
        else: # it is fuzzy
            probaF = np.zeros(shape=(self.__STEPS))
            probaF = self.__p01 / np.mean(self.__p01)
            #print(probaF, np.trapz(y=probaF, x=self.__Rcentres))
            r1 = random.choices(population=list(range(1,self.__STEPS+1)), weights=probaF)[0]
            #print('tirage saut flou')
        #print(r1)
        #input('getSample - 1D')
        return r1

    def Integ(self):
        if self.__STEPS == 0:
            return self.__p0 + self.__p1
        else:
            # print('ICI, self.__p0=', self.__p0, ', self.__p1=', self.__p1, ', self.__p01=', self.__p01)
            # integ = np.trapz(y=self.__p01, x=self.__Rcentres)
            # print('self.__Rcentres=', self.__Rcentres)
            # print('integ trapz=', integ)
            # integ = np.mean(self.__p01)
            # print('integ mean=', integ)
            # input('pause Integ')
            return self.__p0 + self.__p1 + np.mean(self.__p01)

    def plot(self, title):

        print('def plot(self, title): in LoiDiscreteFuzzy.py')
        print('  -->Le dessin de la figure ne fonctionne pas!')
        # fig = plt.figure()
        # ax = fig.gca()
        # plt.bar(x=self.__Rcentres, height=self.__p01, width=1./self.__STEPS, edgecolor='b', alpha=0.6, color='g')

        # #print('n=', n, ', bin=', bins, 'patch=', patches)

        # abscisse0= np.array([0., 0.])
        # data0 = np.array([0., self.__p0])
        # ax.plot(abscisse0, data0, alpha=0.6, color='r')

        # abscisse1= np.array([1., 1.])
        # data1 = np.array([0., self.__p1])
        # ax.plot(abscisse1, data1, alpha=0.6, color='r')

        # ax.set_xlabel('$r$')
        # ax.set_xlim(-0.05, 1.05)
        # ax.set_ylabel(title)
        #plt.show()
        #plt.close()
