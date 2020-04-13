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

from Fuzzy.LoisDiscreteFuzzy import Loi2DDiscreteFuzzy, Loi1DDiscreteFuzzy
from Fuzzy.InterFuzzy        import InterLineaire_Matrix, InterLineaire_Vector
from CommonFun.CommonFun     import From_Cov_to_FQ_bis

def getGaussXY(M, Lambda2, P, Pi2, xn, yn, xnpun, ynpun, verb=False):
    
    MeanX = M[0]*xn + M[1]*yn + M[2]*ynpun + M[3]
    if verb == True:
        print('\n  MeanX  =', MeanX)
        print('  M        =', M)
        print('  Lambda2  =', Lambda2)
        print('  xnpun    =', xnpun)
    if Lambda2 != 0.:
        #GaussX = multivariate_normal.pdf(xnpun, mean=MeanX, cov=Lambda2)
        GaussX = norm.pdf(x=xnpun, loc=MeanX, scale=np.sqrt(Lambda2)).item()
        # if GaussX>100.:
        #     print('GaussX=', GaussX)
        #     print('np.sqrt(Lambda2)=', np.sqrt(Lambda2))
        #     print('MeanX=', MeanX)
        #     print('xnpun=', xnpun)
        #     input('pause')
    else:
        return 0.

    MeanY = P[0]*yn + P[1]
    if verb == True:
        print('  MeanY  =', MeanY)
        print('  P        =', P)
        print('  Pi2    =', Pi2)
        print('  ynpun  =', ynpun)
    if Pi2 != 0.:
        #GaussY = multivariate_normal.pdf(ynpun, mean=MeanY, cov=Pi2)
        GaussY = norm.pdf(x=ynpun, loc=MeanY, scale=np.sqrt(Pi2)).item()
        # if GaussY>100.:
        #     print('GaussY=', GaussY)
        #     print('np.sqrt(Pi2)=', np.sqrt(Pi2))
        #     print('MeanY=', MeanY)
        #     print('ynpun=', ynpun)
        #     input('pause')
    else:
        return 0.

    return GaussX*GaussY

def loiForw(rn, rnp1, probaR2CondR1):

    result = probaR2CondR1(rn, rnp1)
    if not np.isfinite(result):
        print('probaR2CondR1(rn, rnp1)=', probaR2CondR1(rn, rnp1))
        input('Attente loiForw')

    return result

def calcF(indrnp1, rnp1, EPS, STEPS, Rcentres, ProbaF, FS, Tab_GaussXY_np1):

    argument = (rnp1, FS.probaR2CondR1)
    A        = 0.
    for indrn in range(1, STEPS+1):
        # ON NE PEUT PAS REMPLACER PAR UNE SIMPLE SOMME POUR GAGNER DU TEMPS!!!! 
        #   Parce que probaR2CondR1 n'est pas constant sur l'intervalle d'intégration
        #   Par contre le reste l'est, donc on peut le sortir de l'intégration numérique
    
        # Cette solution est la plus rapide car plein de choses sont constantes sur le petit interval à intégrer
        GaussXY = Tab_GaussXY_np1.getindr(indrn, indrnp1)
        if GaussXY > 0.:
            ATemp, errTemp = sc.integrate.quad(func=loiForw, a=float(indrn-1)/STEPS+EPS, b=float(indrn)/STEPS-EPS, args=argument, epsabs=1E-2, epsrel=1E-2, limit=50)
            A += ATemp * GaussXY * ProbaF.getindr(indrn)
    
    rn, indrn = 0., 0
    A0 = FS.probaR2CondR1(rn, rnp1) * Tab_GaussXY_np1.getindr(indrn, indrnp1) * ProbaF.getindr(indrn)
    
    rn, indrn = 1., STEPS+1
    A1 = FS.probaR2CondR1(rn, rnp1) * Tab_GaussXY_np1.getindr(indrn, indrnp1) * ProbaF.getindr(indrn)
    
    if not np.isfinite(A + A0 + A1):
        print('  A  = ', A, ',  A0 = ', A0, ',  A1 = ', A1)
        input('Nan!!')

    return A+A0+A1


def loiBackw(rnp1, rn, probaR2CondR1):
    return loiForw(rn, rnp1, probaR2CondR1)


def calcB(indrn, rn, EPS, STEPS, Rcentres, ProbaB, FS, Tab_GaussXY_np1):

    argument = (rn, FS.probaR2CondR1)
    A        = 0.
    for indrnp1 in range(1, STEPS+1):
        # ON NE PEUT PAS REMPLACER PAR UNE SIMPLE SOMME POUR GAGNER DU TEMPS!!!! 
        #   Parce que probaR2CondR1 n'est pas constant sur l'instervalle d'intégration
        #   Par contre le reste l'est, donc on peut le sortir de l'intégration numérique
    
        # Cette solution est la plus rapide car plein de choses sont constantes sur le petit interval à intégrer
        GaussXY = Tab_GaussXY_np1.getindr(indrn, indrnp1)
        if GaussXY > 0.:
            ATemp, errTemp = sc.integrate.quad(func=loiBackw, a=float(indrnp1-1)/STEPS+EPS, b=float(indrnp1)/STEPS-EPS, args=argument, epsabs=1E-2, epsrel=1E-2, limit=50)
            A += ATemp * GaussXY * ProbaB.getindr(indrnp1)

            if not np.isfinite(A):
                print('A=', A)
                print(ATemp, GaussXY, ProbaB.getindr(indrnp1))
                input('calcB')
    
    rnp1, indrnp1 = 0., 0
    A0 = FS.probaR2CondR1(rn, rnp1) * Tab_GaussXY_np1.getindr(indrn, indrnp1) * ProbaB.getindr(indrnp1)
    
    rnp1, indrnp1 = 1., STEPS+1
    A1 = FS.probaR2CondR1(rn, rnp1) * Tab_GaussXY_np1.getindr(indrn, indrnp1) * ProbaB.getindr(indrnp1)

    if not np.isfinite(A + A0 + A1):
        print('  A  = ', A, ',  A0 = ', A0, ',  A1 = ', A1)

        print('Pour A0')
        rnp1, indrnp1 = 0., 0
        print(rn, indrn)
        print(rnp1, indrnp1)
        print(FS.probaR2CondR1(rn, rnp1))
        print(Tab_GaussXY_np1.getindr(indrn, indrnp1))
        print(ProbaB.getindr(indrnp1))

        print('Pour A1')
        rnp1, indrnp1 = 1., STEPS+1
        print(rn, indrn)
        print(rnp1, indrnp1)
        print(FS.probaR2CondR1(rn, rnp1))
        print(Tab_GaussXY_np1.getindr(indrn, indrnp1))
        print(ProbaB.getindr(indrnp1))
        input('Nan!!')

    return A+A0+A1


############################################################################################################
class Loi2DDiscreteFuzzy_TMC(Loi2DDiscreteFuzzy):

    def __init__(self, EPS, STEPS, Rcentres):
        Loi2DDiscreteFuzzy.__init__(self, EPS, STEPS, Rcentres)

    def Calc_GaussXY(self, M, Lambda2, P, Pi2, zn, znp1):

        xn, yn, xnpun, ynpun = zn[0:1], zn[1:2], znp1[0:1], znp1[1:2]

        # Pour les masses
        indrn   = 0
        indrnp1 = 0
        self.__p00 = getGaussXY(M[indrn, indrnp1], Lambda2[indrn, indrnp1], P[indrn, indrnp1], Pi2[indrn, indrnp1], xn, yn, xnpun, ynpun)
        
        indrn   = self.__STEPS+1
        indrnp1 = 0
        self.__p10 = getGaussXY(M[indrn, indrnp1], Lambda2[indrn, indrnp1], P[indrn, indrnp1], Pi2[indrn, indrnp1], xn, yn, xnpun, ynpun)

        indrn   = self.__STEPS+1
        indrnp1 = self.__STEPS+1
        self.__p11 = getGaussXY(M[indrn, indrnp1], Lambda2[indrn, indrnp1], P[indrn, indrnp1], Pi2[indrn, indrnp1], xn, yn, xnpun, ynpun)
        
        indrn   = 0
        indrnp1 = self.__STEPS+1
        self.__p01 = getGaussXY(M[indrn, indrnp1], Lambda2[indrn, indrnp1], P[indrn, indrnp1], Pi2[indrn, indrnp1], xn, yn, xnpun, ynpun)

        # Pour les arrètes et le coeur
        for ind in range(1, self.__STEPS+1):
            
            indrnp1 = 0
            self.__p00_10[ind-1] = getGaussXY(M[ind, indrnp1], Lambda2[ind, indrnp1], P[ind, indrnp1], Pi2[ind, indrnp1], xn, yn, xnpun, ynpun)

            indrn = self.__STEPS+1
            self.__p10_11[ind-1] = getGaussXY(M[indrn, ind], Lambda2[indrn, ind], P[indrn, ind], Pi2[indrn, ind], xn, yn, xnpun, ynpun)

            indrnp1 = self.__STEPS+1
            self.__p11_01[ind-1] = getGaussXY(M[ind, indrnp1], Lambda2[ind, indrnp1], P[ind, indrnp1], Pi2[ind, indrnp1], xn, yn, xnpun, ynpun)

            indrn = 0
            self.__p01_00[ind-1] = getGaussXY(M[indrn, ind], Lambda2[indrn, ind], P[indrn, ind], Pi2[indrn, ind], xn, yn, xnpun, ynpun)

            # Pour l'intérieur
            for ind2 in range(1, self.__STEPS+1):
                self.__p[ind-1, ind2-1] = getGaussXY(M[ind, ind2], Lambda2[ind, ind2], P[ind, ind2], Pi2[ind, ind2], xn, yn, xnpun, ynpun)


    def CalcPsi(self, PForward_n, PBackward_np1, FS, Tab_GaussXY_np1):

        # Pour les masses
        rn, indrn      = 0., 0
        rnp1, indrnp1  = 0., 0
        self.__p00     = PForward_n.getindr(indrn) * PBackward_np1.getindr(indrnp1) * Tab_GaussXY_np1.getindr(indrn, indrnp1) * FS.probaR2CondR1(rn, rnp1)
        
        rn, indrn      = 1., self.__STEPS+1
        rnp1, indrnp1  = 0., 0
        self.__p10     = PForward_n.getindr(indrn) * PBackward_np1.getindr(indrnp1) * Tab_GaussXY_np1.getindr(indrn, indrnp1) * FS.probaR2CondR1(rn, rnp1)

        rn, indrn      = 1., self.__STEPS+1
        rnp1, indrnp1  = 1., self.__STEPS+1
        self.__p11     = PForward_n.getindr(indrn) * PBackward_np1.getindr(indrnp1) * Tab_GaussXY_np1.getindr(indrn, indrnp1) * FS.probaR2CondR1(rn, rnp1)
        
        rn, indrn      = 0., 0
        rnp1, indrnp1  = 1., self.__STEPS+1
        self.__p01     = PForward_n.getindr(indrn) * PBackward_np1.getindr(indrnp1) * Tab_GaussXY_np1.getindr(indrn, indrnp1) * FS.probaR2CondR1(rn, rnp1)

        # Pour les arètes et le coeur
        for indr, r in enumerate(self.__Rcentres):
            
            # self.__p00_10
            rnp1, indrnp1 = 0., 0
            self.__p00_10[indr] = PForward_n.getindr(indr+1) * PBackward_np1.getindr(indrnp1) * Tab_GaussXY_np1.getindr(indr+1, indrnp1) * FS.probaR2CondR1(r, rnp1)

            # self.__p10_11
            rn, indrn = 1., self.__STEPS+1
            self.__p10_11[indr] = PForward_n.getindr(indrn) * PBackward_np1.getindr(indr+1) * Tab_GaussXY_np1.getindr(indrn, indr+1) * FS.probaR2CondR1(rn, r)

            # self.__p11_01
            rnp1, indrnp1 = 1., self.__STEPS+1
            self.__p11_01[indr] = PForward_n.getindr(indr+1) * PBackward_np1.getindr(indrnp1) * Tab_GaussXY_np1.getindr(indr+1, indrnp1) * FS.probaR2CondR1(r, rnp1)

            # self.__p01_00
            rn, indrn = 0., 0
            self.__p01_00[indr] = PForward_n.getindr(indrn) * PBackward_np1.getindr(indr+1) * Tab_GaussXY_np1.getindr(indrn, indr+1) * FS.probaR2CondR1(rn, r)

            # Pour l'intérieur
            for indr2, r2 in enumerate(self.__Rcentres):
                self.__p[indr, indr2] = PForward_n.getindr(indr+1) * PBackward_np1.getindr(indr2+1) * Tab_GaussXY_np1.getindr(indr+1, indr2+1) * FS.probaR2CondR1(r, r2)


############################################################################################################
class Loi1DDiscreteFuzzy_TMC(Loi1DDiscreteFuzzy):

    def __init__(self, EPS, STEPS, Rcentres):
        Loi1DDiscreteFuzzy.__init__(self, EPS, STEPS, Rcentres)

    def CalcForw1(self, FS, z, MeanCovFuzzy):
        
        alpha, ind = 0., 0 # le premier
        if not np.any(MeanCovFuzzy.getCov(ind)) == False:
            self.__p0 = FS.probaR(alpha) * multivariate_normal.pdf(z, mean=MeanCovFuzzy.getMean(ind), cov=MeanCovFuzzy.getCov(ind))
        else:
            self.__p0   = 0.

        for ind, alpha in enumerate(self.__Rcentres):
            if not np.any(MeanCovFuzzy.getCov(ind+1)) == False:
                self.__p01[ind] = FS.probaR(alpha) * multivariate_normal.pdf(z, mean=MeanCovFuzzy.getMean(ind+1), cov=MeanCovFuzzy.getCov(ind+1))
            else:
                self.__p01[ind] = 0.

        alpha, ind = 1., self.__STEPS+1 # le dernier
        if not np.any(MeanCovFuzzy.getCov(ind)) == False:
            self.__p1 = FS.probaR(alpha) * multivariate_normal.pdf(z, mean=MeanCovFuzzy.getMean(ind), cov=MeanCovFuzzy.getCov(ind))
        else:
            self.__p1 = 0.


    def CalcForB(self, FctCalculForB, probaForB, FS, Tab_GaussXY_np1):

        # les proba sont renvoyées non normalisées
        r, ind = 0., 0
        self.__p0 = FctCalculForB(ind, r, self.__EPS, self.__STEPS, self.__Rcentres, probaForB, FS, Tab_GaussXY_np1)
        
        for ind, r in enumerate(self.__Rcentres):
            self.__p01[ind] = FctCalculForB(ind+1, r, self.__EPS, self.__STEPS, self.__Rcentres, probaForB, FS, Tab_GaussXY_np1)
        
        r, ind = 1., self.__STEPS+1
        self.__p1 = FctCalculForB(ind, r, self.__EPS, self.__STEPS, self.__Rcentres, probaForB, FS, Tab_GaussXY_np1)


    def CalcCond(self, indrn, ProbaGamma_n_rn, ProbaPsi_n, verbose):

        indrnp1 = 0
        self.__p0 = ProbaPsi_n.getindr(indrn, indrnp1) / ProbaGamma_n_rn

        for indrnp1 in range(1, self.__STEPS+1):
            self.__p01[indrnp1-1] = ProbaPsi_n.getindr(indrn, indrnp1) / ProbaGamma_n_rn

        indrnp1 = self.__STEPS+1
        self.__p1 = ProbaPsi_n.getindr(indrn, indrnp1) / ProbaGamma_n_rn

        # Verification a priori
        # integ = self.Integ()
        # if abs(1.-integ)>1e-2: # 1% d'erreur
        #     self.print()
        #     print('integ CalcCond=', integ)
        #     print('sum CalcCond=', self.Sum())
        #     print('indrn=', indrn)
        #     input('Attente dans CalcCond a priori')

        # test if all 0 
        if self.TestIsAllZero():
            # on met une lois uniforme
            self.setValCste(1.)
            self.normalisation(self.Integ())
            self.print()
            print('\nWarning: all the proba cond is 0. when indrn=' + str(indrn))
            #input('Attente')
        else:
            # this normalisation is only to avoid numerical integration pb due to the number of discrete fuzzy steps
            # If the number of fuzzy steps is low then the numerical error is big (STEPS=1 gives about 20% error, STEPS = 5 gives about 2% of error)
            # if abs(1.-integ) > 5.E-2: # no stop till 5% of error
            #     print('  Integ R2 condit. to R1 and Z: integ', integ)
            #     input('PB PB PB proba cond')
            self.normalisation(self.Integ())

        # Verification a posteriori
        integ = self.Integ()
        if abs(1.-integ)>1e-2: # 1% d'erreur
            self.print()
            print('integ CalcCond=', integ)
            print('indrn=', indrn)
            input('Attente dans CalcCond a posteriori')

        # self.print()
        # input('Attente dans CalcCond')
