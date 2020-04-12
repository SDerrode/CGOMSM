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
class Loi2DDiscreteFuzzy_TMC():

    def __init__(self, EPS, STEPS, Rcentres):
        self.__EPS      = EPS
        self.__STEPS    = STEPS
        self.__Rcentres = Rcentres
        if len(Rcentres) != self.__STEPS: input('PB constructeur Loi2DDiscreteFuzzy_TMC')

        self.__p00 = 0.
        self.__p10 = 0.
        self.__p01 = 0.
        self.__p11 = 0.
        
        if self.__STEPS != 0:
            self.__p00_10 = np.zeros(shape=(self.__STEPS))
            self.__p10_11 = np.zeros(shape=(self.__STEPS))
            self.__p11_01 = np.zeros(shape=(self.__STEPS))
            self.__p01_00 = np.zeros(shape=(self.__STEPS))

            self.__p = np.zeros(shape=(self.__STEPS, self.__STEPS))
        else:
            self.__p00_10 = np.empty(shape=(0))
            self.__p10_11 = np.empty(shape=(0))
            self.__p11_01 = np.empty(shape=(0))
            self.__p01_00 = np.empty(shape=(0))

            self.__p = np.empty(shape=(0))


    # def getr(self, r1, r2):
        
    #     if r1==1.:
    #         if r2==0.: return self.__p10
    #         if r2==1.: return self.__p11
    #         return self.__p10_11[math.floor(r2*self.__STEPS)]

    #     if r1==0.:
    #         if r2==0.: return self.__p00
    #         if r2==1.: return self.__p01
    #         return self.__p01_00[math.floor(r2*self.__STEPS)]
        
    #     indr1 = math.floor(r1*self.__STEPS)
    #     if r2==0.: return self.__p00_10[indr1]
    #     if r2==1.: return self.__p11_01[indr1]
    #     return self.__p[indr1, math.floor(r2*self.__STEPS)]

    def getindr(self, indr1, indr2):
        
        if indr1==self.__STEPS+1:
            if indr2==0:              return self.__p10
            if indr2==self.__STEPS+1: return self.__p11
            return self.__p10_11[indr2-1]  

        if indr1==0:
            if indr2==0:              return self.__p00
            if indr2==self.__STEPS+1: return self.__p01
            return self.__p01_00[indr2-1]
        
        if indr2==0:              return self.__p00_10[indr1-1]
        if indr2==self.__STEPS+1: return self.__p11_01[indr1-1]
        return self.__p[indr1-1, indr2-1]
        

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
            rnp1, indrnp1      = 0., 0
            self.__p00_10[indr] = PForward_n.getindr(indr+1) * PBackward_np1.getindr(indrnp1) * Tab_GaussXY_np1.getindr(indr+1, indrnp1) * FS.probaR2CondR1(r, rnp1)

            # self.__p10_11
            rn, indrn          = 1., self.__STEPS+1
            self.__p10_11[indr] = PForward_n.getindr(indrn) * PBackward_np1.getindr(indr+1) * Tab_GaussXY_np1.getindr(indrn, indr+1) * FS.probaR2CondR1(rn, r)

            # self.__p11_01
            rnp1, indrnp1     = 1., self.__STEPS+1
            self.__p11_01[indr] = PForward_n.getindr(indr+1) * PBackward_np1.getindr(indrnp1) * Tab_GaussXY_np1.getindr(indr+1, indrnp1) * FS.probaR2CondR1(r, rnp1)

            # self.__p01_00
            rn, indrn          = 0., 0
            self.__p01_00[indr] = PForward_n.getindr(indrn) * PBackward_np1.getindr(indr+1) * Tab_GaussXY_np1.getindr(indrn, indr+1) * FS.probaR2CondR1(rn, r)

            # Pour l'intérieur
            for indr2, r2 in enumerate(self.__Rcentres):
                self.__p[indr, indr2] = PForward_n.getindr(indr+1) * PBackward_np1.getindr(indr2+1) * Tab_GaussXY_np1.getindr(indr+1, indr2+1) * FS.probaR2CondR1(r, r2)


    def Integ(self):

        if self.__STEPS == 0:
            return self.__p00 + self.__p10 + self.__p01 + self.__p11

        #### pour r1==0.
        integ = np.mean(self.__p01_00) + self.__p00 + self.__p01

        #### pour r1==1.
        integ += np.mean(self.__p10_11) + self.__p10 + self.__p11

        #### La surface à l'intérieur
        pR = np.ndarray(shape=(self.__STEPS))
        for j in range(self.__STEPS):
            pR[j] = np.mean(self.__p[j, :]) + self.__p00_10[j] + self.__p11_01[j]
        integ += np.mean(pR)

        return integ


    def normalisation(self, norm):

        if norm != 0.:
            self.__p00    /= norm
            self.__p10    /= norm
            self.__p01    /= norm
            self.__p11    /= norm    
            
            self.__p00_10 /= norm
            self.__p10_11 /= norm
            self.__p11_01 /= norm
            self.__p01_00 /= norm 
            
            self.__p      /= norm
        else:
            input('pb if norm == 0.')


    def print(self):

        print("Les coins:")
        print('  p01=', self.__p01, ',  p11=', self.__p11)
        print('  p00=', self.__p00, ',  p10=', self.__p10)

        if self.__STEPS != 0:
            print("Les arêtes:")
            for i in range(self.__STEPS):
                print(self.__p00_10[i], end=' ')
            print("\n")
            for i in range(self.__STEPS):
                print(self.__p10_11[i], end=' ')
            print("\n")
            for i in range(self.__STEPS):
                print(self.__p11_01[i], end=' ')
            print("\n")
            for i in range(self.__STEPS):
                print(self.__p01_00[i], end=' ')

            print("\nLe coeur:")
            for indrn in range(self.__STEPS):
                for indrnp1 in range(self.__STEPS):
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
        if self.__STEPS != 0:
            self.__p01 = np.zeros(shape=(self.__STEPS))
        else:
            self.__p01 = np.empty(shape=(0,))
        self.__p1 = 0.
    
    def ProductFB(self, loi1, loi2):
        self.__p0 = loi1.__p0 * loi2.__p0
        for i in range(self.__STEPS):
            self.__p01[i] = loi1.__p01[i] * loi2.__p01[i]
        self.__p1 = loi1.__p1 * loi2.__p1

    def getRcentre(self):
        return self.__Rcentres

    def getSTEPS(self):
        return self.__STEPS

    def getEPS(self):
        return self.__EPS

    # def getr(self, r):
    #     if r==0.: return self.__p0
    #     if r==1.: return self.__p1
    #     return self.__p01[math.floor(r*self.__STEPS)]

    def getindr(self, indr):
        if indr==0:              return self.__p0
        if indr==self.__STEPS+1: return self.__p1
        return self.__p01[indr-1]    

    # def setr(self, r, val):
    #     if   r==0.: self.__p0=val
    #     elif r==1.: self.__p1=val
    #     else:       self.__p01[math.floor(r*self.__STEPS)]=val

    # def setindr(self, indr, val):
    #     if   indr==0:              self.__p0=val
    #     elif indr==self.__STEPS+1: self.__p1=val
    #     else:                      self.__p01[indr]=val
 
    def print(self):
        print('__p0 = ', self.__p0)
        for i, rnp1 in enumerate(self.__Rcentres):
            print('  __p01[',rnp1, ']=', self.__p01[i])
        print('__p1 = ', self.__p1)


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

    def setValCste(self, val):
        self.__p0 = val
        for i in range(self.__STEPS):
            self.__p01[i] = val
        self.__p1 = val

    def CalcForB(self, FctCalculForB, probaForB, FS, Tab_GaussXY_np1):

        # les proba sont renvoyées non normalisées
        r, ind = 0., 0
        self.__p0 = FctCalculForB(ind, r, self.__EPS, self.__STEPS, self.__Rcentres, probaForB, FS, Tab_GaussXY_np1)
        
        for ind, r in enumerate(self.__Rcentres):
            self.__p01[ind] = FctCalculForB(ind+1, r, self.__EPS, self.__STEPS, self.__Rcentres, probaForB, FS, Tab_GaussXY_np1)
        
        r, ind = 1., self.__STEPS+1
        self.__p1 = FctCalculForB(ind, r, self.__EPS, self.__STEPS, self.__Rcentres, probaForB, FS, Tab_GaussXY_np1)

    # def nextAfterZeros(self):
    #     if self.__p0 < 1e-300:
    #         self.__p0 = 1e-300 #np.nextafter(0, 1)*10

    #     if self.__p1 < 1e-300:
    #         self.__p1 = 1e-300 #np.nextafter(0, 1)*10

    #     for i in range(self.__STEPS):
    #         if self.__p01[i] < 1e-300:
    #             self.__p01[i] = 1e-300 #np.nextafter(0, 1)*10


    def normalisation(self, norm, verbose=2):
        if norm != 0.:
            self.__p0  /= norm
            self.__p01 /= norm
            self.__p1  /= norm
        else:
            if verbose>2:
                print('ATTENTION : norm == 0.')

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

    def TestIsAllZero(self):
        if self.__p0 != 0.: return False
        if self.__p1 != 0.: return False
        for ind in range(self.__STEPS):
            if self.__p01[ind] != 0.: return False
        return True

    def getSample(self):

        proba = np.array([self.__p0, self.__p1, 1. - (self.__p0+self.__p1)])
        typeSample = random.choices(population=[0, 1, 2], weights=proba)[0]
        if typeSample==0: 
            indr=0
        elif typeSample==1:
            indr=self.__STEPS+1
        else: # it is fuzzy
            probaF = self.__p01 / (proba[2]*self.__STEPS)
            indr = random.choices(population=list(range(1, self.__STEPS+1)), weights=probaF)[0]
        return indr


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

