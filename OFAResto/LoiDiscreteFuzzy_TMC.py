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
from Fuzzy.InterFuzzy        import InterBiLineaire_Matrix, InterLineaire_Matrix, InterLineaire_Vector
from CommonFun.CommonFun     import From_Cov_to_FQ_bis


def loiForw(rn, rnp1, probaR2CondR1):

    result = probaR2CondR1(rn, rnp1)
    if not np.isfinite(result):
        print('probaR2CondR1(rn, rnp1)=', probaR2CondR1(rn, rnp1))
        input('Attente loiForw')

    return result

def calcF(indrnp1, rnp1, EPS, STEPS, Rcentres, ProbaF, FS, Tab_GaussXY_np1):

    argument = (rnp1, FS.probaR2CondR1)
    A        = 0.
    for indrn in range(STEPS):
        # ON NE PEUT PAS REMPLACER PAR UNE SIMPLE SOMME POUR GAGNER DU TEMPS!!!! 
        #   Parce que probaR2CondR1 n'est pas constant sur l'intervalle d'intégration
        #   Par contre le reste l'est, donc on peut le sortir de l'intégration numérique
    
        # Cette solution est la plus rapide car plein de choses sont constantes sur le petit interval à intégrer
        GaussXY = Tab_GaussXY_np1.getindr(indrn+1, indrnp1)
        if GaussXY > 0.:
            ATemp, errTemp = sc.integrate.quad(func=loiForw, a=float(indrn)/STEPS+EPS, b=float(indrn+1)/STEPS-EPS, args=argument, epsabs=1E-2, epsrel=1E-2, limit=50)
            A += ATemp * GaussXY * ProbaF.getindr(indrn+1)
    
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
    for indrnp1 in range(STEPS):
        # ON NE PEUT PAS REMPLACER PAR UNE SIMPLE SOMME POUR GAGNER DU TEMPS!!!! 
        #   Parce que probaR2CondR1 n'est pas constant sur l'instervalle d'intégration
        #   Par contre le reste l'est, donc on peut le sortir de l'intégration numérique
    
        # Cette solution est la plus rapide car plein de choses sont constantes sur le petit interval à intégrer
        GaussXY = Tab_GaussXY_np1.getindr(indrn, indrnp1+1)
        if GaussXY > 0.:
            ATemp, errTemp = sc.integrate.quad(func=loiBackw, a=float(indrnp1)/STEPS+EPS, b=float(indrnp1+1)/STEPS-EPS, args=argument, epsabs=1E-2, epsrel=1E-2, limit=50)
            A += ATemp * GaussXY * ProbaB.getindr(indrnp1+1)

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

    def __init__(self, EPS, STEPS, interpolation, Rcentres):
        self._interpolation = interpolation
        Loi2DDiscreteFuzzy.__init__(self, EPS, STEPS, Rcentres)


    def CalcPsi(self, PForward_n, PBackward_np1, FS, Tab_GaussXY_np1):

        # Pour les masses
        rn, indrn      = 0., 0
        rnp1, indrnp1  = 0., 0
        self._p00     = PForward_n.getindr(indrn) * PBackward_np1.getindr(indrnp1) * Tab_GaussXY_np1.getindr(indrn, indrnp1) * FS.probaR2CondR1(rn, rnp1)
        
        rn, indrn      = 1., self._STEPSp1
        rnp1, indrnp1  = 0., 0
        self._p10     = PForward_n.getindr(indrn) * PBackward_np1.getindr(indrnp1) * Tab_GaussXY_np1.getindr(indrn, indrnp1) * FS.probaR2CondR1(rn, rnp1)

        rn, indrn      = 1., self._STEPSp1
        rnp1, indrnp1  = 1., self._STEPSp1
        self._p11     = PForward_n.getindr(indrn) * PBackward_np1.getindr(indrnp1) * Tab_GaussXY_np1.getindr(indrn, indrnp1) * FS.probaR2CondR1(rn, rnp1)
        
        rn, indrn      = 0., 0
        rnp1, indrnp1  = 1., self._STEPSp1
        self._p01     = PForward_n.getindr(indrn) * PBackward_np1.getindr(indrnp1) * Tab_GaussXY_np1.getindr(indrn, indrnp1) * FS.probaR2CondR1(rn, rnp1)

        # Pour les arètes et le coeur
        for indr, r in enumerate(self._Rcentres):
            
            # self._p00_10
            rnp1, indrnp1 = 0., 0
            self._p00_10[indr] = PForward_n.getindr(indr+1) * PBackward_np1.getindr(indrnp1) * Tab_GaussXY_np1.getindr(indr+1, indrnp1) * FS.probaR2CondR1(r, rnp1)

            # self._p10_11
            rn, indrn = 1., self._STEPSp1
            self._p10_11[indr] = PForward_n.getindr(indrn) * PBackward_np1.getindr(indr+1) * Tab_GaussXY_np1.getindr(indrn, indr+1) * FS.probaR2CondR1(rn, r)

            # self._p11_01
            rnp1, indrnp1 = 1., self._STEPSp1
            self._p11_01[indr] = PForward_n.getindr(indr+1) * PBackward_np1.getindr(indrnp1) * Tab_GaussXY_np1.getindr(indr+1, indrnp1) * FS.probaR2CondR1(r, rnp1)

            # self._p01_00
            rn, indrn = 0., 0
            self._p01_00[indr] = PForward_n.getindr(indrn) * PBackward_np1.getindr(indr+1) * Tab_GaussXY_np1.getindr(indrn, indr+1) * FS.probaR2CondR1(rn, r)

            # Pour l'intérieur
            for indr2, r2 in enumerate(self._Rcentres):
                self._p[indr, indr2] = PForward_n.getindr(indr+1) * PBackward_np1.getindr(indr2+1) * Tab_GaussXY_np1.getindr(indr+1, indr2+1) * FS.probaR2CondR1(r, r2)


    def predicSauts(self, probaR2CondR1, ProbaForward_n):
        self._p00 = probaR2CondR1(0., 0.) * ProbaForward_n.getindr(0)
        self._p10 = probaR2CondR1(1., 0.) * ProbaForward_n.getindr(self._STEPSp1)
        self._p01 = probaR2CondR1(0., 1.) * ProbaForward_n.getindr(0.)
        self._p11 = probaR2CondR1(1., 1.) * ProbaForward_n.getindr(self._STEPSp1)

        for indr, r in enumerate(self._Rcentres):
            self._p00_10[indr] = probaR2CondR1(r, 0.) * ProbaForward_n.getindr(indr+1)
            self._p10_11[indr] = probaR2CondR1(1., r) * ProbaForward_n.getindr(self._STEPSp1)
            self._p11_01[indr] = probaR2CondR1(r, 1.) * ProbaForward_n.getindr(indr+1)
            self._p01_00[indr] = probaR2CondR1(0., r) * ProbaForward_n.getindr(0)

        for indr1, r1 in enumerate(self._Rcentres):
            for indr2, r2 in enumerate(self._Rcentres):
                self._p[indr1, indr2] = probaR2CondR1(r1, r2) * ProbaForward_n.getindr(indr1+1)


    def CalcProbaTransAposteriori(self, ProbaForward_np1, ProbaForward_n, loijointeAP1, probaR2CondR1, Cov, Mean_Y, yn, ynp1, np1):
        self._p00 = loijointeAP1(0., 0., 0,             ProbaForward_n, probaR2CondR1, Cov, Mean_Y, yn, ynp1, np1, self._interpolation, self._STEPS) / ProbaForward_np1.getindr(0)
        self._p10 = loijointeAP1(1., 0., 0,             ProbaForward_n, probaR2CondR1, Cov, Mean_Y, yn, ynp1, np1, self._interpolation, self._STEPS) / ProbaForward_np1.getindr(0)
        self._p01 = loijointeAP1(0., 1., self._STEPSp1, ProbaForward_n, probaR2CondR1, Cov, Mean_Y, yn, ynp1, np1, self._interpolation, self._STEPS) / ProbaForward_np1.getindr(self._STEPSp1)
        self._p11 = loijointeAP1(1., 1., self._STEPSp1, ProbaForward_n, probaR2CondR1, Cov, Mean_Y, yn, ynp1, np1, self._interpolation, self._STEPS) / ProbaForward_np1.getindr(self._STEPSp1)

        for indr, r in enumerate(self._Rcentres):
            self._p00_10[indr] = loijointeAP1(r, 0., 0,             ProbaForward_n, probaR2CondR1, Cov, Mean_Y, yn, ynp1, np1, self._interpolation, self._STEPS) / ProbaForward_np1.getindr(0)
            self._p10_11[indr] = loijointeAP1(1., r, indr+1,        ProbaForward_n, probaR2CondR1, Cov, Mean_Y, yn, ynp1, np1, self._interpolation, self._STEPS) / ProbaForward_np1.getindr(indr+1)
            self._p11_01[indr] = loijointeAP1(r, 1., self._STEPSp1, ProbaForward_n, probaR2CondR1, Cov, Mean_Y, yn, ynp1, np1, self._interpolation, self._STEPS) / ProbaForward_np1.getindr(self._STEPSp1)
            self._p01_00[indr] = loijointeAP1(0., r, indr+1,        ProbaForward_n, probaR2CondR1, Cov, Mean_Y, yn, ynp1, np1, self._interpolation, self._STEPS) / ProbaForward_np1.getindr(indr+1)

        for indr1, r1 in enumerate(self._Rcentres):
            for indr2, r2 in enumerate(self._Rcentres):
                self._p[indr1, indr2] = loijointeAP1(r1, r2, indr2+1, ProbaForward_n, probaR2CondR1, Cov, Mean_Y, yn, ynp1, np1, self._interpolation, self._STEPS) / ProbaForward_np1.getindr(indr2+1)
    



############################################################################################################
class Loi1DDiscreteFuzzy_TMC(Loi1DDiscreteFuzzy):

    def __init__(self, EPS, STEPS, interpolation, Rcentres):
        self._interpolation = interpolation
        Loi1DDiscreteFuzzy.__init__(self, EPS, STEPS, Rcentres)

    def CalcForw1(self, FS, z, MeanCovFuzzy):
        
        alpha, ind = 0., 0 # le premier
        if not np.any(MeanCovFuzzy.getCov(ind)) == False:
            self._p0 = FS.probaR(alpha) * multivariate_normal.pdf(z, mean=MeanCovFuzzy.getMean(ind), cov=MeanCovFuzzy.getCov(ind))
        else:
            self._p0   = 0.

        for ind, alpha in enumerate(self._Rcentres):
            if not np.any(MeanCovFuzzy.getCov(ind+1)) == False:
                self._p01[ind] = FS.probaR(alpha) * multivariate_normal.pdf(z, mean=MeanCovFuzzy.getMean(ind+1), cov=MeanCovFuzzy.getCov(ind+1))
            else:
                self._p01[ind] = 0.

        alpha, ind = 1., self._STEPSp1 # le dernier
        if not np.any(MeanCovFuzzy.getCov(ind)) == False:
            self._p1 = FS.probaR(alpha) * multivariate_normal.pdf(z, mean=MeanCovFuzzy.getMean(ind), cov=MeanCovFuzzy.getCov(ind))
        else:
            self._p1 = 0.


    def CalcForB(self, FctCalculForB, probaForB, FS, Tab_GaussXY_np1):

        # les proba sont renvoyées non normalisées
        r, ind = 0., 0
        self._p0 = FctCalculForB(ind, r, self._EPS, self._STEPS, self._Rcentres, probaForB, FS, Tab_GaussXY_np1)
        
        for ind, r in enumerate(self._Rcentres):
            self._p01[ind] = FctCalculForB(ind+1, r, self._EPS, self._STEPS, self._Rcentres, probaForB, FS, Tab_GaussXY_np1)
        
        r, ind = 1., self._STEPSp1
        self._p1 = FctCalculForB(ind, r, self._EPS, self._STEPS, self._Rcentres, probaForB, FS, Tab_GaussXY_np1)


    def CalcCond(self, indrn, ProbaGamma_n_rn, ProbaPsi_n, verbose):

        indrnp1 = 0
        self._p0 = ProbaPsi_n.getindr(indrn, indrnp1) / ProbaGamma_n_rn

        for indrnp1 in range(1, self._STEPSp1):
            self._p01[indrnp1-1] = ProbaPsi_n.getindr(indrn, indrnp1) / ProbaGamma_n_rn

        indrnp1 = self._STEPSp1
        self._p1 = ProbaPsi_n.getindr(indrn, indrnp1) / ProbaGamma_n_rn

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


    def setForward_1(self, probaR, Cov, y, Mean_Y):
        
        rnp1, indrnp1 = 0., 0

        rn, indrn = 0., 0
        if self._interpolation==True:
            Mean_Y_rn = InterLineaire_Vector(Mean_Y, rn)
            Cov_rn_0  = InterBiLineaire_Matrix(Cov, rn, rnp1)
        else:
            Mean_Y_rn = Mean_Y[indrn]
            Cov_rn_0  = Cov[indrn*self._STEPS+indrnp1]
        self._p0 = probaR(rn) * norm.pdf(y, loc=Mean_Y_rn, scale=np.sqrt(Cov_rn_0[1, 1])).item()

        rn, indrn = 1., self._STEPSp1
        if self._interpolation==True:
            Mean_Y_rn = InterLineaire_Vector(Mean_Y, rn)
            Cov_rn_0  = InterBiLineaire_Matrix(Cov, rn, rnp1)
        else:
            Mean_Y_rn = Mean_Y[indrn]
            Cov_rn_0  = Cov[indrn*self._STEPS+indrnp1]
        self._p1 = probaR(rn) * norm.pdf(y, loc=Mean_Y_rn, scale=np.sqrt(Cov_rn_0[1, 1])).item()

        for indrn, rn in enumerate(self._Rcentres):
            if self._interpolation==True:
                Mean_Y_rn = InterLineaire_Vector(Mean_Y, rn)
                Cov_rn_0  = InterBiLineaire_Matrix(Cov, rn, rnp1)
            else:
                Mean_Y_rn = Mean_Y[indrn+1]
                Cov_rn_0  = Cov[(indrn+1)*self._STEPS+(indrnp1+1)]
            self._p01[indrn] = probaR(rn) * norm.pdf(y, loc=Mean_Y_rn, scale=np.sqrt(Cov_rn_0[1, 1])).item()
        
        self.normalisation(self.Integ())


    def setForwBack(self, fonction, loijointeAP, proba, probaR2CondR1, Cov, Mean_Y, yn, ynp1, np1):
        rnp1, indrnp1 = 0., 0
        self._p0 = fonction(rnp1, indrnp1, self._interpolation, self._EPS, self._STEPS, loijointeAP, proba, probaR2CondR1, Cov, Mean_Y, yn, ynp1, np1)
        
        for indr, r in enumerate(self._Rcentres):
            self._p01[indr] = fonction(r, indr+1, self._interpolation, self._EPS, self._STEPS, loijointeAP, proba, probaR2CondR1, Cov, Mean_Y, yn, ynp1, np1)
        
        rnp1, indrnp1 = 1., self._STEPSp1
        self._p1 = fonction(rnp1, indrnp1, self._interpolation, self._EPS, self._STEPS, loijointeAP, proba, probaR2CondR1, Cov, Mean_Y, yn, ynp1, np1)


   

