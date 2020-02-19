#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 11:35:45 2017

@author: Stephane
"""
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy as sc
from scipy.stats import norm
from scipy.stats import multivariate_normal

from Fuzzy.InterFuzzy import InterLineaire_Matrix, InterLineaire_Vector
from CommonFun.CommonFun import From_Cov_to_FQ_bis


def loiForw2(rn, rnp1, probaR2CondR1):

    result = probaR2CondR1(rn, rnp1)
    if not np.isfinite(result):
        print('probaR2CondR1(rn, rnp1)=', probaR2CondR1(rn, rnp1))
        input('Attente loiForw')
    return result

def loiForw1(rn, rnp1, probaF, probaR2CondR1):

    result = probaF.get(rn) * probaR2CondR1(rn, rnp1)
    #print('\nrn=', rn, ', probaF.get(rn)=', probaF.get(rn))

    if not np.isfinite(result):
        print('probaF.get(rn)=', probaF.get(rn))
        print('probaR2CondR1(rn, rnp1)=', probaR2CondR1(rn, rnp1))
        input('Attente loiForw')
    return result


def calcF(rnp1, EPS, STEPS, ProbaF, FS, znp1, CovZ, MeanZ):

    argument1 = (rnp1, ProbaF, FS.probaR2CondR1)
    argument2 = (rnp1, FS.probaR2CondR1)
    A         = 0.
    #import time
    #start_time = time.time()
    #for K in range(10000):
    if STEPS != 0:
        for i in range(STEPS):
            # ON NE PEUT PAS REMPLACER PAR UNE SIMPLE SOMME POUR GAGNER DU TEMPS!!!! parce que probaR2CondR1 n'est pas discret

            # Cette solution est plus lebte que l suivante de 30%
            # ATemp, errATemp = sc.integrate.quad(func=loiForw1, a=float(i)/STEPS+EPS, b=float(i+1)/STEPS-EPS, args=argument1, epsabs=1E-2, epsrel=1E-2, limit=50)
            # A += ATemp
            # print('ATemp    =', ATemp)

            # Cette solution est la plus rapide
            ATemp, errATemp = sc.integrate.quad(func=loiForw2, a=float(i)/STEPS+EPS, b=float(i+1)/STEPS-EPS, args=argument2, epsabs=1E-2, epsrel=1E-2, limit=50)
            A += ATemp * ProbaF.get(float(i)/STEPS+2.*EPS)
            # print('ATemp    =', ATemp * ProbaF.get(float(i)/STEPS+EPS))
            # input('attente')
        # La solution suivante est équivalente mais prend 10% de temps en plus
        # C, errCTemp = sc.integrate.quad(func=loiForw1, a=EPS, b=1.-EPS, args=argument, epsabs=1E-5, epsrel=1E-1, limit=50)
    # print('\nA=', A)
    # print('B=', B)
    # print('C=', C)
    #print("\n--- %s seconds ---" % (time.time() - start_time))
    #input('TIME')

    # A0 = loiForw1(0., rnp1, ProbaF, FS.probaR2CondR1)
    # A1 = loiForw1(1., rnp1, ProbaF, FS.probaR2CondR1)
    A0 = loiForw2(0., rnp1, FS.probaR2CondR1) * ProbaF.get(0.)
    A1 = loiForw2(1., rnp1, FS.probaR2CondR1) * ProbaF.get(1.)
    # print('A0 = ', A0)
    # print('A1 = ', A1)
    # input('STOP')
    if not np.isfinite(A + A0 + A1):
        print('A  = ', A)
        print('A0 = ', A0)
        print('A1 = ', A1)
        input('Nan!!')

    # On interpole et on calcul la gaussienne
    CovZ_rnp1  = InterLineaire_Matrix(CovZ,  rnp1)
    MeanZ_rnp1 = InterLineaire_Vector(MeanZ, rnp1)
    Gauss      = multivariate_normal.pdf(znp1, mean=MeanZ_rnp1, cov=CovZ_rnp1)

    return Gauss*(A+A0+A1)


def loiBackw2(rnp1, rn, probaR2CondR1, znp1, CovZ, MeanZ):

    # On interpole et on calcul la gaussienne
    CovZ_rnp1  = InterLineaire_Matrix(CovZ,  rnp1)
    MeanZ_rnp1 = InterLineaire_Vector(MeanZ, rnp1)
    Gauss      = multivariate_normal.pdf(znp1, mean=MeanZ_rnp1, cov=CovZ_rnp1)

    result = Gauss * probaR2CondR1(rn, rnp1)
    if not np.isfinite(result):
        print('probaR2CondR1(rn, rnp1)=', probaR2CondR1(rn, rnp1))
        print('Gauss=', Gauss)
        input('Attente loiBackw')

    return result

def loiBackw1(rnp1, rn, probaB, probaR2CondR1, znp1, CovZ, MeanZ):

    # On interpole et on calcul la gaussienne
    CovZ_rnp1  = InterLineaire_Matrix(CovZ,  rnp1)
    MeanZ_rnp1 = InterLineaire_Vector(MeanZ, rnp1)
    Gauss      = multivariate_normal.pdf(znp1, mean=MeanZ_rnp1, cov=CovZ_rnp1)

    result = Gauss * probaB.get(rnp1) * probaR2CondR1(rn, rnp1)
    if not np.isfinite(result):
        print('probaB.get(rnp1)=', probaB.get(rnp1))
        print('probaR2CondR1(rn, rnp1)=', probaR2CondR1(rn, rnp1))
        print('Gauss=', Gauss)
        input('Attente loiBackw')

    return result


def calcB(rn, EPS, STEPS, ProbaB, FS, znp1, CovZ, MeanZ):
    
    argument1 = (rn, ProbaB, FS.probaR2CondR1, znp1, CovZ, MeanZ)
    argument2 = (rn, FS.probaR2CondR1, znp1, CovZ, MeanZ)
    A         = 0.
    if STEPS != 0:
        for i in range(STEPS):
            # ON NE PEUT PAS REMPLACER PAR UNE SIMPLE SOMME POUR GAGNER DU TEMPS!!!! parce que probaR2CondR1 n'est pas discret

            # Cette solution est plus lente
            # ATemp, errATemp = sc.integrate.quad(func=loiBackw1, a=float(i)/STEPS+EPS, b=float(i+1)/STEPS-EPS, args=argument1, epsabs=1E-10, epsrel=1E-10, limit=200)
            # A += ATemp

            # Cette solution est la plus rapide
            ATemp, errATemp = sc.integrate.quad(func=loiBackw2, a=float(i)/STEPS+EPS, b=float(i+1)/STEPS-EPS, args=argument2, epsabs=1E-2, epsrel=1E-2, limit=50)
            B += ATemp * ProbaB.get(float(i)/STEPS+2.*EPS)
    # print('\nA=', A)
    # print('B=', B)
    # input('TIME')

    # A0 = loiBackw1(0., rn, ProbaB, FS.probaR2CondR1, znp1, CovZ, MeanZ)
    # A1 = loiBackw1(1., rn, ProbaB, FS.probaR2CondR1, znp1, CovZ, MeanZ)
    # print('A0 = ', A0)
    # print('A1 = ', A1)
    A0 = loiBackw2(0., rn, FS.probaR2CondR1, znp1, CovZ, MeanZ) * ProbaB.get(0.)
    A1 = loiBackw2(1., rn, FS.probaR2CondR1, znp1, CovZ, MeanZ) * ProbaB.get(1.)
    # print('A0 = ', A0)
    # print('A1 = ', A1)
    # input('STOP')

    result = A + A0 + A1
    if not np.isfinite(result):
        print('A  = ', A)
        print('A0 = ', A0)
        print('A1 = ', A1)
        input('Nan!!')

    return result



############################################################################################################
class Loi1DDiscreteFuzzy_HMC():

    def __init__(self, EPS, STEPS, Rcentres):
        self.__EPS   = EPS
        self.__STEPS = STEPS
        self.__p0 = 0.
        self.__p1 = 0.
        if self.__STEPS != 0:
            self.__p01 = np.zeros(shape=(STEPS))
        else:
            self.__p01 = np.empty(shape=(0,))
        self.__Rcentres = Rcentres

    def getRcentre(self):
        return self.__Rcentres

    def getSTEPS(self):
        return self.__STEPS

    def get(self, r):
        if r>1.-self.__EPS:
            return self.__p1
        elif r>= self.__EPS:
            indice = math.floor(r*self.__STEPS)
            return self.__p01[indice]
        else:
            return self.__p0

    def set(self, r, val):
        if r>1.-1E-11:
            self.__p1=val
        elif r>= 1E-11:
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


    def CalcForw1(self, probaR, z, CovZ, MeanZ):

        alpha = 0.
        MeanZ_alpha = InterLineaire_Vector(MeanZ, alpha)
        CovZ_alpha  = InterLineaire_Matrix(CovZ, alpha)
        self.__p0   = probaR(alpha) * multivariate_normal.pdf(z, mean=MeanZ_alpha, cov=CovZ_alpha)

        alpha = 1.
        MeanZ_alpha = InterLineaire_Vector(MeanZ, alpha)
        CovZ_alpha  = InterLineaire_Matrix(CovZ, alpha)
        self.__p1   = probaR(alpha) * multivariate_normal.pdf(z, mean=MeanZ_alpha, cov=CovZ_alpha)

        for i, alpha in enumerate(self.__Rcentres):
            MeanZ_alpha   = InterLineaire_Vector(MeanZ, alpha)
            CovZ_alpha    = InterLineaire_Matrix(CovZ, alpha)
            self.__p01[i] = probaR(alpha) * multivariate_normal.pdf(z, mean=MeanZ_alpha, cov=CovZ_alpha)
        
        self.normalisation(self.sum())
        # self.print()
        # print(self.sum())
        # input('pause - set1_1D')

    def setone_1D(self):
        for i, rnp1 in enumerate(self.__Rcentres):
            self.__p01[i] = 1.
        self.__p0 = 1.
        self.__p1 = 1.

    def CalcForB(self, FctCalculForB, probaForB, FS, z, CovZ, MeanZ):

        # les proba sont renvoyées non normalisées
        self.__p0 = FctCalculForB(0., self.__EPS, self.__STEPS, probaForB, FS, z, CovZ, MeanZ)
        for i, r in enumerate(self.__Rcentres):
            self.__p01[i] = FctCalculForB(r, self.__EPS, self.__STEPS, probaForB, FS, z, CovZ, MeanZ)
        self.__p1 = FctCalculForB(1., self.__EPS, self.__STEPS, probaForB, FS, z, CovZ, MeanZ)


    def nextAfterZeros(self):
        if self.__p0 < 1e-300:
            self.__p0 = 1e-300 #np.nextafter(0, 1)*10

        if self.__p1 < 1e-300:
            self.__p1 = 1e-300 #np.nextafter(0, 1)*10

        for i, rnp1 in enumerate(self.__Rcentres):
            if self.__p01[i] < 1e-300:
                self.__p01[i] = 1e-300 #np.nextafter(0, 1)*10

    def normalisation(self, norm):
        for i, rnp1 in enumerate(self.__Rcentres):
            self.__p01[i] /= norm
        self.__p0 /= norm
        self.__p1 /= norm

    def ProductFB(self, probaforw, probabackw):
        for i, r in enumerate(self.__Rcentres):
            self.__p01[i] = probaforw.__p01[i] * probabackw.__p01[i]
        self.__p0 = probaforw.__p0 * probabackw.__p0
        self.__p1 = probaforw.__p1 * probabackw.__p1


    # def fuzzyMPM(self):

    #     # select if hard or fuzzy
    #     hard = False
    #     if self.__p0 + self.__p1 >= 0.5:
    #         hard = True

    #     if hard == True: # its hard
    #         if self.__p0 > self.__p1:
    #             proba_max = self.__p0
    #             flevel_max = 0.
    #         else:
    #             proba_max = self.__p1
    #             flevel_max = 1.
    #     else: # its fuzzy
    #         if self.__STEPS != 0:
    #             proba_max = self.__p01.max()
    #             flevel_max = self.__Rcentres[self.__p01.argmax()]
    #         else:
    #             proba_max  = -1.
    #             flevel_max = -1.

    #     return flevel_max, proba_max


    # def fuzzyMPM2(self):

    #     array = np.array([self.__p0, self.__p1, 1. - self.__p0 - self.__p1])
    #     amax = np.argmax(array)
    #     # print('Array array = ', array)
    #     # print(amax)
    #     # input('pause')

    #     if amax == 0:
    #         proba_max  = self.__p0
    #         flevel_max = 0.
    #     elif amax == 1:
    #         proba_max  = self.__p1
    #         flevel_max = 1.
    #     else:
    #         proba_max  = self.__p01.max()
    #         flevel_max = self.__Rcentres[self.__p01.argmax()]

    #     return flevel_max, proba_max


    def sum(self):
        if self.__STEPS == 0:
            return self.__p0 + self.__p1
        else:
            return self.__p0 + self.__p1 + sum(self.__p01)/self.__STEPS

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
