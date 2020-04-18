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


############################################################################################################
class Loi2DDiscreteFuzzy():

    def __init__(self, EPS, STEPS, Rcentres):
        self._EPS      = EPS
        self._STEPS    = STEPS
        self._Rcentres = Rcentres
        if len(Rcentres) != self._STEPS: input('PB constructeur Loi2DDiscreteFuzzy')

        self._p00 = 0.
        self._p10 = 0.
        self._p01 = 0.
        self._p11 = 0.
        
        if self._STEPS != 0:
            self._p00_10 = np.zeros(shape=(self._STEPS))
            self._p10_11 = np.zeros(shape=(self._STEPS))
            self._p11_01 = np.zeros(shape=(self._STEPS))
            self._p01_00 = np.zeros(shape=(self._STEPS))

            self._p = np.zeros(shape=(self._STEPS, self._STEPS))
        else:
            self._p00_10 = np.empty(shape=(0))
            self._p10_11 = np.empty(shape=(0))
            self._p11_01 = np.empty(shape=(0))
            self._p01_00 = np.empty(shape=(0))

            self._p = np.empty(shape=(0))


    def getr(self, r1, r2):
        
        if r1==1.:
            if r2==0.: return self._p10
            if r2==1.: return self._p11
            return self._p10_11[math.floor(r2*self._STEPS)]

        if r1==0.:
            if r2==0.: return self._p00
            if r2==1.: return self._p01
            return self._p01_00[math.floor(r2*self._STEPS)]
        
        indr1 = math.floor(r1*self._STEPS)
        if r2==0.: return self._p00_10[indr1]
        if r2==1.: return self._p11_01[indr1]
        return self._p[indr1, math.floor(r2*self._STEPS)]


    def setr(self, r1, r2, val):
        
        if r1==1.:
            if r2==0.: self._p10 = val
            elif r2==1.: self._p11 = val
            else: self._p10_11[math.floor(r2*self._STEPS)]=val

        elif r1==0.:
            if r2==0.: self._p00=val
            elif r2==1.: self._p01=val
            else: self._p01_00[math.floor(r2*self._STEPS)]=val
        else:
            indr1 = math.floor(r1*self._STEPS)
            if r2==0.: self._p00_10[indr1]=val
            elif r2==1.: self._p11_01[indr1]=val
            else: self._p[indr1, math.floor(r2*self._STEPS)]=val
    
    def getindr(self, indr1, indr2):
        
        if indr1==self._STEPS+1:
            if indr2==0:              return self._p10
            if indr2==self._STEPS+1: return self._p11
            return self._p10_11[indr2-1]  

        if indr1==0:
            if indr2==0:              return self._p00
            if indr2==self._STEPS+1: return self._p01
            return self._p01_00[indr2-1]
        
        if indr2==0:              return self._p00_10[indr1-1]
        if indr2==self._STEPS+1: return self._p11_01[indr1-1]
        return self._p[indr1-1, indr2-1]


    def Integ(self):

        if self._STEPS == 0:
            return self._p00 + self._p10 + self._p01 + self._p11

        #### pour r1==0.
        integ = np.mean(self._p01_00) + self._p00 + self._p01

        #### pour r1==1.
        integ += np.mean(self._p10_11) + self._p10 + self._p11

        #### La surface à l'intérieur
        pR = np.ndarray(shape=(self._STEPS))
        for j in range(self._STEPS):
            pR[j] = np.mean(self._p[j, :]) + self._p00_10[j] + self._p11_01[j]
        integ += np.mean(pR)

        return integ


    def partialInteg(self, minR1, maxR1, minR2, maxR2):

        integ = 0.
        tab1 = np.zeros(shape=(self._STEPS))
        tab2 = np.zeros(shape=(self._STEPS))
        
        # les arrêtes
        #### pour r1==0.
        r1 = 0.
        if r1>=minR1 and r1<=maxR1:
            for indr2, r2 in enumerate(self._Rcentres):
                if r2>=minR2 and r2<=maxR2:
                    tab1[indr2] = self._p01_00[indr2]
                else:
                    tab1[indr2] = 0.
            integ += np.sum(tab1)/self._STEPS
            if minR2<=0.: integ += self._p00
            if maxR2>=1.: integ += self._p01

        #### pour r1==1.
        r1 = 1.
        if r1>=minR1 and r1<=maxR1:
            for indr2, r2 in enumerate(self._Rcentres):
                if r2>=minR2 and r2<=maxR2:
                    tab1[indr2] = self._p10_11[indr2]
                else:
                    tab1[indr2] = 0.
            integ += np.sum(tab1)/self._STEPS
            if minR2<=0.: integ += self._p10
            if maxR2>=1.: integ += self._p11

        #### pour r1 in ]0,1[
        for indr1, r1 in enumerate(self._Rcentres):
            if r1>=minR1 and r1<=maxR1:
                for indr2, r2 in enumerate(self._Rcentres):
                    if r2>=minR2 and r2<=maxR2:
                        tab1[indr2] = self._p[indr1, indr2]
                    else:
                        tab1[indr2] = 0.
                tab2[indr1] = np.sum(tab1)/self._STEPS
                if minR2<=0.: tab2[indr1]  += self._p00_10[indr1]
                if maxR2>=1.: tab2[indr1]  += self._p11_01[indr1]
        integ +=  np.sum(tab2)/self._STEPS
        
        return integ

    def normalisation(self, norm):

        if norm != 0.:
            self._p00    /= norm
            self._p10    /= norm
            self._p01    /= norm
            self._p11    /= norm    
            
            self._p00_10 /= norm
            self._p10_11 /= norm
            self._p11_01 /= norm
            self._p01_00 /= norm 
            
            self._p      /= norm
        else:
            input('pb if norm == 0.')


    def print(self):

        print("Les coins:")
        print('  p01=', self._p01, ',  p11=', self._p11)
        print('  p00=', self._p00, ',  p10=', self._p10)

        if self._STEPS != 0:
            print("Les arêtes:")
            for i in range(self._STEPS):
                print(self._p00_10[i], end=' ')
            print("\n")
            for i in range(self._STEPS):
                print(self._p10_11[i], end=' ')
            print("\n")
            for i in range(self._STEPS):
                print(self._p11_01[i], end=' ')
            print("\n")
            for i in range(self._STEPS):
                print(self._p01_00[i], end=' ')

            print("\nLe coeur:")
            for indrn in range(self._STEPS):
                for indrnp1 in range(self._STEPS):
                    print(self._p[indrn, indrnp1], end=' ')
                print(" ")



############################################################################################################
class Loi1DDiscreteFuzzy():

    def __init__(self, EPS, STEPS, Rcentres):
        self._EPS      = EPS
        self._STEPS    = STEPS
        self._Rcentres = Rcentres
        if len(Rcentres) != self._STEPS:
            print(self._STEPS)
            print(len(Rcentres), np.shape(self._Rcentres))
            input('PB constructeur Loi1DDiscreteFuzzy')

        self._p0 = 0.
        if self._STEPS != 0:
            self._p01 = np.zeros(shape=(self._STEPS))
        else:
            self._p01 = np.empty(shape=(0,))
        self._p1 = 0.
    
    def ProductFB(self, loi1, loi2):
        self._p0 = loi1._p0 * loi2._p0
        for i in range(self._STEPS):
            self._p01[i] = loi1._p01[i] * loi2._p01[i]
        self._p1 = loi1._p1 * loi2._p1

    def getRcentre(self):
        return self._Rcentres

    def getSTEPS(self):
        return self._STEPS

    def getEPS(self):
        return self._EPS

    def getr(self, r):
        if r==0.: return self._p0
        if r==1.: return self._p1
        return self._p01[math.floor(r*self._STEPS)]

    def getindr(self, indr):
        if indr==0:             return self._p0
        if indr==self._STEPS+1: return self._p1
        return self._p01[indr-1]    

    def setr(self, r, val):
        if   r==0.: self._p0=val
        elif r==1.: self._p1=val
        else:       self._p01[math.floor(r*self._STEPS)]=val

    def setindr(self, indr, val):
        if   indr==0:              self._p0=val
        elif indr==self._STEPS+1: self._p1=val
        else:                      self._p01[indr-1]=val
 
    def print(self):
        print('__p0 = ', self._p0)
        for i, rnp1 in enumerate(self._Rcentres):
            print('  __p01[',rnp1, ']=', self._p01[i])
        print('__p1 = ', self._p1)


    def setValCste(self, val):
        self._p0 = val
        self._p1 = val
        for i, rnp1 in enumerate(self._Rcentres):
            self._p01[i] = val

    # def nextAfterZeros(self):
 #        if self._p0 < 1e-300:
 #            self._p0 = 1e-300 #np.nextafter(0, 1)*10

 #        if self._p1 < 1e-300:
 #            self._p1 = 1e-300 #np.nextafter(0, 1)*10

 #        for i in range(self._STEPS):
 #            if self._p01[i] < 1e-300:
 #                self._p01[i] = 1e-300 #np.nextafter(0, 1)*10

    def normalisation(self, norm, verbose=2):
        if norm != 0.:
            self._p0  /= norm
            self._p01 /= norm
            self._p1  /= norm
        else:
            if verbose>2:
                print('ATTENTION : norm == 0.')

    def TestIsAllZero(self):
        if self._p0 != 0.: return False
        if self._p1 != 0.: return False
        for ind in range(self._STEPS):
            if self._p01[ind] != 0.: return False
        return True

    def getSample(self):

        proba = np.array([self._p0, self._p1, 1. - (self._p0+self._p1)])
        typeSample = random.choices(population=[0, 1, 2], weights=proba)[0]
        if typeSample==0: 
            indr=0
        elif typeSample==1:
            indr=self._STEPS+1
        else: # it is fuzzy
            probaF = self._p01 / (proba[2]*self._STEPS)
            indr = random.choices(population=list(range(1, self._STEPS+1)), weights=probaF)[0]
        return indr


    def Integ(self):
        if self._STEPS == 0:
            return self._p0 + self._p1
        return self._p0 + self._p1 + np.mean(self._p01)