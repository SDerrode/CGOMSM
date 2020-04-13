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
        self.__EPS      = EPS
        self.__STEPS    = STEPS
        self.__Rcentres = Rcentres
        if len(Rcentres) != self.__STEPS: input('PB constructeur Loi2DDiscreteFuzzy')

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


    def setr(self, r1, r2, val):
        
        if r1==1.:
            if r2==0.: self.__p10 = val
            elif r2==1.: self.__p11 = val
            else: self.__p10_11[math.floor(r2*self.__STEPS)]=val

        elif r1==0.:
            if r2==0.: self.__p00=val
            elif r2==1.: self.__p01=val
            else: self.__p01_00[math.floor(r2*self.__STEPS)]=val
        else:
            indr1 = math.floor(r1*self.__STEPS)
            if r2==0.: self.__p00_10[indr1]=val
            elif r2==1.: self.__p11_01[indr1]=val
            else: self.__p[indr1, math.floor(r2*self.__STEPS)]=val
    
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


    def partialInteg(self, minR1, maxR1, minR2, maxR2):

        integ = 0.
        tab1 = np.zeros(shape=(self.__STEPS))
        tab2 = np.zeros(shape=(self.__STEPS))
        
        # les arrêtes
        #### pour r1==0.
        r1 = 0.
        if r1>=minR1 and r1<=maxR1:
            for indr2, r2 in enumerate(self.__Rcentres):
                if r2>=minR2 and r2<=maxR2:
                    tab1[indr2] = self.__p01_00[indr2]
                else:
                    tab1[indr2] = 0.
            integ += np.sum(tab1)/self.__STEPS
            if minR2<=0.: integ += self.__p00
            if maxR2>=1.: integ += self.__p01

        #### pour r1==1.
        r1 = 1.
        if r1>=minR1 and r1<=maxR1:
            for indr2, r2 in enumerate(self.__Rcentres):
                if r2>=minR2 and r2<=maxR2:
                    tab1[indr2] = self.__p10_11[indr2]
                else:
                    tab1[indr2] = 0.
            integ += np.sum(tab1)/self.__STEPS
            if minR2<=0.: integ += self.__p10
            if maxR2>=1.: integ += self.__p11

        #### pour r1 in ]0,1[
        for indr1, r1 in enumerate(self.__Rcentres):
            if r1>=minR1 and r1<=maxR1:
                for indr2, r2 in enumerate(self.__Rcentres):
                    if r2>=minR2 and r2<=maxR2:
                        tab1[indr2] = self.__p[indr1, indr2]
                    else:
                        tab1[indr2] = 0.
                tab2[indr1] = np.sum(tab1)/self.__STEPS
                if minR2<=0.: tab2[indr1]  += self.__p00_10[indr1]
                if maxR2>=1.: tab2[indr1]  += self.__p11_01[indr1]
        integ +=  np.sum(tab2)/self.__STEPS
        
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
class Loi1DDiscreteFuzzy():

    def __init__(self, EPS, STEPS, Rcentres):
        self.__EPS      = EPS
        self.__STEPS    = STEPS
        self.__Rcentres = Rcentres
        if len(Rcentres) != self.__STEPS:
            input('PB constructeur Loi1DDiscreteFuzzy')

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

    def setr(self, r, val):
        if   r==0.: self.__p0=val
        elif r==1.: self.__p1=val
        else:       self.__p01[math.floor(r*self.__STEPS)]=val

    def setindr(self, indr, val):
        if   indr==0:              self.__p0=val
        elif indr==self.__STEPS+1: self.__p1=val
        else:                      self.__p01[indr-1]=val
 
    def print(self):
        print('__p0 = ', self.__p0)
        for i, rnp1 in enumerate(self.__Rcentres):
            print('  __p01[',rnp1, ']=', self.__p01[i])
        print('__p1 = ', self.__p1)

    # def nextAfterZeros(self):
 #        if self.__p0 < 1e-300:
 #            self.__p0 = 1e-300 #np.nextafter(0, 1)*10

 #        if self.__p1 < 1e-300:
 #            self.__p1 = 1e-300 #np.nextafter(0, 1)*10

 #        for i in range(self.__STEPS):
 #            if self.__p01[i] < 1e-300:
 #                self.__p01[i] = 1e-300 #np.nextafter(0, 1)*10

    def normalisation(self, norm, verbose=2):
        if norm != 0.:
            self.__p0  /= norm
            self.__p01 /= norm
            self.__p1  /= norm
        else:
            if verbose>2:
                print('ATTENTION : norm == 0.')

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