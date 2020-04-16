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

from Fuzzy.InterFuzzy    import InterBiLineaire_Matrix, InterLineaire_Matrix, InterLineaire_Vector
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
    
############################################################################################################
class Tab2DDiscreteFuzzy():

    def __init__(self, EPS, STEPS, interpolation, Rcentres, dim):
        self._EPS           = EPS
        self._STEPS         = STEPS
        self._interpolation = interpolation
        self._Rcentres      = Rcentres
        self._dim           = dim
        if len(Rcentres) != self._STEPS: input('PB constructeur Tab2DDiscreteFuzzy')

        self._p00 = np.zeros(shape=dim)
        self._p10 = np.zeros(shape=dim)
        self._p01 = np.zeros(shape=dim)
        self._p11 = np.zeros(shape=dim)

        if self._STEPS != 0:
            
            dim1=(self._STEPS,) + dim
            self._p00_10 = np.zeros(shape=dim1)
            self._p10_11 = np.zeros(shape=dim1)
            self._p11_01 = np.zeros(shape=dim1)
            self._p01_00 = np.zeros(shape=dim1)

            dim2=(self._STEPS,self._STEPS,) + dim
            self._p = np.zeros(shape=dim2)
        else:
            self._p00_10 = np.empty(shape=dim)
            self._p10_11 = np.empty(shape=dim)
            self._p11_01 = np.empty(shape=dim)
            self._p01_00 = np.empty(shape=dim)

            self._p = np.empty(shape=dim)


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
            if indr2==0:             return self._p10
            if indr2==self._STEPS+1: return self._p11
            return self._p10_11[indr2-1]  

        if indr1==0:
            if indr2==0:             return self._p00
            if indr2==self._STEPS+1: return self._p01
            return self._p01_00[indr2-1]
        
        if indr2==0:             return self._p00_10[indr1-1]
        if indr2==self._STEPS+1: return self._p11_01[indr1-1]
        return self._p[indr1-1, indr2-1]

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

 
    def Mean_Z(self, Mean_X, Mean_Y, rn):

        n_x = np.shape(Mean_X)[1]
        n_y = np.shape(Mean_Y)[1]
        n_z = n_x + n_y

        if self._interpolation == True:
            Mean_X_rn = InterLineaire_Vector(Mean_X, rn)
            Mean_Y_rn = InterLineaire_Vector(Mean_Y, rn)
        else:
            indrn     = getindrnFromrn(self._STEPS, rn)
            Mean_X_rn = Mean_X[indrn]
            Mean_Y_rn = Mean_Y[indrn]
 
        MeanZ_inter=np.zeros(shape=(n_z, 1))
        MeanZ_inter[0:n_x  , 0] = Mean_X_rn
        MeanZ_inter[n_x:n_z, 0] = Mean_Y_rn
        return MeanZ_inter

    def CovAQ(self, Cov, rn, rnp1):

        n_z = np.shape(Cov)[2]//2
        if self._interpolation == True:
            Cov_rn_rnp1 = InterBiLineaire_Matrix(Cov, rn, rnp1)
        else:
            indrn       = getindrnFromrn(self._STEPS, rn)
            indrnp1     = getindrnFromrn(self._STEPS, rnp1)
            Cov_rn_rnp1 = Cov[indrn*self._STEPS+indrnp1]

        return From_Cov_to_FQ_bis(Cov_rn_rnp1, n_z)

    def set1_a_2D(self, Cov, Mean_X, Mean_Y, r1, r2, yn, Expect):
        n_x = np.shape(Mean_X)[1]
        n_y = np.shape(Mean_Y)[1]
        n_z = n_x + n_y
        A_r1_r2, useless     = self.CovAQ(Cov, r1, r2)
        N                    = self.Mean_Z( Mean_X, Mean_Y, r2) - np.dot(A_r1_r2, self.Mean_Z( Mean_X, Mean_Y, r1))
        E_Zn_dp_rnnp1_yun_yn = np.zeros(shape=(n_z, 1))
        E_Zn_dp_rnnp1_yun_yn[0:n_x] = np.reshape(Expect, newshape=(n_x))
        E_Zn_dp_rnnp1_yun_yn[n_x:]  = yn
        return np.dot(A_r1_r2, E_Zn_dp_rnnp1_yun_yn) + N

    def set1_2D(self, Cov, Mean_X, Mean_Y, yn, tab_E_Xnp1_dp1):
        self._p00 = self.set1_a_2D(Cov, Mean_X, Mean_Y, 0., 0., yn, tab_E_Xnp1_dp1.getr(0.))
        self._p10 = self.set1_a_2D(Cov, Mean_X, Mean_Y, 1., 0., yn, tab_E_Xnp1_dp1.getr(1.))
        self._p01 = self.set1_a_2D(Cov, Mean_X, Mean_Y, 0., 1., yn, tab_E_Xnp1_dp1.getr(0.))
        self._p11 = self.set1_a_2D(Cov, Mean_X, Mean_Y, 1., 1., yn, tab_E_Xnp1_dp1.getr(1.))

        for j, r in enumerate(self._Rcentres):
            self._p00_10[j] = self.set1_a_2D(Cov, Mean_X, Mean_Y, r, 0., yn, tab_E_Xnp1_dp1.getr(r))
            self._p10_11[j] = self.set1_a_2D(Cov, Mean_X, Mean_Y, 1., r, yn, tab_E_Xnp1_dp1.getr(1.))
            self._p11_01[j] = self.set1_a_2D(Cov, Mean_X, Mean_Y, r, 1., yn, tab_E_Xnp1_dp1.getr(r))
            self._p01_00[j] = self.set1_a_2D(Cov, Mean_X, Mean_Y, 0., r, yn, tab_E_Xnp1_dp1.getr(0.))

        for i, r1 in enumerate(self._Rcentres):
            for j, r2 in enumerate(self._Rcentres):
                self._p[i, j] = self.set1_a_2D(Cov, Mean_X, Mean_Y, r1, r2, yn, tab_E_Xnp1_dp1.getr(r1))

    def set33_a_2D(self, Cov, r1, r2, yn, Expect, Expect2):
        
        n_x = np.shape(Expect)[0]
        n_z = np.shape(Cov)[2]//2
        
        A_r1_r2, Q_r1_r2 = self.CovAQ(Cov, r1, r2)
        
        E_Zn_dp_rnnp1_yun_yn = np.zeros(shape=(n_z, 1))
        E_Zn_dp_rnnp1_yun_yn[0:n_x] = np.reshape(Expect, newshape=(n_x))
        E_Zn_dp_rnnp1_yun_yn[n_x:]  = yn
        
        E_ZnZnT_dp_rnnp1_yun_yn = np.zeros(shape=(n_z, n_z))
        E_ZnZnT_dp_rnnp1_yun_yn[0:n_x, 0:n_x] = np.reshape(Expect2, newshape=(n_x, n_x))
        E_ZnZnT_dp_rnnp1_yun_yn[n_x:,  0:n_x] = np.dot(yn, np.transpose(Expect))
        E_ZnZnT_dp_rnnp1_yun_yn[0:n_x, n_x:]  = np.dot(Expect, np.transpose(yn))
        E_ZnZnT_dp_rnnp1_yun_yn[n_x:, n_x:]   = np.dot(yn, np.transpose(yn))
            
        Var_Zn_dp_rnnp1_yun_yn  = E_ZnZnT_dp_rnnp1_yun_yn - np.outer(E_Zn_dp_rnnp1_yun_yn, E_Zn_dp_rnnp1_yun_yn)
        
        return np.dot(np.dot(A_r1_r2, Var_Zn_dp_rnnp1_yun_yn), np.transpose(A_r1_r2)) + Q_r1_r2

    def set33_2D(self, Cov, yn, tab_E_Xnp1_dp1, tab_E2_Xnp1_dp1):
        
        self._p00 = self.set33_a_2D(Cov, 0., 0., yn, tab_E_Xnp1_dp1.getr(0.), tab_E2_Xnp1_dp1.getr(0.))
        self._p10 = self.set33_a_2D(Cov, 1., 0., yn, tab_E_Xnp1_dp1.getr(1.), tab_E2_Xnp1_dp1.getr(1.))
        self._p01 = self.set33_a_2D(Cov, 0., 1., yn, tab_E_Xnp1_dp1.getr(0.), tab_E2_Xnp1_dp1.getr(0.))
        self._p11 = self.set33_a_2D(Cov, 1., 1., yn, tab_E_Xnp1_dp1.getr(1.), tab_E2_Xnp1_dp1.getr(1.))

        for j, r in enumerate(self._Rcentres):
            self._p00_10[j] = self.set33_a_2D(Cov, r, 0., yn, tab_E_Xnp1_dp1.getr(r),  tab_E2_Xnp1_dp1.getr(r))
            self._p10_11[j] = self.set33_a_2D(Cov, 1., r, yn, tab_E_Xnp1_dp1.getr(1.), tab_E2_Xnp1_dp1.getr(1.))
            self._p11_01[j] = self.set33_a_2D(Cov, r, 1., yn, tab_E_Xnp1_dp1.getr(r),  tab_E2_Xnp1_dp1.getr(r))
            self._p01_00[j] = self.set33_a_2D(Cov, 0., r, yn, tab_E_Xnp1_dp1.getr(0),  tab_E2_Xnp1_dp1.getr(0.))

        for i, r1 in enumerate(self._Rcentres):
            for j, r2 in enumerate(self._Rcentres):
                self._p[i, j] = self.set33_a_2D(Cov, r1, r2, yn, tab_E_Xnp1_dp1.getr(r1), tab_E2_Xnp1_dp1.getr(r1))

    def set4_a_2D(self, r1, r2, E_Znp1, VAR_Znp1, ynp1):
        n_y = np.shape(ynp1)[0]
        n_z = np.shape(E_Znp1)[0]
        n_x = n_z-n_y

        result = E_Znp1[0:n_x] + VAR_Znp1[0:n_x, n_x:] / VAR_Znp1[n_x:, n_x:] * (ynp1 - E_Znp1[n_x:])
        if not np.isfinite(float(result[0])):
            print('E_Znp1[0]=', E_Znp1[0])
            print('delta=', delta)
            print('VAR_Znp1=', VAR_Znp1)
            input('attente set4_a')
        return result

    def set4_2D(self, tab_E_Znp1, tab_VAR_Znp1, ynp1):

        self._p00 = self.set4_a_2D(0., 0., tab_E_Znp1.getr(0., 0.), tab_VAR_Znp1.getr(0., 0.), ynp1)
        self._p10 = self.set4_a_2D(1., 0., tab_E_Znp1.getr(1., 0.), tab_VAR_Znp1.getr(1., 0.), ynp1)
        self._p01 = self.set4_a_2D(0., 1., tab_E_Znp1.getr(0., 1.), tab_VAR_Znp1.getr(0., 1.), ynp1)
        self._p11 = self.set4_a_2D(1., 1., tab_E_Znp1.getr(1., 1.), tab_VAR_Znp1.getr(1., 1.), ynp1)

        for j, r in enumerate(self._Rcentres):
            self._p00_10[j] = self.set4_a_2D(r, 0., tab_E_Znp1.getr(r, 0.), tab_VAR_Znp1.getr(r, 0.), ynp1)
            self._p10_11[j] = self.set4_a_2D(1., r, tab_E_Znp1.getr(1., r), tab_VAR_Znp1.getr(1., r), ynp1)
            self._p11_01[j] = self.set4_a_2D(r, 1., tab_E_Znp1.getr(r, 1.), tab_VAR_Znp1.getr(r, 1.), ynp1)
            self._p01_00[j] = self.set4_a_2D(0., r, tab_E_Znp1.getr(0., r), tab_VAR_Znp1.getr(0., r), ynp1)

        for i, r1 in enumerate(self._Rcentres):
            for j, r2 in enumerate(self._Rcentres):
                self._p[i,j] = self.set4_a_2D(r1, r2, tab_E_Znp1.getr(r1, r2), tab_VAR_Znp1.getr(r1, r2), ynp1)

    def set5_a_2D(self, r1, r2, E_Xnp1_dp, VAR_Znp1):
        return E_Xnp1_dp*E_Xnp1_dp + VAR_Znp1[0, 0] - VAR_Znp1[0, 1] / VAR_Znp1[1, 1] * VAR_Znp1[1, 0]

    def set5_2D(self, tab_E_Xnp1_dp, tab_VAR_Znp1):

        self._p00 = self.set5_a_2D(0., 0., tab_E_Xnp1_dp.getr(0., 0.), tab_VAR_Znp1.getr(0., 0.))
        self._p10 = self.set5_a_2D(1., 0., tab_E_Xnp1_dp.getr(1., 0.), tab_VAR_Znp1.getr(1., 0.))
        self._p01 = self.set5_a_2D(0., 1., tab_E_Xnp1_dp.getr(0., 1.), tab_VAR_Znp1.getr(0., 1.))
        self._p11 = self.set5_a_2D(1., 1., tab_E_Xnp1_dp.getr(1., 1.), tab_VAR_Znp1.getr(1., 1.))

        for j, r in enumerate(self._Rcentres):
            self._p00_10[j] = self.set5_a_2D(r, 0., tab_E_Xnp1_dp.getr(r, 0.), tab_VAR_Znp1.getr(r, 0.))
            self._p10_11[j] = self.set5_a_2D(1., r, tab_E_Xnp1_dp.getr(1., r), tab_VAR_Znp1.getr(1., r))
            self._p11_01[j] = self.set5_a_2D(r, 1., tab_E_Xnp1_dp.getr(r, 1.), tab_VAR_Znp1.getr(r, 1.))
            self._p01_00[j] = self.set5_a_2D(0., r, tab_E_Xnp1_dp.getr(0., r), tab_VAR_Znp1.getr(0., r))

        for i, r1 in enumerate(self._Rcentres):
            for j, r2 in enumerate(self._Rcentres):
                self._p[i,j] = self.set5_a_2D(r1, r2, tab_E_Xnp1_dp.getr(r1, r2), tab_VAR_Znp1.getr(r1, r2))

    def Calc_GaussXY(self, M, Lambda2, P, Pi2, zn, znp1, n_x):

        xn, yn, xnpun, ynpun = zn[0:n_x], zn[n_x:], znp1[0:n_x], znp1[n_x:]

        # Pour les masses
        indrn   = 0
        indrnp1 = 0
        self._p00 = getGaussXY(M[indrn, indrnp1], Lambda2[indrn, indrnp1], P[indrn, indrnp1], Pi2[indrn, indrnp1], xn, yn, xnpun, ynpun)
        
        indrn   = self._STEPS+1
        indrnp1 = 0
        self._p10 = getGaussXY(M[indrn, indrnp1], Lambda2[indrn, indrnp1], P[indrn, indrnp1], Pi2[indrn, indrnp1], xn, yn, xnpun, ynpun)

        indrn   = self._STEPS+1
        indrnp1 = self._STEPS+1
        self._p11 = getGaussXY(M[indrn, indrnp1], Lambda2[indrn, indrnp1], P[indrn, indrnp1], Pi2[indrn, indrnp1], xn, yn, xnpun, ynpun)
        
        indrn   = 0
        indrnp1 = self._STEPS+1
        self._p01 = getGaussXY(M[indrn, indrnp1], Lambda2[indrn, indrnp1], P[indrn, indrnp1], Pi2[indrn, indrnp1], xn, yn, xnpun, ynpun)

        # Pour les arrètes et le coeur
        for ind in range(1, self._STEPS+1):
            
            indrnp1 = 0
            self._p00_10[ind-1] = getGaussXY(M[ind, indrnp1], Lambda2[ind, indrnp1], P[ind, indrnp1], Pi2[ind, indrnp1], xn, yn, xnpun, ynpun)

            indrn = self._STEPS+1
            self._p10_11[ind-1] = getGaussXY(M[indrn, ind], Lambda2[indrn, ind], P[indrn, ind], Pi2[indrn, ind], xn, yn, xnpun, ynpun)

            indrnp1 = self._STEPS+1
            self._p11_01[ind-1] = getGaussXY(M[ind, indrnp1], Lambda2[ind, indrnp1], P[ind, indrnp1], Pi2[ind, indrnp1], xn, yn, xnpun, ynpun)

            indrn = 0
            self._p01_00[ind-1] = getGaussXY(M[indrn, ind], Lambda2[indrn, ind], P[indrn, ind], Pi2[indrn, ind], xn, yn, xnpun, ynpun)

            # Pour l'intérieur
            for ind2 in range(1, self._STEPS+1):
                self._p[ind-1, ind2-1] = getGaussXY(M[ind, ind2], Lambda2[ind, ind2], P[ind, ind2], Pi2[ind, ind2], xn, yn, xnpun, ynpun)

############################################################################################################
class Tab1DDiscreteFuzzy():

    def __init__(self, EPS, STEPS, interpolation, Rcentres, dim):
        
        self._EPS           = EPS
        self._STEPS         = STEPS
        self._interpolation = interpolation
        self._Rcentres      = Rcentres
        self._dim           = dim
        if len(Rcentres) != self._STEPS:
            print(self._STEPS)
            print(len(Rcentres), np.shape(self._Rcentres))
            input('PB constructeur Tab1DDiscreteFuzzy')

        self._p0 = np.zeros(shape=dim)
        if self._STEPS != 0:
            dim1=(self._STEPS,) + dim
            self._p01 = np.zeros(shape=dim1)
        else:
            self._p01 = np.empty(shape=(0,))
        self._p1 = np.zeros(shape=dim)

    
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
        if indr==0:              return self._p0
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

    def set3a_1D(self, Mean_X, Mean_Y, Cov, ynp1):

        rnp1, indrnp1 = 0., 0

        rn, indrn = 0., 0
        if self._interpolation==True:
            Mean_X_rn = InterLineaire_Vector(Mean_X, rn)
            Mean_Y_rn = InterLineaire_Vector(Mean_Y, rn)
            Cov_rn_0  = InterBiLineaire_Matrix(Cov, rn, rnp1)
        else:
            Mean_Y_rn = Mean_Y[indrn]
            Mean_X_rn = Mean_X[indrn]
            Cov_rn_0  = Cov[indrn*self._STEPS+indrnp1]
        self._p0 = Mean_X_rn + Cov_rn_0[0, 1] / Cov_rn_0[1, 1] * (ynp1 - Mean_Y_rn)

        rn, indrn = 1., self._STEPS+1
        if self._interpolation==True:
            Mean_X_rn = InterLineaire_Vector(Mean_X, rn)
            Mean_Y_rn = InterLineaire_Vector(Mean_Y, rn)
            Cov_rn_0  = InterBiLineaire_Matrix(Cov, rn, rnp1)
        else:
            Mean_Y_rn = Mean_Y[indrn]
            Mean_X_rn = Mean_X[indrn]
            Cov_rn_0  = Cov[indrn*self._STEPS+indrnp1]
        self._p1 = Mean_X_rn + Cov_rn_0[0, 1] / Cov_rn_0[1, 1] * (ynp1 - Mean_Y_rn)

        for indrn, rn in enumerate(self._Rcentres):
            if self._interpolation==True:
                Mean_X_rn = InterLineaire_Vector(Mean_X, rn)
                Mean_Y_rn = InterLineaire_Vector(Mean_Y, rn)
                Cov_rn_0  = InterBiLineaire_Matrix(Cov, rn, rnp1)
            else:
                Mean_Y_rn = Mean_Y[indrn]
                Mean_X_rn = Mean_X[indrn]
                Cov_rn_0  = Cov[indrn*self._STEPS+indrnp1]
            self._p01[indrn] = Mean_X_rn + Cov_rn_0[0, 1] / Cov_rn_0[1, 1] * (ynp1 - Mean_Y_rn)


    def set3b_1D(self, Mean_X, Mean_Y, Cov, ynp1, tab_E_Xnp1_dp1):

        rnp1, indrnp1 = 0., 0

        rn, indrn = 0., 0
        if self._interpolation==True:
            Mean_X_rn = InterLineaire_Vector(Mean_X, rn)
            Mean_Y_rn = InterLineaire_Vector(Mean_Y, rn)
            Cov_rn_0  = InterBiLineaire_Matrix(Cov, rn, 0.)
        else:
            Mean_Y_rn = Mean_Y[indrn]
            Mean_X_rn = Mean_X[indrn]
            Cov_rn_0  = Cov[indrn*self._STEPS+indrnp1]
        Var_n_n_rn = Cov_rn_0[0, 0] - Cov_rn_0[0, 1]*Cov_rn_0[0, 1] / Cov_rn_0[1, 1]
        self._p0  = Var_n_n_rn + tab_E_Xnp1_dp1.getr(rn)*tab_E_Xnp1_dp1.getr(rn)

        rn, indrn = 1., self._STEPS+1
        if self._interpolation==True:
            Mean_X_rn = InterLineaire_Vector(Mean_X, rn)
            Mean_Y_rn = InterLineaire_Vector(Mean_Y, rn)
            Cov_rn_0  = InterBiLineaire_Matrix(Cov, rn, 0.)
        else:
            Mean_Y_rn = Mean_Y[indrn]
            Mean_X_rn = Mean_X[indrn]
            Cov_rn_0  = Cov[indrn*self._STEPS+indrnp1]
        Var_n_n_rn = Cov_rn_0[0, 0] - Cov_rn_0[0, 1]*Cov_rn_0[0, 1] / Cov_rn_0[1, 1]
        self._p1  = Var_n_n_rn + tab_E_Xnp1_dp1.getr(rn)*tab_E_Xnp1_dp1.getr(rn)

        for indrn, rn in enumerate(self._Rcentres):
            if self._interpolation==True:
                Mean_X_rn = InterLineaire_Vector(Mean_X, rn)
                Mean_Y_rn = InterLineaire_Vector(Mean_Y, rn)
                Cov_rn_0  = InterBiLineaire_Matrix(Cov, rn, 0.)
            else:
                Mean_Y_rn = Mean_Y[indrn]
                Mean_X_rn = Mean_X[indrn]
                Cov_rn_0  = Cov[indrn*self._STEPS+indrnp1]
            Var_n_n_rn        = Cov_rn_0[0, 0] - Cov_rn_0[0, 1]*Cov_rn_0[0, 1] / Cov_rn_0[1, 1]
            self._p01[indrn] = Var_n_n_rn + tab_E_Xnp1_dp1.getr(rn)*tab_E_Xnp1_dp1.getr(rn)


    def set4_1D (self, Integ_CalcE_X_np1_dp_rnpun, p_rn_d_rnpun_yun_ynpun, tab_E, np1):
        self._p0 = Integ_CalcE_X_np1_dp_rnpun(self._EPS, self._STEPS, self._Rcentres, 0., p_rn_d_rnpun_yun_ynpun, tab_E, np1)
        if np.isnan(self._p0):
            print('set4 0 : ', self._p0)
            input('Attente')
        
        for i, rnp1 in enumerate(self._Rcentres):
            self._p01[i] = Integ_CalcE_X_np1_dp_rnpun(self._EPS, self._STEPS, self._Rcentres, rnp1, p_rn_d_rnpun_yun_ynpun, tab_E, np1)
            if np.isnan(self._p01[i]):
                print('set4 ]0,1[: ', self._p01[i])
                input('Attente')

        self._p1 = Integ_CalcE_X_np1_dp_rnpun(self._EPS, self._STEPS, self._Rcentres, 1., p_rn_d_rnpun_yun_ynpun, tab_E, np1)
        if np.isnan(self._p1):
            print('set4 1 : ', self._p1)
            input('Attente')
