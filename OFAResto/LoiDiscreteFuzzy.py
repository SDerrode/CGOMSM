#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 11:35:45 2017

@author: Stephane
"""
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import norm

from Fuzzy.InterFuzzy import InterBiLineaire_Matrix, InterLineaire_Vector
from CommonFun.CommonFun import From_Cov_to_FQ_bis

def getindrnFromrn(STEPS, rn):
    if rn == 0:  return 0
    if rn == 1.: return STEPS+1
    return int(math.floor(rn*STEPS))

class Loi2DDiscreteFuzzy():

    def __init__(self, EPS, interpolation, STEPS, Rcentres, Mean_X, Mean_Y, Cov, dim1):
        self.__EPS      = EPS
        self.__STEPS    = STEPS
        self.__interpolation = interpolation
        self.__Rcentres = Rcentres
        self.__Cov      = Cov
        self.__Mean_X   = Mean_X
        self.__Mean_Y   = Mean_Y

        self.__n_x = np.shape(self.__Mean_X)[1]
        self.__n_y = np.shape(self.__Mean_Y)[1]
        self.__n_z = self.__n_x + self.__n_y

        # print('dim1=', dim1)
        self.__p00 = np.zeros(shape=dim1)
        self.__p10 = np.zeros(shape=dim1)
        self.__p01 = np.zeros(shape=dim1)
        self.__p11 = np.zeros(shape=dim1)
        dim2 = (STEPS,) + dim1
        # print('dim2=', dim2)
        # input('pause')
        self.__p00_01 = np.zeros(shape=dim2)
        self.__p01_11 = np.zeros(shape=dim2)
        self.__p11_10 = np.zeros(shape=dim2)
        self.__p10_00 = np.zeros(shape=dim2)
        dim3 = (STEPS, STEPS) + dim1
        self.__p = np.zeros(shape=dim3)

    def CovAQ(self, rn, rnp1):

        if self.__interpolation == True:
            Cov_rn_rnp1 = InterBiLineaire_Matrix(self.__Cov, rn, rnp1)
        else:
            indrn   = getindrnFromrn(self.__STEPS, rn)
            indrnp1 = getindrnFromrn(self.__STEPS, rnp1)
            Cov_rn_rnp1 = self.__Cov[indrn*self.__STEPS+indrnp1]

        return From_Cov_to_FQ_bis(Cov_rn_rnp1, self.__n_z)

    def Integ2D(self):

        if self.__STEPS == 0:
            return self.__p00 + self.__p10 + self.__p01 + self.__p11

        #### pour r1==0.
        integ = np.mean(self.__p00_01) + self.__p00 + self.__p01

        #### pour r1==1.
        integ += np.mean(self.__p11_10) + self.__p10 + self.__p11

        #### La surface à l'intérieur
        pR = np.ndarray(shape=(self.__STEPS))
        for j in range(self.__STEPS):
            pR[j] = np.mean(self.__p[j, :]) + self.__p01_11[j] + self.__p10_00[j]
        integ += np.mean(pR)

        return integ


    def fuzzyMPM_2D(self):

        loi = Loi1DDiscreteFuzzy(self.__EPS, self.__interpolation, self.__STEPS, self.__Rcentres)

        # pour r == 0.
        loi.set(0., np.mean(self.__p00_01) + self.__p00 + self.__p01)

        # pour r == 1.
        loi.set(1., np.mean(self.__p11_10) + self.__p10 + self.__p11)

        # pour l'intérieur
        for i, r in enumerate(self.__Rcentres):
            loi.set(r, np.mean(self.__p[i, :]) + self.__p01_11[i] + self.__p10_00[i])

        return loi.fuzzyMPM_1D()


    def fuzzyMPM2_2D(self):

        loi = Loi1DDiscreteFuzzy(self.__EPS, self.__interpolation, self.__STEPS, self.__Rcentres)

        # pour r == 0.
        loi.set(0., np.mean(self.__p00_01) + self.__p00 + self.__p01)

        # pour r == 1.
        loi.set(1., np.mean(self.__p11_10) + self.__p10 + self.__p11)

        # pour l'intérieur
        for i, r in enumerate(self.__Rcentres):
            loi.set(r, np.mean(self.__p[i, :]) + self.__p01_11[i] + self.__p10_00[i])

        return loi.fuzzyMPM2_1D()

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

    def Mean_Z(self, rn):

        if self.__interpolation == True:
            Mean_X_rn = InterLineaire_Vector(self.__Mean_X, rn)
            Mean_Y_rn = InterLineaire_Vector(self.__Mean_Y, rn)
        else:
            indrn     = getindrnFromrn(self.__STEPS, rn)
            Mean_X_rn = self.__Mean_X[indrn]
            Mean_Y_rn = self.__Mean_Y[indrn]
 
        MeanZ_inter=np.zeros(shape=(self.__n_z, 1))
        MeanZ_inter[0:self.__n_x         , 0] = Mean_X_rn
        MeanZ_inter[self.__n_x:self.__n_z, 0] = Mean_Y_rn
        return MeanZ_inter

    def get(self, r1, r2):
        i1 = math.floor(r1*self.__STEPS)
        i2 = math.floor(r2*self.__STEPS)
        if r1>1.-self.__EPS:
            if r2>1.-self.__EPS: return self.__p11
            elif r2 >= self.__EPS : return self.__p11_10[i2]
            else: return self.__p10
        elif r1 >= self.__EPS:
            if r2>1.-self.__EPS: return self.__p01_11[i1]
            elif r2 >= self.__EPS: return self.__p[i1,i2]
            else: return self.__p10_00[i1]
        else:
            if r2>1.-self.__EPS: return self.__p01
            elif r2 >= self.__EPS: return self.__p00_01[i2]
            else: return self.__p00

    def print(self):
        print('__p00 = ', self.__p00)
        print('__p10 = ', self.__p10)
        print('__p01 = ', self.__p01)
        print('__p11 = ', self.__p11)

        for i, r in enumerate(self.__Rcentres):
            print('__p00_01[',r, ']=', self.__p00_01[i])
            print('__p01_11[',r, ']=', self.__p01_11[i])
            print('__p11_10[',r, ']=', self.__p11_10[i])
            print('__p10_00[',r, ']=', self.__p10_00[i])
        
        for i, r1 in enumerate(self.__Rcentres):
            for j, r2 in enumerate(self.__Rcentres):
                print('__p=', self.__p[i, j, :, :])

    def test(self, E2, E):
        return E2 - np.dot(E, np.transpose(E))

    def test_VarianceNeg_2D(self, tab_E_Xnp1_dp2):

        OK= True
        if self.test(self.get(0., 0.), tab_E_Xnp1_dp2.get(0., 0.)) <0.: 
            print('A(0., 0.)=', self.test(self.get(0., 0.), tab_E_Xnp1_dp2.get(0., 0.))); OK = False
        if self.test(self.get(1., 0.), tab_E_Xnp1_dp2.get(1., 0.)) <0.: 
            print('A(1., 0.)=', self.test(self.get(1., 0.), tab_E_Xnp1_dp2.get(1., 0.))); OK = False
        if self.test(self.get(0., 1.), tab_E_Xnp1_dp2.get(0., 1.)) <0.: 
            print('A(0., 1.)=', self.test(self.get(0., 1.), tab_E_Xnp1_dp2.get(0., 1.))); OK = False
        if self.test(self.get(1., 1.), tab_E_Xnp1_dp2.get(1., 1.)) <0.: 
            print('A(1., 1.)=', self.test(self.get(1., 1.), tab_E_Xnp1_dp2.get(1., 1.))); OK = False

        for j, r in enumerate(self.__Rcentres):
            if self.test(self.get(0., r), tab_E_Xnp1_dp2.get(0., r)) <0.: 
                print('A(0., r)=', self.test(self.get(0., r), tab_E_Xnp1_dp2.get(0., r))); OK = False
            if self.test(self.get(r, 1.), tab_E_Xnp1_dp2.get(r, 1.)) <0.: 
                print('A(r, 1.)=', self.test(self.get(r, 1.), tab_E_Xnp1_dp2.get(r, 1.))); OK = False
            if self.test(self.get(1., r), tab_E_Xnp1_dp2.get(1., r)) <0.: 
                print('A(1., r)=', self.test(self.get(1., r), tab_E_Xnp1_dp2.get(1., r))); OK = False
            if self.test(self.get(r, 0.), tab_E_Xnp1_dp2.get(r, 0.)) <0.: 
                print('A(r, 0.)=', self.test(self.get(r, 0.), tab_E_Xnp1_dp2.get(r, 0.))); OK = False

        for i, r1 in enumerate(self.__Rcentres):
            for j, r2 in enumerate(self.__Rcentres):
                if self.test(self.get(r1, r2), tab_E_Xnp1_dp2.get(r1, r2)) <0.: 
                    print('A(r1, r2)=', self.test(self.get(r1, r2), tab_E_Xnp1_dp2.get(r1, r2))); OK = False

        return OK

    def test_VarianceNeg_2D_b(self):

        OK = True
        if self.get(0., 0.)[0,0] <0. or self.get(0., 0.)[1,1] <0.: 
            print('Ab(0., 0.)=', self.get(0., 0.)); OK = False
        if self.get(1., 0.)[0,0] <0. or self.get(1., 0.)[1,1] <0.: 
            print('Ab(1., 0.)=', self.get(1., 0.)); OK = False
        if self.get(0., 1.)[0,0] <0. or self.get(0., 1.)[1,1] <0.: 
            print('Ab(0., 1.)=', self.get(0., 1.)); OK = False
        if self.get(1., 1.)[0,0] <0. or self.get(1., 1.)[1,1] <0.: 
            print('Ab(1., 1.)=', self.get(1., 1.)); OK = False

        for j, r in enumerate(self.__Rcentres):
            if self.get(0., r)[0,0] <0. or self.get(0., r)[1,1] <0.: 
                print('Ab(0., r)=', self.get(0., r)); OK = False
            if self.get(r, 1.)[0,0] <0. or self.get(r, 1.)[1,1] <0.: 
                print('Ab(r, 1.)=', self.get(r, 1.)); OK = False
            if self.get(1., r)[0,0] <0. or self.get(1., r)[1,1] <0.: 
                print('Ab(1., r)=', self.get(1., r)); OK = False
            if self.get(r, 0.)[0,0] <0. or self.get(r, 0.)[1,1] <0.: 
                print('Ab(r, 0.)=', self.get(r, 0.)); OK = False

        for i, r1 in enumerate(self.__Rcentres):
            for j, r2 in enumerate(self.__Rcentres):
                if self.get(r1, r2)[0,0]<0. or self.get(r1, r2)[1,1]<0.: 
                    print('Ab(r1, r2)=', self.get(r1, r2)); OK = False
        
        return OK

    def set1c_2D(self, probaR2CondR1, ProbaForward_n):
        self.__p00 = probaR2CondR1(0., 0.) * ProbaForward_n.get(0.)
        self.__p10 = probaR2CondR1(1., 0.) * ProbaForward_n.get(1.)
        self.__p01 = probaR2CondR1(0., 1.) * ProbaForward_n.get(0.)
        self.__p11 = probaR2CondR1(1., 1.) * ProbaForward_n.get(1.)

        for j, r in enumerate(self.__Rcentres):
            self.__p00_01[j, :] = probaR2CondR1(0., r) * ProbaForward_n.get(0.)
            self.__p01_11[j, :] = probaR2CondR1(r, 1.) * ProbaForward_n.get(r)
            self.__p11_10[j, :] = probaR2CondR1(1., r) * ProbaForward_n.get(1.)
            self.__p10_00[j, :] = probaR2CondR1(r, 0.) * ProbaForward_n.get(r)

        for i, r1 in enumerate(self.__Rcentres):
            for j, r2 in enumerate(self.__Rcentres):
                self.__p[i, j, :, :] = probaR2CondR1(r1, r2) * ProbaForward_n.get(r1)


    def set1b_2D(self, ProbaForward_np1, ProbaForward_np, loijointeAP1, probaR2CondR1, yn, ynp1, np1):
        self.__p00 = loijointeAP1(0., 0., ProbaForward_np, probaR2CondR1, self.__Cov, yn, ynp1, self.__Mean_Y, np1, self.__interpolation, self.__STEPS) / ProbaForward_np1.get(0.)
        self.__p10 = loijointeAP1(1., 0., ProbaForward_np, probaR2CondR1, self.__Cov, yn, ynp1, self.__Mean_Y, np1, self.__interpolation, self.__STEPS) / ProbaForward_np1.get(0.)
        self.__p01 = loijointeAP1(0., 1., ProbaForward_np, probaR2CondR1, self.__Cov, yn, ynp1, self.__Mean_Y, np1, self.__interpolation, self.__STEPS) / ProbaForward_np1.get(1.)
        self.__p11 = loijointeAP1(1., 1., ProbaForward_np, probaR2CondR1, self.__Cov, yn, ynp1, self.__Mean_Y, np1, self.__interpolation, self.__STEPS) / ProbaForward_np1.get(1.)

        for j, r in enumerate(self.__Rcentres):
            self.__p00_01[j, :] = loijointeAP1(0., r, ProbaForward_np, probaR2CondR1, self.__Cov, yn, ynp1, self.__Mean_Y, np1, self.__interpolation, self.__STEPS) / ProbaForward_np1.get(r)
            self.__p01_11[j, :] = loijointeAP1(r, 1., ProbaForward_np, probaR2CondR1, self.__Cov, yn, ynp1, self.__Mean_Y, np1, self.__interpolation, self.__STEPS) / ProbaForward_np1.get(1.)
            self.__p11_10[j, :] = loijointeAP1(1., r, ProbaForward_np, probaR2CondR1, self.__Cov, yn, ynp1, self.__Mean_Y, np1, self.__interpolation, self.__STEPS) / ProbaForward_np1.get(r)
            self.__p10_00[j, :] = loijointeAP1(r, 0., ProbaForward_np, probaR2CondR1, self.__Cov, yn, ynp1, self.__Mean_Y, np1, self.__interpolation, self.__STEPS) / ProbaForward_np1.get(0.)

        for i, r1 in enumerate(self.__Rcentres):
            for j, r2 in enumerate(self.__Rcentres):
                self.__p[i, j, :, :] = loijointeAP1(r1, r2, ProbaForward_np, probaR2CondR1, self.__Cov, yn, ynp1, self.__Mean_Y, np1, self.__interpolation, self.__STEPS) / ProbaForward_np1.get(r2)

    def set1_a_2D(self, r1, r2, yn, Expect):
        # print('r1=', r1)
        # print('r2=', r2)
        # print('yn=', yn)
        # print('Expect=', Expect)
        A_r1_r2, useless     = self.CovAQ(r1, r2)
        N                    = self.Mean_Z(r2) - np.dot(A_r1_r2, self.Mean_Z(r1))
        E_Zn_dp_rnnp1_yun_yn = np.array([[Expect], [yn]])
        # print('A_r1_r2=', A_r1_r2)
        # print('N=', N)
        # print('E_Zn_dp_rnnp1_yun_yn=', E_Zn_dp_rnnp1_yun_yn)
        # input('pause')

        return np.dot(A_r1_r2, E_Zn_dp_rnnp1_yun_yn) + N

    def set1_2D(self, yn, tab_E_Xnp1_dp1):
        self.__p00 = self.set1_a_2D(0., 0., yn, tab_E_Xnp1_dp1.get(0.))
        self.__p10 = self.set1_a_2D(1., 0., yn, tab_E_Xnp1_dp1.get(1.))
        self.__p01 = self.set1_a_2D(0., 1., yn, tab_E_Xnp1_dp1.get(0.))
        self.__p11 = self.set1_a_2D(1., 1., yn, tab_E_Xnp1_dp1.get(1.))

        for j, r in enumerate(self.__Rcentres):
            # print('\nself.__p00_01[j, :]=\n', self.__p00_01[j, :])
            # print('self.set1_a_2D(0., r,  yn, tab_E_Xnp1_dp1.get(0.))=\n', self.set1_a_2D(0., r,  yn, tab_E_Xnp1_dp1.get(0.)))
            # input('pause')
            self.__p00_01[j, :] = self.set1_a_2D(0., r,  yn, tab_E_Xnp1_dp1.get(0.))
            self.__p01_11[j, :] = self.set1_a_2D(r,  1., yn, tab_E_Xnp1_dp1.get(r))
            self.__p11_10[j, :] = self.set1_a_2D(1., r,  yn, tab_E_Xnp1_dp1.get(1.))
            self.__p10_00[j, :] = self.set1_a_2D(r,  0., yn, tab_E_Xnp1_dp1.get(r))

        for i, r1 in enumerate(self.__Rcentres):
            for j, r2 in enumerate(self.__Rcentres):
                self.__p[i, j, :, :] = self.set1_a_2D(r1, r2, yn, tab_E_Xnp1_dp1.get(r1))

    def set33_a_2D(self, r1, r2, yn, Expect, Expect2):
        A_r1_r2, Q_r1_r2        = self.CovAQ(r1, r2)
        E_Zn_dp_rnnp1_yun_yn    = np.array([[Expect], [yn]])
        E_ZnZnT_dp_rnnp1_yun_yn = np.array([[Expect2, np.dot(Expect, np.transpose(yn)) ], [np.dot(yn, np.transpose(Expect)), np.dot(yn, np.transpose(yn))]])
        Var_Zn_dp_rnnp1_yun_yn  = E_ZnZnT_dp_rnnp1_yun_yn - np.dot(E_Zn_dp_rnnp1_yun_yn, np.transpose(E_Zn_dp_rnnp1_yun_yn))
        return np.dot(np.dot(A_r1_r2, Var_Zn_dp_rnnp1_yun_yn), np.transpose(A_r1_r2)) + Q_r1_r2

    def set33_2D(self, yn, tab_E_Xnp1_dp1, tab_E2_Xnp1_dp1):
        self.__p00 = self.set33_a_2D(0., 0., yn, tab_E_Xnp1_dp1.get(0.), tab_E2_Xnp1_dp1.get(0.))
        self.__p10 = self.set33_a_2D(1., 0., yn, tab_E_Xnp1_dp1.get(1.), tab_E2_Xnp1_dp1.get(1.))
        self.__p01 = self.set33_a_2D(0., 1., yn, tab_E_Xnp1_dp1.get(0.), tab_E2_Xnp1_dp1.get(0.))
        self.__p11 = self.set33_a_2D(1., 1., yn, tab_E_Xnp1_dp1.get(1.), tab_E2_Xnp1_dp1.get(1.))

        for j, r in enumerate(self.__Rcentres):
            self.__p00_01[j, :] = self.set33_a_2D(0., r,  yn, tab_E_Xnp1_dp1.get(0.), tab_E2_Xnp1_dp1.get(0.))
            self.__p01_11[j, :] = self.set33_a_2D(r,  1., yn, tab_E_Xnp1_dp1.get(r),  tab_E2_Xnp1_dp1.get(r))
            self.__p11_10[j, :] = self.set33_a_2D(1., r,  yn, tab_E_Xnp1_dp1.get(1.), tab_E2_Xnp1_dp1.get(1.))
            self.__p10_00[j, :] = self.set33_a_2D(r,  0., yn, tab_E_Xnp1_dp1.get(r),  tab_E2_Xnp1_dp1.get(r))

        for i, r1 in enumerate(self.__Rcentres):
            for j, r2 in enumerate(self.__Rcentres):
                self.__p[i, j, :, :] = self.set33_a_2D(r1, r2, yn, tab_E_Xnp1_dp1.get(r1), tab_E2_Xnp1_dp1.get(r1))

    def set4_a_2D(self, r1, r2, E_Znp1, VAR_Znp1, ynp1):
        result = E_Znp1[0] + VAR_Znp1[0, 1] / VAR_Znp1[1, 1] * (ynp1 - E_Znp1[1])
        if not np.isfinite(float(result[0])):
            print('E_Znp1[0]=', E_Znp1[0])
            print('delta=', delta)
            print('VAR_Znp1=', VAR_Znp1)
            input('attente set4_a')
        return float(result[0])

    def set4_2D(self, tab_E_Znp1, tab_VAR_Znp1, ynp1):

        self.__p00 = self.set4_a_2D(0., 0., tab_E_Znp1.get(0., 0.), tab_VAR_Znp1.get(0., 0.), ynp1)
        self.__p10 = self.set4_a_2D(1., 0., tab_E_Znp1.get(1., 0.), tab_VAR_Znp1.get(1., 0.), ynp1)
        self.__p01 = self.set4_a_2D(0., 1., tab_E_Znp1.get(0., 1.), tab_VAR_Znp1.get(0., 1.), ynp1)
        self.__p11 = self.set4_a_2D(1., 1., tab_E_Znp1.get(1., 1.), tab_VAR_Znp1.get(1., 1.), ynp1)

        for j, r in enumerate(self.__Rcentres):
            self.__p00_01[j] = self.set4_a_2D(0., r, tab_E_Znp1.get(0., r), tab_VAR_Znp1.get(0., r), ynp1)
            self.__p01_11[j] = self.set4_a_2D(r, 1., tab_E_Znp1.get(r, 1.), tab_VAR_Znp1.get(r, 1.), ynp1)
            self.__p11_10[j] = self.set4_a_2D(1., r, tab_E_Znp1.get(1., r), tab_VAR_Znp1.get(1., r), ynp1)
            self.__p10_00[j] = self.set4_a_2D(r, 0., tab_E_Znp1.get(r, 0.), tab_VAR_Znp1.get(r, 0.), ynp1)

        for i, r1 in enumerate(self.__Rcentres):
            for j, r2 in enumerate(self.__Rcentres):
                self.__p[i,j] = self.set4_a_2D(r1, r2, tab_E_Znp1.get(r1, r2), tab_VAR_Znp1.get(r1, r2), ynp1)

    def set5_a_2D(self, r1, r2, E_Xnp1_dp, VAR_Znp1):
        return E_Xnp1_dp*E_Xnp1_dp + VAR_Znp1[0, 0] - VAR_Znp1[0, 1] / VAR_Znp1[1, 1] * VAR_Znp1[1, 0]

    def set5_2D(self, tab_E_Xnp1_dp, tab_VAR_Znp1):

        self.__p00 = self.set5_a_2D(0., 0., tab_E_Xnp1_dp.get(0., 0.), tab_VAR_Znp1.get(0., 0.))
        self.__p10 = self.set5_a_2D(1., 0., tab_E_Xnp1_dp.get(1., 0.), tab_VAR_Znp1.get(1., 0.))
        self.__p01 = self.set5_a_2D(0., 1., tab_E_Xnp1_dp.get(0., 1.), tab_VAR_Znp1.get(0., 1.))
        self.__p11 = self.set5_a_2D(1., 1., tab_E_Xnp1_dp.get(1., 1.), tab_VAR_Znp1.get(1., 1.))

        for j, r in enumerate(self.__Rcentres):
            self.__p00_01[j] = self.set5_a_2D(0, r, tab_E_Xnp1_dp.get(0., r), tab_VAR_Znp1.get(0., r))
            self.__p01_11[j] = self.set5_a_2D(r, 1, tab_E_Xnp1_dp.get(r, 1.), tab_VAR_Znp1.get(r, 1.))
            self.__p11_10[j] = self.set5_a_2D(1, r, tab_E_Xnp1_dp.get(1., r), tab_VAR_Znp1.get(1., r))
            self.__p10_00[j] = self.set5_a_2D(r, 0, tab_E_Xnp1_dp.get(r, 0.), tab_VAR_Znp1.get(r, 0.))

        for i, r1 in enumerate(self.__Rcentres):
            for j, r2 in enumerate(self.__Rcentres):
                self.__p[i,j] = self.set5_a_2D(r1, r2, tab_E_Xnp1_dp.get(r1, r2), tab_VAR_Znp1.get(r1, r2))


class Loi1DDiscreteFuzzy():

    def __init__(self, EPS, interpolation, STEPS, Rcentres):

        self.__EPS      = EPS
        self.__STEPS    = STEPS
        self.__interpolation = interpolation
        self.__Rcentres = Rcentres
        if len(Rcentres) != self.__STEPS:
            input('PB constructeur Loi1DDiscreteFuzzy_TMC')

        self.__p0 = 0.
        if self.__STEPS != 0:
            self.__p01 = np.zeros(shape=(self.__STEPS))
        else:
            self.__p01 = np.empty(shape=(0,))
        self.__p1 = 0.

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
        if r>1.-self.__EPS:
            self.__p1=val
        elif r>= self.__EPS:
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

    def test_VarianceNeg_1D(self, tab_E_Xnp1_dp1):

        OK = True

        alpha = 0.
        if self.get(alpha) - tab_E_Xnp1_dp1.get(alpha)**2 <0.: 
            print('A(', alpha, ')=', self.test(self.get(alpha), tab_E_Xnp1_dp1.get(alpha)))
            print('E2= ', self.get(alpha), ', E = ', tab_E_Xnp1_dp1.get(alpha))
            OK = False


        alpha = 1.
        if self.get(alpha) - tab_E_Xnp1_dp1.get(alpha)**2 <0.: 
            print('A(', alpha, ')=', self.test(self.get(alpha), tab_E_Xnp1_dp1.get(alpha)))
            print('E2= ', self.get(alpha), ', E = ', tab_E_Xnp1_dp1.get(alpha))
            OK = False

        for j, alpha in enumerate(self.__Rcentres):
            if self.get(alpha) - tab_E_Xnp1_dp1.get(alpha)**2 <0.:
                print('A(', alpha, ')=', self.test(self.get(alpha), tab_E_Xnp1_dp1.get(alpha)))
                print('E2= ', self.get(alpha), ', E = ', tab_E_Xnp1_dp1.get(alpha))
                OK = False

        return OK

    def set1_1D(self, probaR, Cov, y, Mean_Y):
        
        rnp1, indrnp1 = 0., 0

        rn, indrn = 0., 0
        if self.__interpolation==True:
            Mean_Y_rn = InterLineaire_Vector(Mean_Y, rn)
            Cov_rn_0  = InterBiLineaire_Matrix(Cov, rn, rnp1)
        else:
            Mean_Y_rn = Mean_Y[indrn]
            Cov_rn_0  = Cov[indrn*self.__STEPS+indrnp1]
        self.__p0 = probaR(rn) * norm.pdf(y, loc=Mean_Y_rn, scale=np.sqrt(Cov_rn_0[1, 1])).item()

        rn, indrn = 1., self.__STEPS+1
        if self.__interpolation==True:
            Mean_Y_rn = InterLineaire_Vector(Mean_Y, rn)
            Cov_rn_0  = InterBiLineaire_Matrix(Cov, rn, rnp1)
        else:
            Mean_Y_rn = Mean_Y[indrn]
            Cov_rn_0  = Cov[indrn*self.__STEPS+indrnp1]
        self.__p1 = probaR(rn) * norm.pdf(y, loc=Mean_Y_rn, scale=np.sqrt(Cov_rn_0[1, 1])).item()

        for indrn, rn in enumerate(self.__Rcentres):
            if self.__interpolation==True:
                Mean_Y_rn = InterLineaire_Vector(Mean_Y, rn)
                Cov_rn_0  = InterBiLineaire_Matrix(Cov, rn, rnp1)
            else:
                Mean_Y_rn = Mean_Y[indrn+1]
                Cov_rn_0  = Cov[(indrn+1)*self.__STEPS+(indrnp1+1)]
            self.__p01[indrn] = probaR(rn) * norm.pdf(y, loc=Mean_Y_rn, scale=np.sqrt(Cov_rn_0[1, 1])).item()
        
        self.normalisation(self.Integ1D())

    def setone_1D(self):
        for i, rnp1 in enumerate(self.__Rcentres):
            self.__p01[i] = 1.
        self.__p0 = 1.
        self.__p1 = 1.
                      
    def set2_1D(self, fonction, loijointeAP, proba, probaR2CondR1, Cov, yn, ynp1, Mean_Y, np1):
        self.__p0 = fonction(0., self.__interpolation, self.__EPS, self.__STEPS, loijointeAP, proba, probaR2CondR1, Cov, yn, ynp1, Mean_Y, np1)
        for i, r in enumerate(self.__Rcentres):
            self.__p01[i] = fonction(r, self.__interpolation, self.__EPS, self.__STEPS, loijointeAP, proba, probaR2CondR1, Cov, yn, ynp1, Mean_Y, np1)
        self.__p1 = fonction(1., self.__interpolation, self.__EPS, self.__STEPS, loijointeAP, proba, probaR2CondR1, Cov, yn, ynp1, Mean_Y, np1)

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

    def set3a_1D(self, Mean_X, Mean_Y, Cov, ynp1):

        rnp1, indrnp1 = 0., 0

        rn, indrn = 0., 0
        if self.__interpolation==True:
            Mean_X_rn = InterLineaire_Vector(Mean_X, rn)
            Mean_Y_rn = InterLineaire_Vector(Mean_Y, rn)
            Cov_rn_0  = InterBiLineaire_Matrix(Cov, rn, rnp1)
        else:
            Mean_Y_rn = Mean_Y[indrn]
            Mean_X_rn = Mean_X[indrn]
            Cov_rn_0  = Cov[indrn*self.__STEPS+indrnp1]
        self.__p0 = Mean_X_rn + Cov_rn_0[0, 1] / Cov_rn_0[1, 1] * (ynp1 - Mean_Y_rn)


        rn, indrn = 1., self.__STEPS+1
        if self.__interpolation==True:
            Mean_X_rn = InterLineaire_Vector(Mean_X, rn)
            Mean_Y_rn = InterLineaire_Vector(Mean_Y, rn)
            Cov_rn_0  = InterBiLineaire_Matrix(Cov, rn, rnp1)
        else:
            Mean_Y_rn = Mean_Y[indrn]
            Mean_X_rn = Mean_X[indrn]
            Cov_rn_0  = Cov[indrn*self.__STEPS+indrnp1]
        self.__p1 = Mean_X_rn + Cov_rn_0[0, 1] / Cov_rn_0[1, 1] * (ynp1 - Mean_Y_rn)

        for indrn, rn in enumerate(self.__Rcentres):
            if self.__interpolation==True:
                Mean_X_rn = InterLineaire_Vector(Mean_X, rn)
                Mean_Y_rn = InterLineaire_Vector(Mean_Y, rn)
                Cov_rn_0  = InterBiLineaire_Matrix(Cov, rn, rnp1)
            else:
                Mean_Y_rn = Mean_Y[indrn]
                Mean_X_rn = Mean_X[indrn]
                Cov_rn_0  = Cov[indrn*self.__STEPS+indrnp1]
            self.__p01[indrn] = Mean_X_rn + Cov_rn_0[0, 1] / Cov_rn_0[1, 1] * (ynp1 - Mean_Y_rn)


    def set3b_1D(self, Mean_X, Mean_Y, Cov, ynp1, tab_E_Xnp1_dp1):

        rnp1, indrnp1 = 0., 0

        rn, indrn = 0., 0
        if self.__interpolation==True:
            Mean_X_rn = InterLineaire_Vector(Mean_X, rn)
            Mean_Y_rn = InterLineaire_Vector(Mean_Y, rn)
            Cov_rn_0  = InterBiLineaire_Matrix(Cov, rn, 0.)
        else:
            Mean_Y_rn = Mean_Y[indrn]
            Mean_X_rn = Mean_X[indrn]
            Cov_rn_0  = Cov[indrn*self.__STEPS+indrnp1]
        Var_n_n_rn = Cov_rn_0[0, 0] - Cov_rn_0[0, 1]*Cov_rn_0[0, 1] / Cov_rn_0[1, 1]
        self.__p0  = Var_n_n_rn + tab_E_Xnp1_dp1.get(rn)*tab_E_Xnp1_dp1.get(rn)

        rn, indrn = 1., self.__STEPS+1
        if self.__interpolation==True:
            Mean_X_rn = InterLineaire_Vector(Mean_X, rn)
            Mean_Y_rn = InterLineaire_Vector(Mean_Y, rn)
            Cov_rn_0  = InterBiLineaire_Matrix(Cov, rn, 0.)
        else:
            Mean_Y_rn = Mean_Y[indrn]
            Mean_X_rn = Mean_X[indrn]
            Cov_rn_0  = Cov[indrn*self.__STEPS+indrnp1]
        Var_n_n_rn = Cov_rn_0[0, 0] - Cov_rn_0[0, 1]*Cov_rn_0[0, 1] / Cov_rn_0[1, 1]
        self.__p1  = Var_n_n_rn + tab_E_Xnp1_dp1.get(rn)*tab_E_Xnp1_dp1.get(rn)

        for indrn, rn in enumerate(self.__Rcentres):
            if self.__interpolation==True:
                Mean_X_rn = InterLineaire_Vector(Mean_X, rn)
                Mean_Y_rn = InterLineaire_Vector(Mean_Y, rn)
                Cov_rn_0  = InterBiLineaire_Matrix(Cov, rn, 0.)
            else:
                Mean_Y_rn = Mean_Y[indrn]
                Mean_X_rn = Mean_X[indrn]
                Cov_rn_0  = Cov[indrn*self.__STEPS+indrnp1]
            Var_n_n_rn        = Cov_rn_0[0, 0] - Cov_rn_0[0, 1]*Cov_rn_0[0, 1] / Cov_rn_0[1, 1]
            self.__p01[indrn] = Var_n_n_rn + tab_E_Xnp1_dp1.get(rn)*tab_E_Xnp1_dp1.get(rn)


    def set4_1D (self, Integ_CalcE_X_np1_dp_rnpun, p_rn_d_rnpun_yun_ynpun, tab_E, np1):
        self.__p0 = Integ_CalcE_X_np1_dp_rnpun(self.__EPS, self.__STEPS, self.__Rcentres, 0., p_rn_d_rnpun_yun_ynpun, tab_E, np1)
        if np.isnan(self.__p0):
            print('set4 0 : ', self.__p0)
            input('Attente')
        
        for i, rnp1 in enumerate(self.__Rcentres):
            self.__p01[i] = Integ_CalcE_X_np1_dp_rnpun(self.__EPS, self.__STEPS, self.__Rcentres, rnp1, p_rn_d_rnpun_yun_ynpun, tab_E, np1)
            if np.isnan(self.__p01[i]):
                print('set4 ]0,1[: ', self.__p01[i])
                input('Attente')

        self.__p1 = Integ_CalcE_X_np1_dp_rnpun(self.__EPS, self.__STEPS, self.__Rcentres, 1., p_rn_d_rnpun_yun_ynpun, tab_E, np1)
        if np.isnan(self.__p1):
            print('set4 1 : ', self.__p1)
            input('Attente')

    def set6_1D(self, probaforw, probabackw):
        for i, rn in enumerate(self.__Rcentres):
            self.__p01[i] = probaforw.__p01[i] * probabackw.__p01[i]
        self.__p0 = probaforw.__p0 * probabackw.__p0
        self.__p1 = probaforw.__p1 * probabackw.__p1


    def fuzzyMPM_1D(self):

        # select if hard or fuzzy
        hard = False
        if self.__p0 + self.__p1 >= 0.5:
            hard = True

        if hard == True: # its hard
            if self.__p0 > self.__p1:
                proba_max = self.__p0
                flevel_max = 0.
            else:
                proba_max = self.__p1
                flevel_max = 1.
        else: # its fuzzy
            if self.__STEPS != 0:
                proba_max = self.__p01.max()
                flevel_max = self.__Rcentres[self.__p01.argmax()]
            else:
                proba_max  = -1.
                flevel_max = -1.

        return flevel_max, proba_max


    def fuzzyMPM2_1D(self):

        array = np.array([self.__p0, self.__p1, 1. - self.__p0 - self.__p1])
        amax = np.argmax(array)

        if amax == 0:
            proba_max  = self.__p0
            flevel_max = 0.
        elif amax == 1:
            proba_max  = self.__p1
            flevel_max = 1.
        else:
            proba_max  = self.__p01.max()
            flevel_max = self.__Rcentres[self.__p01.argmax()]

        # print('Array array = ', array)
        # print(amax, flevel_max, proba_max)
        # input('pause')
        return flevel_max, proba_max

    def Integ1D(self):
        if self.__STEPS == 0:
            return self.__p0 + self.__p1
        else:
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
