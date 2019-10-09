#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 11:04:45 2017

@author: Fay
"""

import math
import numpy as np
#from scipy.stats import norm

from CommonFun.CommonFun import From_Cov_to_FQ_bis, is_pos_def
from Fuzzy.InterFuzzy import InterBiLineaire_Matrix, InterLineaire_Vector


class RestorationPKF:
    """
    Pairwise Kalman filtering or smoothing for CGPMSM
    """

    def restore_withfuzzyjump(self, Y, R, Cov, Mean_X, Mean_Y, Likelihood=False, smooth=True):

        R = np.array(np.squeeze(R), ndmin=2)

        n_y, N   = np.shape(Y)
        n_r, n_x = np.shape(Mean_X)
        n_z      = n_x + n_y
        s_xz     = slice(n_x, n_z)
        s_0x     = slice(0,   n_x)
        # s_0z     = slice(0,   n_z)

        # State estimation arrays
        E_Xn_n     = np.zeros((n_x,      N))
        Cov_Xn_n   = np.zeros((n_x, n_x, N))
        E_Xn_np1   = np.zeros((n_x,      N))
        Cov_Xn_np1 = np.zeros((n_x, n_x, N))
        E_Xn_N     = np.zeros((n_x,      N))
        Cov_Xn_N   = np.zeros((n_x, n_x, N))
        C_n_np1_N  = np.zeros((n_x, n_x, N))

        E_Yn_np1   = np.zeros((n_y, N))
        M_znp1     = np.zeros((n_z, 1))
        M_zn       = np.zeros((n_z, 1))
        A_n        = np.zeros((n_x, n_x, N))

        likelihood = 0

        # ====================== Le premier ====================== #
        i = 0
        alpha = R[0, i]
        MeanX_alpha = InterLineaire_Vector(Mean_X, alpha)
        MeanY_alpha = InterLineaire_Vector(Mean_Y, alpha)
        Cov_alpha_0 = InterBiLineaire_Matrix(Cov, alpha, 0.)
        Temp = np.dot(Cov_alpha_0[s_0x, s_xz], np.linalg.inv(Cov_alpha_0[s_xz, s_xz]))
        E_Xn_n  [:,    i] = MeanX_alpha + np.dot(Temp, Y[:, i] -MeanY_alpha)
        Cov_Xn_n[:, :, i] = Cov_alpha_0[s_0x, s_0x] - np.dot(Temp, Cov_alpha_0[s_xz, s_0x])

        # ====================== Les suivants ====================== #
        for i in range(N-1):
            alpha = R[0, i]
            beta  = R[0, i+1]

            # interpolation of the covariance, then conversion to F, Q
            F_temp, Q_temp = From_Cov_to_FQ_bis(InterBiLineaire_Matrix(Cov, alpha, beta), n_z)

            # interpolation of the two means
            M_zn[s_0x,   0] = InterLineaire_Vector(Mean_X, alpha)
            M_zn[s_xz,   0] = InterLineaire_Vector(Mean_Y, alpha)
            M_znp1[s_0x, 0] = InterLineaire_Vector(Mean_X, beta)
            M_znp1[s_xz, 0] = InterLineaire_Vector(Mean_Y, beta)

            F_xx  = F_temp[s_0x, s_0x]
            F_xy  = F_temp[s_0x, s_xz]
            F_yx  = F_temp[s_xz, s_0x]
            F_yxT = F_yx.T
            F_yy  = F_temp[s_xz, s_xz]
            Q_xx  = Q_temp[s_0x, s_0x]
            Q_xy  = Q_temp[s_0x, s_xz]
            Q_yx  = Q_temp[s_xz, s_0x]
            Q_yy  = Q_temp[s_xz, s_xz]

            dot_Q_xy_Q_yy_inv = np.dot(Q_xy, np.linalg.inv(Q_yy))

            N_z = M_znp1 - np.dot(F_temp, M_zn)
            N_x = N_z[s_0x, 0]
            N_y = N_z[s_xz, 0]

            # ----------------- Filtering (Forward) ---------------- #
            A_n[:, :, i] = F_xx - np.dot(dot_Q_xy_Q_yy_inv, F_yx)
            Q2           = Q_xx - np.dot(dot_Q_xy_Q_yy_inv, Q_yx)

            S_n_np1      = Q_yy + np.dot(np.dot(F_yx, Cov_Xn_n[:, :, i]), F_yxT)
            S_n_np1_inv  = np.linalg.inv(S_n_np1)

            K_n_np1      = np.dot(np.dot(Cov_Xn_n[:, :, i], F_yxT), S_n_np1_inv)
            E_Yn_np1     = np.dot(F_yx, E_Xn_n[:, i]) + np.dot(F_yy, Y[:, i])+N_y

            E_Xn_np1 [:,    i] = E_Xn_n[:, i] + np.dot(K_n_np1, (Y[:, i+1] - E_Yn_np1))
            Cov_Xn_np1[:, :, i] = Cov_Xn_n[:, :, i] - np.dot(np.dot(K_n_np1, S_n_np1), np.transpose(K_n_np1))
            B_n = np.dot(dot_Q_xy_Q_yy_inv, Y[:, i+1]) + np.dot((F_xy - np.dot(dot_Q_xy_Q_yy_inv, F_yy)), Y[:, i]) + N_x - np.dot(dot_Q_xy_Q_yy_inv, N_y)

            # Filter X
            E_Xn_n  [:,    i+1] = np.dot(A_n[:, :, i], E_Xn_np1[:, i]) + B_n
            Cov_Xn_n[:, :, i+1] = Q2 + np.dot(np.dot(A_n[:, :, i], Cov_Xn_np1[:, :, i]), np.transpose(A_n[:, :, i]))

            if Likelihood is True:
                likelihood -= math.log(np.linalg.det(S_n_np1), math.e) - np.squeeze(np.dot(np.dot((Y[:, i+1] - E_Yn_np1)[np.newaxis], S_n_np1_inv), (Y[:, i+1] - E_Yn_np1)[np.newaxis].T))

        E_Xn_np1  [:,    N-1] = E_Xn_n  [:,    N-1]
        Cov_Xn_np1[:, :, N-1] = Cov_Xn_n[:, :, N-1]

        # ----------------- Smoothing (Backward) ----------------- #
        if smooth:
            E_Xn_N  [:,    N-1] = E_Xn_np1  [:,    N-1]
            Cov_Xn_N[:, :, N-1] = Cov_Xn_np1[:, :, N-1]

            for i in range(N-2, -1, -1):
                K_n_N             = np.dot(np.dot(Cov_Xn_np1[:, :, i], np.transpose(A_n[:, :, i])), np.linalg.inv(Cov_Xn_n[:, :, i+1]))
                E_Xn_N  [:,    i] = E_Xn_np1 [:,    i] + np.dot(K_n_N, (E_Xn_N[:, i+1] - E_Xn_n[:, i+1]))
                Cov_Xn_N[:, :, i] = Cov_Xn_np1[:, :, i] + np.dot(np.dot(K_n_N, (Cov_Xn_N[:, :, i +1] - Cov_Xn_n[:, :, i+1])), np.transpose(K_n_N))

                C_n_np1_N[:, :, i] = np.dot(K_n_N, Cov_Xn_N[:, :, i+1])

        if Likelihood is True:
            return E_Xn_n, Cov_Xn_n, E_Xn_N, Cov_Xn_N, likelihood

        return E_Xn_n, Cov_Xn_n, E_Xn_N, Cov_Xn_N


    def restore_withjump(self, Y, R, F, Q, Cov, Mean_X, Mean_Y, Likelihood=False):

        n_y, N   = np.shape(Y)
        n_r, n_x = np.shape(Mean_X)
        n_z      = n_x + n_y
        s_xz     = slice(n_x, n_z)
        s_0x     = slice(0,   n_x)
        s_0z     = slice(0,   n_z)

        # State estimation arrays
        E_Xn_n     = np.zeros((n_x,      N))
        Cov_Xn_n   = np.zeros((n_x, n_x, N))
        E_Xn_np1   = np.zeros((n_x,      N))
        Cov_Xn_np1 = np.zeros((n_x, n_x, N))
        E_Xn_N     = np.zeros((n_x,      N))
        Cov_Xn_N   = np.zeros((n_x, n_x, N))
        C_n_np1_N  = np.zeros((n_x, n_x, N))

        E_Yn_np1  = np.zeros((n_y, N))
        M_znp1    = np.zeros((n_z, 1))
        M_zn      = np.zeros((n_z, 1))
        A_n       = np.zeros((n_x, n_x, N))

        likelihood = 0

        # ====================== Le premier ====================== #
        i = 0
        rn = R[i]
        l  = rn*n_r+rn
        Temp = np.dot(Cov[l, s_0x, s_xz], np.linalg.inv(Cov[l, s_xz, s_xz]))
        E_Xn_n  [:,    i] = Mean_X[rn, :] + np.dot(Temp, Y[:, i] - Mean_Y[rn, :])
        Cov_Xn_n[:, :, i] = Cov[l, s_0x, s_0x] - np.dot(Temp, Cov[l, s_xz, s_0x])

        # ====================== Les suivants ====================== #
        for i in range(N-1):

            j = R[i]
            k = R[i+1]
            l = j * n_r + k
            F_temp          = F[l, :, :]
            Q_temp          = Q[l, :, :]
            M_zn[s_0x,   0] = Mean_X[j, :]
            M_zn[s_xz,   0] = Mean_Y[j, :]
            M_znp1[s_0x, 0] = Mean_X[k, :]
            M_znp1[s_xz, 0] = Mean_Y[k, :]

            F_xx  = F_temp[s_0x, s_0x]
            F_xy  = F_temp[s_0x, s_xz]
            F_yx  = F_temp[s_xz, s_0x]
            F_yxT = F_yx.T
            F_yy  = F_temp[s_xz, s_xz]
            Q_xx  = Q_temp[s_0x, s_0x]
            Q_xy  = Q_temp[s_0x, s_xz]
            Q_yx  = Q_temp[s_xz, s_0x]
            Q_yy  = Q_temp[s_xz, s_xz]

            dot_Q_xy_Q_yy_inv = np.dot(Q_xy, np.linalg.inv(Q_yy))

            N_z = M_znp1 - np.dot(F_temp, M_zn)
            N_x = N_z[s_0x, 0]
            N_y = N_z[s_xz, 0]

            # ----------------- Filtering (Forward) ---------------- #
            A_n[:, :, i] = F_xx - np.dot(dot_Q_xy_Q_yy_inv, F_yx)
            Q2           = Q_xx - np.dot(dot_Q_xy_Q_yy_inv, Q_yx)

            S_n_np1      = Q_yy + np.dot(np.dot(F_yx, Cov_Xn_n[:, :, i]), F_yxT)
            S_n_np1_inv  = np.linalg.inv(S_n_np1)

            K_n_np1      = np.dot(np.dot(Cov_Xn_n[:, :, i], F_yxT), S_n_np1_inv)
            E_Yn_np1     = np.dot(F_yx, E_Xn_n[:, i]) + np.dot(F_yy, Y[:, i])+N_y

            E_Xn_np1  [:,    i] = E_Xn_n  [:,    i] + np.dot(K_n_np1, (Y[:, i+1] - E_Yn_np1))
            Cov_Xn_np1[:, :, i] = Cov_Xn_n[:, :, i] - np.dot(np.dot(K_n_np1, S_n_np1), np.transpose(K_n_np1))
            B_n = np.dot(dot_Q_xy_Q_yy_inv, Y[:, i+1]) + np.dot((F_xy - np.dot(dot_Q_xy_Q_yy_inv, F_yy)), Y[:, i]) + N_x - np.dot(dot_Q_xy_Q_yy_inv, N_y)

            # Filter X
            E_Xn_n  [:,    i+1] = np.dot(A_n[:, :, i], E_Xn_np1[:, i]) + B_n
            Cov_Xn_n[:, :, i+1] = Q2 + np.dot(np.dot(A_n[:, :, i], Cov_Xn_np1[:, :, i]), np.transpose(A_n[:, :, i]))

            if Likelihood is True:
                likelihood -= math.log(np.linalg.det(S_n_np1), math.e) - np.squeeze(np.dot(np.dot((Y[:, i+1] - E_Yn_np1)[np.newaxis], S_n_np1_inv), (Y[:, i+1] - E_Yn_np1)[np.newaxis].T))

        # E_Xn_np1  [:,    N-1] = E_Xn_n  [:,    N-1]
        # Cov_Xn_np1[:, :, N-1] = Cov_Xn_n[:, :, N-1]

        # ----------------- Smoothing (Backward) ----------------- #
        E_Xn_N  [:,    N-1] = E_Xn_n  [:,    N-1]
        Cov_Xn_N[:, :, N-1] = Cov_Xn_n[:, :, N-1]

        for i in range(N-2, -1, -1):
            K_n_N             = np.dot(np.dot(Cov_Xn_np1[:, :, i], np.transpose(A_n[:, :, i])), np.linalg.inv(Cov_Xn_n[:, :, i+1]))
            E_Xn_N  [:,    i] = E_Xn_np1  [:,    i] + np.dot(K_n_N, (E_Xn_N[:, i+1] - E_Xn_n[:, i+1]))
            Cov_Xn_N[:, :, i] = Cov_Xn_np1[:, :, i] + np.dot(np.dot(K_n_N, (Cov_Xn_N[:, :, i +1] - Cov_Xn_n[:, :, i+1])), np.transpose(K_n_N))

            C_n_np1_N[:, :, i] = np.dot(K_n_N, Cov_Xn_N[:, :, i+1])

        if Likelihood is True:
            return E_Xn_n, Cov_Xn_n, E_Xn_N, Cov_Xn_N, likelihood

        return E_Xn_n, Cov_Xn_n, E_Xn_N, Cov_Xn_N




    def restore_withjump2(self, Y, R, F, Q, Cov, Mean_X, Mean_Y):

        n_y, N   = np.shape(Y)
        n_r, n_x = np.shape(Mean_X)
        n_z      = n_x + n_y
        s_xz     = slice(n_x, n_z)
        s_0x     = slice(0,   n_x)
        s_0z     = slice(0,   n_z)

        # State estimation arrays
        E_Xnp1_Ynp1   = np.zeros((n_z,      N))
        Cov_Xnp1_Ynp1 = np.zeros((n_z, n_z, N))
        E_Xn_n        = np.zeros((n_x,      N))
        Cov_Xn_n      = np.zeros((n_x, n_x, N))
        E_Xn_N        = np.zeros((n_x,      N))
        Cov_Xn_N      = np.zeros((n_x, n_x, N))
        
        E_Yn_np1   = np.zeros((n_y, N))
        M_znp1     = np.zeros((n_z, 1))
        M_zn       = np.zeros((n_z, 1))

        # ====================== Le premier ====================== #
        i = 0
        rn = R[i]
        l  = rn*n_r+rn
        Temp = np.dot(Cov[l, s_0x, s_xz], np.linalg.inv(Cov[l, s_xz, s_xz]))
        E_Xn_n  [:,    i] = Mean_X[rn, :] + np.dot(Temp, Y[:, i] - Mean_Y[rn, :])
        Cov_Xn_n[:, :, i] = Cov[l, s_0x, s_0x] - np.dot(Temp, Cov[l, s_xz, s_0x])

        # ====================== Les suivants ====================== #
        for i in range(N-1):

            j = R[i]
            k = R[i+1]
            l = j * n_r + k

            M_zn[s_0x,   0] = Mean_X[j, :]
            M_zn[s_xz,   0] = Mean_Y[j, :]
            M_znp1[s_0x, 0] = Mean_X[k, :]
            M_znp1[s_xz, 0] = Mean_Y[k, :]
            N_z = M_znp1 - np.dot(F[l, :, :], M_zn)

            Fzx = F[l, s_0z, s_0x]
            Fzy = F[l, s_0z, s_xz]

            E_Xnp1_Ynp1  [:,    i+1] = np.dot(Fzx, E_Xn_n[:, i]) + np.dot(Fzy, Y[:, i]) + np.reshape(N_z, newshape=(n_z))
            # print('shape of Fzx = ', np.shape(Fzx))
            # print('shape of Cov_Xn_n[:, :, i] = ', np.shape(Cov_Xn_n[:, :, i]))
            # input('pause')
            Cov_Xnp1_Ynp1[:, :, i+1] = np.dot(np.dot(Fzx, Cov_Xn_n[:, :, i]), np.transpose(Fzx)) + Q[l, :, :]
            
            # marginalisation --> filter
            Temp                = np.dot(Cov_Xnp1_Ynp1[s_0x, s_xz, i+1], np.linalg.inv(Cov_Xnp1_Ynp1[s_xz, s_xz, i+1]))
            E_Xn_n  [:,    i+1] = E_Xnp1_Ynp1[s_0x, i+1] + np.dot(Temp, Y[:, i+1] - E_Xnp1_Ynp1[s_xz, i+1])
            Cov_Xn_n[:, :, i+1] = Cov_Xnp1_Ynp1[s_0x, s_0x, i+1] - np.dot(Temp, Cov_Xnp1_Ynp1[s_xz, s_0x, i+1])


        # ----------------- Smoothing non-recursive algo ----------------- #
        # This algo is independent from the filtering

        if n_x>1 or n_y>1:
            print('This algorihtm only works for n_x=n_y=1!')
            exit(1)

        # Construct the Y cov matrix*******************************************
        VectY   = np.zeros((n_z, n_y))
        Sigma_Y = np.zeros((N, N))
        for k in range(N):
            # the term (k,k)
            j = 0
            l = R[k+j]*n_r+R[k+j]
            Sigma_Y[k+j, k] = Cov[l, n_x:n_z, n_x:n_z]

            # the term above k
            if k+1<N:

                j = 1
                l     = R[k+j-1]*n_r+R[k+j]
                VectY = Cov[l, n_z:2*n_z, n_x:n_z]
                Sigma_Y[k+j, k] = VectY[n_x:n_z, :]

                for j in range(2, N-k):
                    l     = R[k+j-1]*n_r+R[k+j]
                    VectY = np.dot(F[l, :, :], VectY)
                    Sigma_Y[k+j, k] = VectY[n_x:n_z, :]

        # completion of the matrices by transposition
        for p in range(1, N):
            for q in range(0, p):
                Sigma_Y [q, p] = np.transpose(Sigma_Y [p, q])


        # Construct the XY cov matrix******************************************
        VectYX   = np.zeros((n_z, n_x))
        Sigma_YX = np.zeros((N, N))
        for i in range(N):

            # the term (i,i)
            j = 0
            l = R[i+j]*n_r+R[i+j]
            Sigma_YX[i+j, i] = Cov[l, n_x:n_z, 0:n_x]

            # the term above i
            if i+1<N:

                j = 1
                l      = R[i+j-1]*n_r+R[i+j]
                VectYX = Cov[l, n_z:2*n_z, 0:n_x]
                Sigma_YX[i+j, i] = VectYX[n_x:n_z, :]

                for j in range(2, N-i):
                    l      = R[i+j-1]*n_r+R[i+j]
                    VectYX = np.dot(F[l, :, :], VectYX)
                    Sigma_YX[i+j, i] = VectYX[n_x:n_z, :]


            # the term below i
            if i>0:

                j = 1
                l      = R[i-j]*n_r+R[i-j+1]
                Sigma_YX[i-j, i] = Cov[l, n_x:n_z, n_z:n_z+n_x]

                for j in range(2, i+1):
                    l      = R[i-1]*n_r+R[i]
                    Sigma_YX[i-j, i] =  Sigma_YX[i-j, i-1] * np.transpose(F[l, 0:n_x, 0:n_x]) + \
                                        Sigma_Y [i-j, i-1] * np.transpose(F[l, 0:n_x, n_x:n_z])

        # # print('R = ', R)
        # print('Sigma_Y = \n', Sigma_Y)
        # print('Is Y mat cov ? --> ', is_pos_def(Sigma_Y))
        # print('Sigma_YX = \n', Sigma_YX)
        # ##Sigma_YX[0,1]=0.3 -----> c'est la bonne réponse!
        # print('Sigma_YX[0,1]=', Sigma_YX[0,1], ' - valuer attendue 3 !!!!!!!!!')
        # input('attente XY')

        # pour chacune des données
        invSigma_Y = np.linalg.inv(Sigma_Y) # to save time
        for i in range(N):

            l = R[i]*n_r+R[i]
            Temp               = np.dot(np.transpose(Sigma_YX[:, i]), invSigma_Y)
            E_Xn_N  [:,    i]  = Mean_X[R[i], :] + np.dot(Temp, np.transpose(Y) - Mean_Y[R, :])
            Cov_Xn_N[:, :, i]  = Cov[l, 0:n_x, 0:n_x] - np.dot(Temp, Sigma_YX[:, i])
            # print('i=', i)
            # print('  E_Xn_N  [:,    i]=', E_Xn_N  [:,    i])
            # print('  Cov_Xn_N[:, :, i]=', Cov_Xn_N[:, :, i])
            # input('pause smooth')

        return E_Xn_n, Cov_Xn_n, E_Xn_N, Cov_Xn_N
