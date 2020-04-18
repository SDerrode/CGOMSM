#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 11:35:45 2017

@author: Fay
"""

import numpy as np
import scipy as sp
from scipy.stats import multivariate_normal, norm

from CommonFun.CommonFun import Test_isCGOMSM_from_F

def fonction(y, Cov_CGPMSM, Mean_X, Mean_Y, IProba, n_r, s_xz, s_0x):

    # print('Array y = ', y)
    num = 0.
    p_y = 0.
    for r in range(n_r):
        l = r*n_r+r
        SigX2 = Cov_CGPMSM[l, s_0x, s_0x]
        Br    = Cov_CGPMSM[l, s_0x, s_xz]
        SigY2 = Cov_CGPMSM[l, s_xz, s_xz]

        p_y_cd_j = norm.pdf(y, loc=Mean_Y[r, :], scale=np.sqrt(SigY2))
        # print('  Array p_y_cd_j = ', p_y_cd_j)
        p_j_y = IProba[r] * p_y_cd_j
        # print('  Array p_j_y = ', p_j_y)

        # Calcul de p(y)
        p_y += p_j_y
        # print('  Array p_y = ', p_y)

        # calcul de la moyenne conditionnelle
        moycond = Mean_X[r, :] + np.dot(Br, np.linalg.inv(SigY2)) * (y - Mean_Y[r, :])
        # print('  np.dot(Br, np.linalg.inv(SigY2))=', np.dot(Br, np.linalg.inv(SigY2)))
        # print('Array (y - Mean_Y[r]) = ', (y - Mean_Y[r]))
        # print('  Array moycond = ', moycond)

        num += p_j_y * moycond
        # print('  Array num = ', num)

    # input('pause function')
    if p_y == 0.:
        # print('num=', num)
        # print('p_y=', p_y)
        # input('pause')
        return 0.
    return num * num / p_y



class RestorationOFA:
    """
    Optimal filtering approximation
    """

    def Calcul_MajMSE_Aveugle(self, Cov_CGPMSM, Mean_X, Mean_Y, IProba, N, n_x, n_z, n_r):

        if n_x>1 or n_z>2:
            exit('Le calcul du majorant ne fonctionne que pour n_x=n_y=1 !!')

        s_xz = slice(n_x, n_z)
        s_0x = slice(0,   n_x)
        s_0z = slice(0,   n_z)

        Tab_MajMSE_Aveugle = np.zeros(shape=(2, n_x, N))

        # Majorant aveugle à sauts connus : A REVOIR
        ######################################################################
        MatCoeff_KJ= np.zeros(shape=(n_x, n_x))
        for r in range(n_r):
            l = r*n_r+r
            SigX2 = Cov_CGPMSM[l, s_0x, s_0x]
            Br    = Cov_CGPMSM[l, s_0x, s_xz]
            SigY2 = Cov_CGPMSM[l, s_xz, s_xz]
            BrT   = Cov_CGPMSM[l, s_xz, s_0x]

            MatCoeff_KJ += (SigX2 - np.dot(np.dot(Br, np.linalg.inv(SigY2)), BrT)) * IProba[r]

        # print('Array MatCoeff_KJ = ', MatCoeff_KJ)
        # on ne prend que les variances
        for dimx in range(n_x):
            Tab_MajMSE_Aveugle[0, dimx, :] = np.multiply(np.ones(N), MatCoeff_KJ[dimx, dimx])

        # Majorant aveugle à sauts inconnus
        ######################################################################
        
        # premier terme de la somme
        term1 = np.zeros(shape=(n_x, n_x))
        for r in range(n_r):
            l = r*n_r+r
            SigX2 = Cov_CGPMSM[l, s_0x, s_0x]
            term1 += IProba[r] * (SigX2 + np.dot(Mean_X[r, :], np.transpose(Mean_X[r, :])))

        argument = (Cov_CGPMSM, Mean_X, Mean_Y, IProba, n_r, s_xz, s_0x)
        term2, err = sp.integrate.quad(func=fonction, a=-50, b=50, args=argument, epsabs=1E-3, epsrel=1E-3, limit=200)

        # MSE unkown jumps
        MatCoeff_UJ = term1 - term2
        # print('Array MatCoeff_UJ = ', MatCoeff_UJ)
        for dimx in range(n_x):
            Tab_MajMSE_Aveugle[1, dimx, :] = np.multiply(np.ones(N), MatCoeff_UJ[dimx, dimx])

        return Tab_MajMSE_Aveugle


    def DFBSmoothing(self, E_Xn_dp_rn_y1_to_yn_DIR, E2_Xn_dp_rn_y1_to_yn_DIR, E_Xn_dp_rn_y1_to_yn_REV, E2_Xn_dp_rn_y1_to_yn_REV, E_Xn_dp_yn_rn_DIR, E2_Xn_dp_yn_rn_DIR, p_rn_dp_y1_to_yN_DIR, n_x):
        """"
            Le fameux nouveau lissage (double filtering based smoothing)
        """
        n_r, N = np.shape(p_rn_dp_y1_to_yN_DIR)

        E_X_OSA_NEW   = np.zeros(shape=(n_x, N))
        E2_X_OSA_NEW  = np.zeros(shape=(n_x, n_x, N))
        Cov_X_OSA_NEW = np.zeros(shape=(n_x, n_x, N))
        for n in range(N):
            for r in range(n_r):
                E_X_OSA_NEW [:,    n] += p_rn_dp_y1_to_yN_DIR[r, n] * (E_Xn_dp_rn_y1_to_yn_DIR [:,    r, n] + E_Xn_dp_rn_y1_to_yn_REV [ :,   r, n] - E_Xn_dp_yn_rn_DIR [:,    r, n])
                E2_X_OSA_NEW[:, :, n] += p_rn_dp_y1_to_yN_DIR[r, n] * (E2_Xn_dp_rn_y1_to_yn_DIR[:, :, r, n] + E2_Xn_dp_rn_y1_to_yn_REV[:, :, r, n] - E2_Xn_dp_yn_rn_DIR[:, :, r, n])

            Etemp = np.reshape(E_X_OSA_NEW [:, n], newshape=(n_x, 1))
            Cov_X_OSA_NEW[:, :, n] = E2_X_OSA_NEW[:, :, n] - np.dot(Etemp, np.transpose(Etemp))

        return E_X_OSA_NEW, Cov_X_OSA_NEW

    def Compute_E_E2_COV_Xn_dp_yn_rn(self, Y, COV_CGOMSM, Mean_X, Mean_Y):
        """
        Third term for the Double Filtering based Smoothing
        """
        N, n_y   = np.shape(Y)
        n_r, n_x = np.shape(Mean_X)
        n_z      = n_x + n_y
        s_xz     = slice(n_x, n_z)
        s_0x     = slice(0,   n_x)

        E_Xn_dp_yn_rn   = np.zeros(shape=(n_x,      n_r, N))
        COV_Xn_dp_yn_rn = np.zeros(shape=(n_x, n_x, n_r, N))
        E2_Xn_dp_yn_rn  = np.zeros(shape=(n_x, n_x, n_r, N))
        for n in range(N):
            for r in range(n_r):
                l = r*n_r+r
                Temp = np.dot(COV_CGOMSM[l, s_0x, s_xz], np.linalg.inv(COV_CGOMSM[l, s_xz, s_xz]))
                E_Xn_dp_yn_rn  [:,    r, n] = Mean_X[r, :] + np.dot(Temp, (Y[n, :] - Mean_Y[r, :]))
                COV_Xn_dp_yn_rn[:, :, r, n] = COV_CGOMSM[l, s_0x, s_0x] - np.dot(Temp, COV_CGOMSM[l, s_xz, s_0x])

                Etemp = np.reshape(E_Xn_dp_yn_rn[:, r, n], newshape=(n_x, 1))
                E2_Xn_dp_yn_rn[:, :, r, n] = COV_Xn_dp_yn_rn[:, :, r, n] + np.dot(Etemp, np.transpose(Etemp))

        return E_Xn_dp_yn_rn, E2_Xn_dp_yn_rn, COV_Xn_dp_yn_rn

    def restore_withoutjumps(self, Y, A, Q, COV_CGO, Mean_X, Mean_Y, TProba, IProba):

        N, n_y   = np.shape(Y)
        n_r, n_x = np.shape(Mean_X)
        n_z      = n_x + n_y
        n_r_2    = n_r**2
        s_xz     = slice(n_x, n_z)
        s_0x     = slice(0,   n_x)
        s_0z     = slice(0,   n_z)
        s_zzx    = slice(n_z, n_z+n_x)

        if Test_isCGOMSM_from_F(A, n_x, verbose=True) == False:
            exit('ATTENTION (restore_withoutjumps): ce nest pas un CGOMSM!!! --> IMPOSSIBLE\n')

        GGT = np.zeros(shape=(n_r_2, n_x, n_x)) # OK
        F   = np.zeros(shape=(n_r_2, n_x, n_x)) # OK
        J   = np.zeros(shape=(n_r_2, n_x, n_y)) # OK
        I   = np.zeros(shape=(n_r_2, n_x, n_y)) # OK
        H   = np.zeros(shape=(n_r_2, n_x, 1))   # OK

        for l in range(n_r_2):
            A[l, s_xz, s_0x] = 0. # be sure it is a CGOMSM (to zeroify very small values)

            J  [l, :, :] = np.dot(Q[l, s_0x, s_xz], sp.linalg.inv(Q[l, s_xz, s_xz]))
            I  [l, :, :] = A[l, s_0x, s_xz] - np.dot(J[l, :, :], A[l, s_xz, s_xz])
            GGT[l, :, :] = Q[l, s_0x, s_0x] - np.dot(J[l, :, :], Q[l, s_xz, s_0x])
            F  [l, :, :] = A[l, s_0x, s_0x]

        p_rn_dp_y1_to_yn     = np.zeros(shape=(N,   n_r     ))
        p_rn_dp_y1_to_yN     = np.zeros(shape=(N,   n_r     ))
        p_rn_rnp1_d_y1_ynp1  = np.zeros(shape=(n_r, n_r     ))
        # Alpha                = np.zeros(shape=(N,   n_r     ))   # = Beta Forward
        Beta                 = np.zeros(shape=(N,   n_r     ))   # = Beta Backward
        p_rnp1_ynp1_dp_rn_yn = np.zeros(shape=(N,   n_r, n_r))
        TAB_normalis         = np.zeros(shape=(N))

        E_Xn_dp_rn_y1_to_yn   = np.zeros(shape=(N, n_x,      n_r))
        E2_Xn_dp_rn_y1_to_yn  = np.zeros(shape=(N, n_x, n_x, n_r))
        COV_Xn_dp_rn_y1_to_yn = np.zeros(shape=(N, n_x, n_x, n_r))

        E_R_n   = np.zeros(shape=(N))
        E_R_N   = np.zeros(shape=(N))
        E_X_n   = np.zeros(shape=(N, n_x     ))
        E2_X_n  = np.zeros(shape=(N, n_x, n_x))
        COV_X_n = np.zeros(shape=(N, n_x, n_x))
        E_X_N   = np.zeros(shape=(N, n_x     ))
        E2_X_N  = np.zeros(shape=(N, n_x, n_x))
        COV_X_N = np.zeros(shape=(N, n_x, n_x))

        # Précalul des moyennes
        Mean_Z_n   = np.zeros(shape=(n_z))
        Mean_Z_np1 = np.zeros(shape=(n_z))
        N_xy  = np.zeros(shape=(n_r_2, n_z))
        N2_xy = np.zeros(shape=(n_r_2, n_x))
        for rn in range(n_r):
            Mean_Z_n = np.hstack((Mean_X[rn, :], Mean_Y[rn, :]))
            for rnp1 in range(n_r):
                l = rn*n_r+rnp1
                Mean_Z_np1  = np.hstack((Mean_X[rnp1, :], Mean_Y[rnp1, :]))
                N_xy[l, :]  = Mean_Z_np1 - np.dot(A[l, :, :], Mean_Z_n)
                N2_xy[l, :] = N_xy[l, s_0x] - np.dot(J[l, :, :], N_xy[l, s_xz])

        # --------------- Filtering ------------- #
        # --------------------------------------- #

        # ====================== Le premier ====================== #
        i = 0
        for rn in range(n_r):
            p_rn_dp_y1_to_yn[i, rn] = IProba[rn] * multivariate_normal.pdf(x=Y[i, :], mean=Mean_Y[rn, :], cov=COV_CGO[rn*n_r+rn, s_xz, s_xz])
        TAB_normalis[i] = np.sum(p_rn_dp_y1_to_yn[i, :])
        p_rn_dp_y1_to_yn[i, :] /= TAB_normalis[i]
        E_R_n[i] = np.argmax(p_rn_dp_y1_to_yn[i, :])

        for rn in range(n_r):
            l = rn*n_r+rn
            Temp = np.dot(COV_CGO[l, s_0x, s_xz], np.linalg.inv(COV_CGO[l, s_xz, s_xz]))
            E_Xn_dp_rn_y1_to_yn  [i, :,    rn] = Mean_X[rn, :] + np.dot(Temp, Y[i, :] - Mean_Y[rn, :])
            COV_Xn_dp_rn_y1_to_yn[i, :, :, rn] = COV_CGO[l, s_0x, s_0x] - np.dot(Temp, COV_CGO[l, s_xz, s_0x])

            Etemp = np.reshape(E_Xn_dp_rn_y1_to_yn[i, :, rn], newshape=(n_x, 1))
            E2_Xn_dp_rn_y1_to_yn [i, :, :, rn] = COV_Xn_dp_rn_y1_to_yn[i, :, :, rn] + np.dot(Etemp, np.transpose(Etemp))
            
            E_X_n  [i, :   ] += p_rn_dp_y1_to_yn[i, rn] * E_Xn_dp_rn_y1_to_yn  [i, :,    rn]
            E2_X_n [i, :, :] += p_rn_dp_y1_to_yn[i, rn] * E2_Xn_dp_rn_y1_to_yn [i, :, :, rn]

        Etemp = np.reshape(E_X_n [i, :], newshape=(n_x, 1))
        COV_X_n[i, :, :] = E2_X_n[i, :, :] - np.dot(Etemp, np.transpose(Etemp))


        # ====================== Les suivants ====================== #
        for i in range(1, N):

            for rn in range(n_r):

                # chaine normale CMC
                for rnp1 in range(n_r):
                    #loi p(yn, ynp1 | rn, rnp1)
                    # no need: loc = np.hstack((Mean_Y[rn, :], Mean_Y[rnp1, :]))
                    temp  = np.delete(np.delete(COV_CGO[rn*n_r+rnp1, :, :], s_zzx, axis=0), s_0x, axis=0)
                    scale = np.delete(np.delete(temp, s_zzx, axis=1), s_0x, axis=1)
                    # loi marginale p(ynp1 | rn, rnp1, yn)
                    Temp   = np.dot(scale[n_y:2*n_y, 0:n_y], np.linalg.inv(scale[0:n_y, 0:n_y]))
                    locmar = Mean_Y[rnp1, :] + np.dot(Temp, Y[i-1, :]-Mean_Y[rn, :])
                    scalemar = scale[n_y:2*n_y, n_y:2*n_y] - np.dot(Temp, scale[0:n_y, n_y:2*n_y])
                    p_rnp1_ynp1_dp_rn_yn[i, rn, rnp1] = multivariate_normal.pdf(Y[i, :], mean=locmar, cov=scalemar) * TProba[rn, rnp1]

            for rnp1 in range(n_r):
                p_rn_rnp1_d_y1_ynp1[:, rnp1] = p_rnp1_ynp1_dp_rn_yn[i, :, rnp1] * p_rn_dp_y1_to_yn[i-1, :]
                p_rn_dp_y1_to_yn[i, rnp1] = np.sum(p_rnp1_ynp1_dp_rn_yn[i, :, rnp1] * p_rn_dp_y1_to_yn[i-1, :])
            p_rn_rnp1_d_y1_ynp1 /= np.sum(p_rn_rnp1_d_y1_ynp1)

            TAB_normalis[i] = np.sum(p_rn_dp_y1_to_yn[i, :])
            p_rn_dp_y1_to_yn[i, :] /= TAB_normalis[i]
            E_R_n[i] = np.argmax(p_rn_dp_y1_to_yn[i, :])

            ############ LES ETATS
            for rnp1 in range(n_r):
                if p_rn_dp_y1_to_yn[i, rnp1]<1E-300:
                    # print('p_rn_dp_y1_to_yn[i, :]=', p_rn_dp_y1_to_yn[i, :])
                    # input('tab p_rn_dp_y1_to_yn < 1E-12')
                    p_rn_dp_y1_to_yn[i, rnp1]=1E-300
                    input('pause')

                for rn in range(n_r):
                    l = rn*n_r+rnp1

                    # Calcul de H
                    H[l, :, 0] = np.dot(I[l, :, :], Y[i-1, :]) + np.dot(J[l, :, :], Y[i, :]) + N2_xy[l, :]
                    
                    # calcul de p(rn | rnp1, yunnpun)
                    p_rn_d_rnp1_y1_ynp1 = p_rn_rnp1_d_y1_ynp1[rn, rnp1] / p_rn_dp_y1_to_yn[i, rnp1]

                    # calcul de l'espérance cond aux sauts
                    Ereshaped = np.reshape(E_Xn_dp_rn_y1_to_yn[i-1, :, rn], newshape=(n_x, 1))
                    E_Xn_dp_rn_y1_to_yn[i, :, rnp1] += p_rn_d_rnp1_y1_ynp1 * np.reshape(H[l, :, :] + np.dot(F[l, :, :], Ereshaped), newshape=(n_x))

                    # calcul de la cov. cond aux sauts
                    T1 = np.dot(np.dot(F[l, :, :], E2_Xn_dp_rn_y1_to_yn[i-1, :, :, rn]), np.transpose(F[l, :, :]))
                    FE = np.dot(F[l, :, :], Ereshaped)
                    T2 = np.dot(FE, np.transpose(H[l, :, :]))
                    T3 = np.dot(H[l, :, :], np.dot(np.transpose(Ereshaped), np.transpose(F[l, :, :])))
                    T5 = np.dot(H[l, :, :], np.transpose(H[l, :, :]))
                    E2_Xn_dp_rn_y1_to_yn[i, :, :, rnp1] += p_rn_d_rnp1_y1_ynp1 * (T1+T2+T3+GGT[l, :, :]+T5)

                Etemp = np.reshape(E_Xn_dp_rn_y1_to_yn[i, :, rnp1], newshape=(n_x, 1))
                COV_Xn_dp_rn_y1_to_yn[i, :, :, rnp1] = E2_Xn_dp_rn_y1_to_yn [i, :, :, rnp1] - np.dot(Etemp, np.transpose(Etemp))

            for rnp1 in range(n_r):
                E_X_n  [i, :,  ] += p_rn_dp_y1_to_yn[i, rnp1] * E_Xn_dp_rn_y1_to_yn  [i, :,    rnp1]
                E2_X_n [i, :, :] += p_rn_dp_y1_to_yn[i, rnp1] * E2_Xn_dp_rn_y1_to_yn [i, :, :, rnp1]
                if np.isnan(E_X_n[i, :]):
                    print('i=', i)
                    print('E_X_n[i, :]=', E_X_n[i, :])
                    input('pause')

            Etemp = np.reshape(E_X_n [i, :], newshape=(n_x, 1))
            COV_X_n[i, :, :] = E2_X_n[i, :, :] - np.dot(Etemp, np.transpose(Etemp))


        # --------------- Smoothing ------------- #
        # --------------------------------------- #

        # ====================== Le dernier ====================== #
        i = N-1
        Beta[i, :] = np.ones(n_r)
        p_rn_dp_y1_to_yN[i, :] = p_rn_dp_y1_to_yn[i, :] * Beta[i, :]
        # print('Array p_rn_dp_y1_to_yN[i, :] = ', p_rn_dp_y1_to_yN[i, :])
        # input('pause')

        # ====================== Les précédents ====================== #
        for i in range(N-2, -1, -1):
            for rn in range(n_r):
                Beta[i, rn] = np.sum(p_rnp1_ynp1_dp_rn_yn[i+1, rn, :] * Beta[i+1, :])
            Beta[i, :] /= TAB_normalis[i+1]

            p_rn_dp_y1_to_yN[i, :] = p_rn_dp_y1_to_yn[i, :] * Beta[i, :]
        
        for i in range(N):
            # selection du saut le plus probable (MPM)
            E_R_N[i]  = np.argmax(p_rn_dp_y1_to_yN[i, :])
            # calcul des esp et variance des etats
            for rnp1 in range(n_r):
                E_X_N [i, :,  ] += p_rn_dp_y1_to_yN[i, rnp1]  * E_Xn_dp_rn_y1_to_yn [i, :,    rnp1]
                E2_X_N[i, :, :] += p_rn_dp_y1_to_yN[i, rnp1]  * E2_Xn_dp_rn_y1_to_yn[i, :, :, rnp1]

            Etemp = np.reshape(E_X_N  [i, :], newshape=(n_x, 1))
            COV_X_N[i, :, :] = E2_X_N[i, :, :] - np.dot(Etemp, np.transpose(Etemp))

        return E_X_n, E2_X_n, COV_X_n, E_X_N, E2_X_N, COV_X_N, None, None, None, E_Xn_dp_rn_y1_to_yn, E2_Xn_dp_rn_y1_to_yn, COV_Xn_dp_rn_y1_to_yn, p_rn_dp_y1_to_yN, E_R_n, E_R_N, None


    def restore_MultiD(self, Y, A, Q, Mean_X, Mean_Y, TProba, IProba):
        N, n_y   = np.shape(Y)
        n_r, n_x = np.shape(Mean_X)
        n_z      = n_x + n_y
        n_r_2    = n_r**2
        s_xz     = slice(n_x, n_z)
        s_0x     = slice(0,   n_x)

        GGT = np.zeros(shape=(n_r_2, n_x, n_x))
        F   = np.zeros(shape=(n_r_2, n_x, n_x))
        J   = np.zeros(shape=(n_r_2, n_x, n_y))
        I   = np.zeros(shape=(n_r_2, n_x, n_y))
        H   = np.zeros(shape=(n_r_2, n_x))

        for j in range(n_r_2):
            A[j, s_xz, s_0x] = np.zeros(shape=(n_y, n_x))
            J[j, :, :] = np.dot(Q[j, s_0x, s_xz], np.linalg.inv(Q[j, s_xz, s_xz]))
            I[j, :, :] = A[j, s_0x, s_xz] - np.dot(J[j], A[j, s_xz, s_xz])
            GGT[j, :, :] = Q[j, s_0x, s_0x]-np.dot(J[j], Q[j, s_xz, s_0x])
            # CGPMSM
            F[j, :, :] = A[j, s_0x, s_0x]-np.dot(J[j, :, :], A[j, s_xz, s_0x])

        jump = np.reshape(TProba, (n_r_2, 1))

        # = Alpha Forward
        p_rn_dp_y1_to_yn = np.zeros(shape=(N, n_r))
        # = Beta Backward
        Beta = np.zeros(shape=(N, n_r))
        p_rn_dp_y1_to_yN = np.zeros(shape=(N, n_r))

        E_R_n = np.zeros(shape=(N))
        E_R_N = np.zeros(shape=(N))

        p_yn1_dp_rn_rnp1_yn   = np.zeros(shape=(n_r_2))
        p_rnp1_yn1_dp_rn_yn   = np.zeros(shape=(N-1, n_r_2))
        p_temp                = np.zeros(shape=(n_r_2))
        p_rn_rn1_dp_y1_to_yn1 = np.zeros(shape=(n_r_2))
        p_rn_dp_rn1_y1_to_yn1 = np.zeros(shape=(n_r_2))

        E_Xn_dp_rn_y1_to_yn   = np.zeros(shape=(N, n_r, n_x))
        Cov_Xn_dp_rn_y1_to_yn = np.zeros(shape=(N, n_r, n_x, n_x))
        E_X_n  = np.zeros(shape=(N, n_x))
        Cov_Xn = np.zeros(shape=(N, n_x, n_x))
        E_X_N  = np.zeros(shape=(N, n_x))
        Cov_XN = np.zeros(shape=(N, n_x, n_x))

        # p(rn/y(1~n))  # give the initial values
        p_rn_dp_y1_to_yn[0, :] = np.ones(n_r)/n_r
        Beta[N-1, :] = np.ones(n_r)


        for i in range(n_r):
            E_Xn_dp_rn_y1_to_yn  [0, i, :]    = np.transpose(Mean_X)
            Cov_Xn_dp_rn_y1_to_yn[0, i, :, :] = np.ones(n_x)
            E_X_n[0, :]     += p_rn_dp_y1_to_yn[0, i]*E_Xn_dp_rn_y1_to_yn    [0, i, :]
            Cov_Xn[0, :, :] += p_rn_dp_y1_to_yn[0, i] * Cov_Xn_dp_rn_y1_to_yn[0, i, :, :]

        N_xy    = np.zeros(shape=(N, n_r_2, n_z))
        R_value = np.linspace(0, n_r-1, n_r)

        for i in range(1, N):
            for l in range(n_r_2):
                # p(Yn+1/r(n~n+1), yn)
                p_yn1_dp_rn_rnp1_yn[l] = multivariate_normal.pdf(Y[i, :].T, (np.dot(A[l, s_xz, s_xz], (Y[i-1, :]-Mean_Y[l//n_r, :]))).T + np.transpose(Mean_Y[l%n_r, :]), Q[l, s_xz, s_xz])

                # p(rn+1,Yn+1/rn,yn)
                p_rnp1_yn1_dp_rn_yn[i-1, l] = jump[l]*p_yn1_dp_rn_rnp1_yn[l]
                p_temp[l]                   = p_rnp1_yn1_dp_rn_yn[i-1, l] * p_rn_dp_y1_to_yn[i-1, l//n_r]
                if p_temp[l] < 1e-300: p_temp[l] = 1e-300

            p_traverse_rn = np.sum(np.reshape(p_temp, (n_r, n_r)), axis=0)
            # p(yn+1/y(1~n))
            p_traverse_rn_rn1 = np.sum(p_traverse_rn)

            for l in range(n_r_2):
                N_xy[i, l, :] = np.hstack(((Mean_X[l%n_r, :] - np.dot(A[l, s_0x, s_0x], Mean_X[l//n_r, :])- np.dot(A[l, s_0x, s_xz], Mean_Y[l//n_r, :])), (Mean_Y[l%n_r, :]-np.dot(A[l, s_xz, s_0x], Mean_X[l//n_r, :])-np.dot(A[l, s_xz, s_xz], Mean_Y[l//n_r, :]))))

            for l in range(n_r_2):
                p_rn_rn1_dp_y1_to_yn1[l] = p_temp[l]/p_traverse_rn_rn1
                #p00/rn+1=0, rn=0,1  p01/rn+1=1, rn=0,1
                p_rn_dp_rn1_y1_to_yn1[l] = p_temp[l]/p_traverse_rn[(l%n_r)]
                H[l, :] = np.dot(J[l, :, :], (Y[i, :] - np.dot(A[l, s_xz, s_xz], Y[i-1, :]) - N_xy[i, l, s_xz, i])) + np.dot(A[l, s_0x, s_xz], Y[i-1, :])+N_xy[i, l, s_0x]

            # p(rn+1|yn+1)
            p_rn_dp_y1_to_yn[i, :] = np.sum(np.reshape(p_rn_rn1_dp_y1_to_yn1, (n_r, n_r)), axis=0)

            for j in range(n_r):
                l, q = 0, 0
                while j+l < n_r_2:
                    E_Xn_dp_rn_y1_to_yn[i, j, :] += p_rn_dp_rn1_y1_to_yn1[j+l] * (np.dot(F[j+l, :, :], E_Xn_dp_rn_y1_to_yn[i-1, q, :]) + H[j+l, :])

                    Cov_Xn_dp_rn_y1_to_yn[i, j, :, :] += p_rn_dp_rn1_y1_to_yn1[j+l]*(np.dot(np.dot(F[j+l, :, :], Cov_Xn_dp_rn_y1_to_yn[i-1, q, :, :]), np.transpose(F[j+l, :, :])) + \
                            np.dot(np.dot(F[j+l, :, :], E_Xn_dp_rn_y1_to_yn[i-1, q, :]), np.transpose(H[j+l, :])) + np.dot(np.dot(H[j+l, :], np.transpose(E_Xn_dp_rn_y1_to_yn[i-1, q, :])), \
                            np.transpose(F[j+l, :, :])) + GGT[j+l, :, :]+ np.dot(H[j+l, :], np.transpose(H[j+l, :])))
                    l = l+n_r
                    q += 1

            E_X_n[i, :]     = np.sum(p_rn_dp_y1_to_yn[i, :][np.newaxis].T             * E_Xn_dp_rn_y1_to_yn  [i, :, :], axis=0)
            Cov_Xn[i, :, :] = np.sum(p_rn_dp_y1_to_yn[i, :][np.newaxis][np.newaxis].T * Cov_Xn_dp_rn_y1_to_yn[i, :, :, :], axis=0)

            E_R_n[i, :] = R_value[p_rn_dp_y1_to_yn[i, :] == max(p_rn_dp_y1_to_yn[i, :])]

        # --------------- Smoothing ------------- #
        p_temp1 = np.zeros(shape=(n_r_2))

        for i in range(N-2, -1, -1):
            for l in range(n_r_2):
                j = l//n_r
                k = l%n_r
                p_temp[l]  = p_rnp1_yn1_dp_rn_yn[i, l] * Beta[i+1, k]
                p_temp1[l] = p_rnp1_yn1_dp_rn_yn[i, l] * p_rn_dp_y1_to_yn[i, j]

            p_traverse_rn     = np.sum(np.reshape(p_temp, (n_r, n_r)), axis=1)
            p_traverse_rn_rn1 = np.sum(p_temp1)  # p(yn+1/y(1~n))
            if p_traverse_rn_rn1 == 0.: p_traverse_rn_rn1 = 10**(-10)
            Beta [i, :]       = p_traverse_rn/p_traverse_rn_rn1

        p_rn_dp_y1_to_yN = p_rn_dp_y1_to_yn * Beta

        for i in range(N):
            E_X_N [i, :]    = np.sum(p_rn_dp_y1_to_yN[i, :][np.newaxis].T             * E_Xn_dp_rn_y1_to_yn[i, :, :], axis=0)
            Cov_XN[i, :, :] = np.sum(p_rn_dp_y1_to_yN[i, :][np.newaxis][np.newaxis].T * Cov_Xn_dp_rn_y1_to_yn[i, :, :, :], axis=0)
            E_R_N [i, :]    = R_value[p_rn_dp_y1_to_yN[i, :] == max(p_rn_dp_y1_to_yN[i, :])][0]

        return E_X_n, E_X_N, E_R_n, E_R_N
