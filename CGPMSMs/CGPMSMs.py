# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 11:17:15 2017

@author: Fay
"""

import numpy             as np
import matplotlib.pyplot as plt
import scipy             as sc

from CommonFun.CommonFun import From_FQ_to_Cov_Lyapunov, From_Cov_to_FQ
from CommonFun.CommonFun import Test_isCGOMSM_from_Cov



def simuCGPMSM(A, B, Cov, Mean_X, Mean_Y, MProba, TProba, N):

    n_r, n_x = np.shape(Mean_X)
    n_y      = np.shape(Mean_Y)[1]
    n_z      = n_x+n_y
    n_r_2    = n_r**2
    s_xz     = slice(n_x, n_z)
    s_0x     = slice(0,   n_x)
    s_0z     = slice(0,   n_z)

    # Le bruit
    W = np.random.normal(loc=0., scale=1., size=(n_z, N))

    # La chaine de Markov
    R = np.zeros(shape=(1, N))
    np1 = 0
    R[0, np1] = np.random.choice(a=n_r, size=1, p=MProba, replace=False)
    for np1 in range(1, N):
        R[0, np1] = np.random.choice(a=n_r, size=1, p=TProba[int(R[0, np1-1]), :], replace=False)


    # MProbaNum = np.zeros(shape=np.shape(MProba))
    # for np1 in range(N):
    #     MProbaNum[int(R[0, np1])]+=1.
    # MProbaNum /= N
    # print('Array MProbaNum = ', MProbaNum)
    # print('Array MProba = ', MProba)
    # input('pause')

    # JProbaNum = np.zeros(shape=np.shape(TProba))
    # for n in range(N-1):
    #     JProbaNum[int(R[0, n]), int(R[0, n+1])]+=1.
    # JProbaNum /= N
    # print('Array JProbaNum = ', JProbaNum)
    # input('pause')


    # Précalcul des moyennes
    Mean_Z_j = np.zeros(shape=(n_z, 1))
    Mean_Z_k = np.zeros(shape=(n_z, 1))
    N_xy     = np.zeros(shape=(n_r**2, n_z))
    for rn in range(n_r):
        for rnp1 in range(n_r):
            l = rn*n_r+rnp1
            Mean_Z_j   = np.hstack((Mean_X[rn,   :], Mean_Y[rn,   :]))
            Mean_Z_k   = np.hstack((Mean_X[rnp1, :], Mean_Y[rnp1, :]))
            N_xy[l, :] = Mean_Z_k - np.dot(A[l, :, :], Mean_Z_j)

    # Simulation de Z = [[X],[Y]]
    #######################################
    Z = np.zeros(shape=(n_z, N)) # Z = [[X],[Y]]

    # Le premier
    i = 0
    rn = int(R[0, i])
    l  = rn * n_r + 0
    Mean_Z_j = np.hstack((Mean_X[rn, :], Mean_Y[rn, :]))
    Z[:, i]  = np.random.multivariate_normal(mean=Mean_Z_j, cov=Cov[l, s_0z, s_0z], size=1)

    # les suivants
    for i in range(1, N):
        rnp1 = int(R[0, i])
        l    = rn*n_r + rnp1
        Z[:, i] = np.dot(A[l, :, :], Z[:, i-1]) + np.dot(B[l, :, :], W[:, i]) + N_xy[l, :]
        rn = rnp1

    X = Z[s_0x, :]
    Y = Z[s_xz, :]
    return X, R, Y


def GetParamNearestCGO_cov(Cov, n_x):

    Cov_CGO = np.copy(Cov)

    # Si c'est déjà un CGO, alors rien à faire
    if Test_isCGOMSM_from_Cov(Cov, n_x) == False:

        n_r_2, n_z_mp2, useless2 = np.shape(Cov)
        n_z  = n_z_mp2//2
        n_r  = int(np.sqrt(n_r_2))
        s_xz = slice(n_x, n_z)
        s_0x = slice(0,   n_x)
        s_0z = slice(0,   n_z)

        for l in range(n_r**2):
            Bj      = Cov[l, s_0x,   s_xz]
            GammaYY = Cov[l, s_xz, s_xz]
            Cjk     = Cov[l, s_xz, n_z+n_x:2*n_z]
            Cov_CGO[l, s_0x        , n_z+n_x:2*n_z] = np.dot( np.dot( Bj, np.transpose(np.linalg.inv(GammaYY))), Cjk)
            Cov_CGO[l, n_z+n_x:2*n_z, s_0x]         = Cov_CGO[l, s_0x, n_z+n_x:2*n_z].T

    return Cov_CGO


def GetBackCov_CGPMSM(Cov_CGPMSM_ORIG, n_x):

    n_r_2, n_z_2, useless2 = np.shape(Cov_CGPMSM_ORIG)
    n_z  = int(n_z_2/2)
    n_r  = int(np.sqrt(n_r_2))
    s_xz = slice(n_x, n_z)
    s_0x = slice(0,   n_x)
    s_0z = slice(0,   n_z)
    s_z2z = slice(n_z, 2*n_z)

    Cov_CGPMSM_REV = np.copy(Cov_CGPMSM_ORIG)
    for j in range(n_r):
        for k in range(n_r):
            l1 = j*n_r+k
            l2 = k*n_r+j

            # matrices block-diagonale (Sigma) : on inverse b_k et b_j
            # Cov_CGPMSM_REV[l1, s_0x, s_xz] = Cov_CGPMSM_ORIG[l1, slice(n_z, n_z+n_x), slice(n_z+n_x, 2*n_z)]
            # Cov_CGPMSM_REV[l1, slice(n_z, n_z+n_x), slice(n_z+n_x, 2*n_z)] = Cov_CGPMSM_ORIG[l1, s_0x, s_xz]

            # matrices block-diagonale (Sigma) : on inverse b_k et b_j
            
            # terme djk = ekj.T
            Cov_CGPMSM_REV[l1, s_0x, n_z+n_x:2*n_z] = np.transpose(Cov_CGPMSM_ORIG[l2, s_xz, n_z:n_z+n_x])
            # terme ejk = dkj.T
            Cov_CGPMSM_REV[l1, s_xz, n_z:n_z+n_x]   = np.transpose(Cov_CGPMSM_ORIG[l2, s_0x,   n_z+n_x:2*n_z])

            # Transposée du bloc (adec)
            Cov_CGPMSM_REV[l1, n_z:2*n_z, 0:n_z] = np.transpose(Cov_CGPMSM_REV[l1, 0:n_z, n_z:2*n_z])

    # print('CA VA ?Array Cov_CGPMSM_REV = \n', Cov_CGPMSM_REV)
    # input('pause')

    return Cov_CGPMSM_REV


def Adjust_Cov_to_OtherModel(Cov_Z1_Z2_dp_R1_R2, Model = 'CGOMSM'):  # only for 1 D, 2 classes
    n_r_2, n_z_2, useless2 = np.shape(Cov_Z1_Z2_dp_R1_R2)
    n_z = int(n_z_2/2)
    n_r = int(np.sqrt(n_r_2))

    Cov_Z1_Z2_dp_R1_R2_modi = np.copy(Cov_Z1_Z2_dp_R1_R2)
    assert n_z_multi2 == 4 and n_r ==2, print('only for 1 D, 2 classes')
    if Model is 'CGOMSM':
        for i in range(n_r**2):
            b       = Cov_Z1_Z2_dp_R1_R2[i, 0, 1]
            c       = Cov_Z1_Z2_dp_R1_R2[i, 1, 3]
            sigma_y = Cov_Z1_Z2_dp_R1_R2[i, 1, 1]
            Cov_Z1_Z2_dp_R1_R2_modi[i, 3, 0] = Cov_Z1_Z2_dp_R1_R2_modi[i, 0, 3] = b*c/sigma_y
#            Cov_Z1_Z2_dp_R1_R2_modi[i, 3, 0] = Cov_Z1_Z2_dp_R1_R2_modi[i, 0, 3]
    elif Model is 'CGLSSM':
        for i in range(n_r**2):
            a       = Cov_Z1_Z2_dp_R1_R2[i, 0, 2]
            bj      = Cov_Z1_Z2_dp_R1_R2[i, 0, 1]
            bk      = Cov_Z1_Z2_dp_R1_R2[i, 2, 3]
            sigma_x = Cov_Z1_Z2_dp_R1_R2[i, 0, 0]
            Cov_Z1_Z2_dp_R1_R2_modi[i,0,3] = Cov_Z1_Z2_dp_R1_R2_modi[i, 3, 0] = a*bk/sigma_x
            Cov_Z1_Z2_dp_R1_R2_modi[i,1,2] = Cov_Z1_Z2_dp_R1_R2_modi[i, 2, 1] = a*bj/sigma_x
            Cov_Z1_Z2_dp_R1_R2_modi[i,1,3] = Cov_Z1_Z2_dp_R1_R2_modi[i, 3, 1] = a*bj*bk/(sigma_x**2)
    else:
        raise TypeError('Model should be CGOMSM or CGLSSM')
    return (Cov_Z1_Z2_dp_R1_R2_modi)
