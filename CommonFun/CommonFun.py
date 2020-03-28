#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 15:28:34 2017

@author: Fay
"""

from collections import defaultdict
import numpy            as np
from   numpy import linalg as LA
import scipy            as sp
import scipy.cluster.vq as vq
from   scipy.stats import multivariate_normal


import warnings
import sys


def ImpressionMatLatex(Mat, ch, n_r, M, decim=3, file=sys.stdin):
    print('\\begin{eqnarray*}', file=file)
    for i in range(n_r**2):
        j = i//n_r
        k = i%n_r
        print('&\\bs{%s}_{%d,%d}^{\\zundeux} &= \\begin{bmatrix}'%(ch,j,k), file=file)
        for l in range(M):
            for m in range(M):
                print(np.around(Mat[i,l,m], decimals=decim), end=' & ', file=file)
            print('\\\\', file=file)
        print('\\end{bmatrix}', file=file)
    print('\\end{eqnarray*}', file=file)


def Test_isCGOMSM_from_F(F, n_x, tol=1E-8, verbose=False):

    n_r_2, n_z = np.shape(F)[0:2]

    Ok    = True
    for l in range(n_r_2):
        for z1 in range(n_x, n_z):
            for z2 in range(n_x):
                if F[l, z1, z2] > tol:
                    if verbose==True:
                        print('l=', l)
                        print('F[l, :, :]=', F[l, :, :])
                    Ok = False
                    break
    return Ok


def Test_isCGOMSM_from_Cov(Cov, n_x, tol=1E-8, verbose=False):

    n_r_2, n_z_mp2, useless2 = np.shape(Cov)
    n_z = n_z_mp2//2
    n_r = int(np.sqrt(n_r_2))

    Ok = True
    for l in range(n_r**2):
        Djk     = Cov[l, 0:n_x, n_z+n_x:2*n_z]
        Bj      = Cov[l, 0:n_x,   n_x:n_z]
        GammaYY = Cov[l, n_x:n_z, n_x:n_z]
        Cjk     = Cov[l, n_x:n_z, n_z+n_x:2*n_z]
        SOL     = Djk - np.dot( np.dot( Bj, np.transpose(np.linalg.inv(GammaYY))), Cjk)

        for z1 in range(np.shape(SOL)[0]):
            for z2 in range(np.shape(SOL)[1]):
                if SOL[z1, z2] > tol:
                    if verbose==True:
                        print('l=', l)
                        print('Cov[l, :, :]=', Cov[l, :, :])
                        print('SOL=', SOL)
                    Ok = False
                    break
    return Ok

def From_Cov_to_FQ(Cov_Z1_Z2_dp_R1_R2):
    """
    Convert a cov matrix of the second parametrization of CGPMSM into a couple (F,Q) of the first parametrization.
    """
    n_r_2, n_z_2 = np.shape(Cov_Z1_Z2_dp_R1_R2)[0:2]
    n_z          = int(n_z_2/2)

    F = np.zeros(shape=(n_r_2, n_z, n_z))
    Q = np.zeros(shape=(n_r_2, n_z, n_z))
    for i in range(n_r_2):
        F[i,:,:], Q[i,:,:] = From_Cov_to_FQ_bis(Cov_Z1_Z2_dp_R1_R2[i,:,:], n_z)
        print('Q[i,:,:]=', Q[i,:,:])
        if is_pos_def(Q[i,:,:]) == False:
            print('i=', i, ' --> PROBLEM with Q matrix in From_Cov_to_FQ!!')
            print(Q[i,:,:])
            input('pause in From_Cov_to_FQ')

    return F, Q

def From_Cov_to_FQ_bis(Cov_Z1_Z2_dp_R1_R2, n_z):
    """
    Convert a cov matrix of the second parametrization of CGPMSM into a couple (F,Q) of the first parametrization.
    """

    Ct_zn_zn     = Cov_Z1_Z2_dp_R1_R2[0:n_z,     0:n_z]
    Ct_znp1_znp1 = Cov_Z1_Z2_dp_R1_R2[n_z:2*n_z, n_z:2*n_z]
    Ct_zn_znp1   = Cov_Z1_Z2_dp_R1_R2[0:n_z,     n_z:2*n_z]

    F = np.dot(np.transpose(Ct_zn_znp1), np.linalg.inv(Ct_zn_zn))
    Q = Ct_znp1_znp1 - np.dot(F, Ct_zn_znp1)

    return F, Q

def From_FQ_to_Cov_Lyapunov(F, Q, n_x):
    n_r_2, n_z, n_z = np.shape(F)
    n_y = n_z - n_x
    n_r = int(np.sqrt(n_r_2))
    print('n_r=', n_r)
    print('n_z=', n_z)
    print('n_y=', n_y)
    print('n_x=', n_x)

    Gamma = np.zeros(shape=(n_r, n_z, n_z))
    SigmaT = np.zeros(shape=(n_r_2, n_z, n_z))
    Cov_Z1_Z2_dp_R1_R2 = np.zeros(shape=(n_r_2, n_z*2,n_z*2))
    
    for i in range(n_r):
        # temp = np.dot( np.linalg.inv(np.eye(n_z**2) - np.kron(F[i*n_r+i,:,:], F[i*n_r+i,:,:])), np.reshape(Q[i*n_r+i,:,:], (n_z**2, 1), order='F') )
        # Gamma[i,:,:] = np.reshape(temp, (n_z, n_z), order='F')
        # print('Fei: Gamma[i,:,:]=', Gamma[i,:,:])
    
        # Identical to Fei algo (above), but maybe more efficient
        Gamma[i,:,:] = sp.linalg.solve_discrete_lyapunov(F[i*n_r+i,:,:], Q[i*n_r+i,:,:], method=None)
        print('scipy: Gamma[i,:,:]=', Gamma[i,:,:])
        
        if is_pos_def(Gamma[i,:,:]) == False:
            print('i=', i, ' --> PROBLEM with Gamma matrix in From_FQ_to_Cov_Lyapunov!!')
            print(Gamma[i,:,:])
            input('pause in From_FQ_to_Cov_Lyapunov')

         # test si A X A ^t - X + Q == 0
        TestLyapunov = np.dot(np.dot(F[i*n_r+i,:,:], Gamma[i,:,:]), np.transpose(F[i*n_r+i,:,:])) - Gamma[i,:,:] + Q[i*n_r+i,:,:]
        if np.all(TestLyapunov<1E-5) == False:
            print('i=', i, 'TestLyapunov=', TestLyapunov)
            input('temp TEST')

    input('attente')

    for i in range(n_r_2):
        j = i//n_r
        # k = i%n_r
        SigmaT[i,:,:] = np.dot(F[i,:,:], Gamma[j,:,:])
        print('scipy: SigmaT[i,:,:]=', SigmaT[i,:,:])
    input('attente')
    
    for i in range(n_r_2):
        j = i//n_r
        k = i%n_r
        Cov_Z1_Z2_dp_R1_R2[i, 0:n_z, 0:n_z] = Gamma[j,:,:]
        Cov_Z1_Z2_dp_R1_R2[i, n_z:,  n_z:]  = Gamma[k,:,:]
        Cov_Z1_Z2_dp_R1_R2[i, 0:n_z, n_z:]  = np.transpose(SigmaT[i,:,:])
        Cov_Z1_Z2_dp_R1_R2[i, n_z:,  0:n_z] = SigmaT[i,:,:]
        
    if Test_isCGOMSM_from_Cov(Cov_Z1_Z2_dp_R1_R2, n_x) == False:
        print('i=', i, ' --> PROBLEM with Cov_Z1_Z2_dp_R1_R2 matrix : not CGOMSM!!')
        print(Cov_Z1_Z2_dp_R1_R2[i,:,:])
        input('pause in From_FQ_to_Cov_Lyapunov')

    for i in range(n_r_2):
        if is_pos_def(Cov_Z1_Z2_dp_R1_R2[i,:,:]) == False:
            print('i=', i, ' --> PROBLEM with Cov_Z1_Z2_dp_R1_R2 matrix in From_FQ_to_Cov_Lyapunov!!')
            print(Cov_Z1_Z2_dp_R1_R2[i,:,:])
            input('pause in From_FQ_to_Cov_Lyapunov')

    return Cov_Z1_Z2_dp_R1_R2

def NearPD(A):
    ZERO = 1E-12
    EPS  = 1E-12
    W, V      = np.linalg.eig(A)
    # W[W == 0] = np.spacing(1)      # making 0 eigenvalues non-zero    
    # W[W < 0]  = np.abs(W[W < 0])
    W[W <= ZERO] = EPS
    W = W*np.eye(np.int(np.sqrt(np.size(A))))
    # Anew      = np.dot(W*V, np.linalg.inv(V))
    Anew      = np.dot(np.dot(V, W),V.T)
    return (Anew)

def K_Means(Y, k):
    Y_whiten = vq.whiten(np.transpose(Y))
    Centroid, R_kmeans = vq.kmeans2(Y_whiten, k, iter=100, minit='random')
    return R_kmeans

def MSE_PK_MARG(X, Y, axis=0):
    return np.zeros(shape=(np.shape(X)[axis]))

def SE_PKn_MARG(X, Y, axis=0):
    return np.zeros(shape=(np.shape(X)))

def MSE_PK(Est, real):
    """MSE of estimated X"""
    #SP_mse_sum = np.mean((Est[0, 1:-1]-real[0, 1:-1])**2)
    # print('shape of Est = ', np.shape(Est))
    # print('shape of real = ', np.shape(real))
    if Est.ndim == 2:
        SP_mse_sum = np.mean((Est-real)**2, axis=1)
    else:
        SP_mse_sum = np.mean((Est-real)**2, axis=0)

    return SP_mse_sum

def SE_PKn(Est, real):
    """MSE of estimated X"""
    SquareError = (Est-real)**2
    return SquareError

def Error_Ratio(Est, Real):
    return np.mean(np.abs(Est-Real))

def getprobamarkov(JProba):
    n_r = np.shape(JProba)[0]
    IProba = np.sum(JProba, axis= 1).T
    TProba = np.zeros(shape = np.shape(JProba))
    for r in range(n_r):
        TProba[r, :] = JProba[r, :] / IProba[r]
    
    # SProba = np.zeros(shape = np.shape(IProba))
    eig_value, eig_vector = LA.eig(TProba.T)
    I = np.where(eig_value >= 0.9999)
    if np.size(I) != 1:
        print('Array I = ', I)
        inp('PROBLEM : the size of I should be 1')
    SProba = np.reshape(eig_vector[:, I] / np.sum(eig_vector[:, I]), newshape=(n_r))
    # print('Array SProba = ', SProba)
    # input('pause')

    return IProba, TProba, SProba

def Readin_data(filename):
    f = open(filename, 'r')
    K = 0
    flag = 0
    m = 0
    for line in f:
        if line.startswith("#"):
            flag = 0
        else:
            flag = 1
        if flag-m == 1:
            K = K+1
        if flag == 1:
            if K == 1:
                X = mat_read(m, line)
            if K == 2:
                R = mat_read(m, line)
            if K == 3:
                Y = mat_read(m, line)
        m = flag
    f.close()

    return (X, R, Y)


def Readin_ABMeansProba(filenameParam):

    K2   = 0
    K3   = 0
    flag = 0
    m    = 0
    Name = ''
    okJump = False
    with open(filenameParam, 'r') as f:
        for line in f:

            if line.startswith("#"):
                flag=0
            else:
                flag=1
            if line.startswith("# A"):
                Name = 'A'
                K2 = K2+1
            if line.startswith("# B"):
                Name = 'B'
                K3 = K3+1
            if line.startswith("# joint jump matrix"):
                Name='joint_proba'
            if line.startswith("# mean of X"):
                Name = 'Mean_X'
            if line.startswith("# mean of Y"):
                Name = 'Mean_Y'

            if (flag == 1 and Name == 'A'):
                locals()['A%s'%K2] = mat_read(m, line)
            if (flag == 1 and Name == 'B'):
                locals()['B%s'%K3] = mat_read(m, line)
            if (flag==1 and Name=='joint_proba'):
                okJump = True
                joint_proba=mat_read(m,line)
            if (flag == 1 and Name == 'Mean_X'):
                Mean_X = mat_read(m, line)
            if (flag == 1 and Name == 'Mean_Y'):
                Mean_Y = mat_read(m, line)

            m = flag

    # print('locals=', locals())
    # input('pause')

    # Verification of the dimensions of matrices and vectors
    if okJump == True:
        n_rp1, n_rp2 = np.shape(joint_proba)
    n_rX, n_x    = np.shape(Mean_X)
    n_rY, n_y    = np.shape(Mean_Y)
    n_z          = n_x + n_y
    n_z_2        = n_z*2

    if (okJump == True and (not (n_rp1 == n_rp2 == n_rX == n_rY))) or (okJump == False and (not (n_rX == n_rY))):
        input('probleme - n_r dimension incoherent in the parameter file')
        exit(1)
    else:
        n_r = n_rX

    # Test the number of matrices
    OK = True
    for r in range(n_r**2):
        if 'A'+str(r+1) not in locals():
            print('A Problem --> r=', r)
            OK = False
            break
        if 'B'+str(r+1) not in locals():
            print('B Problem --> r=', r)
            OK = False
            break
    if OK == False:
        input('PROBLEM in Readin_ABMeansProba')

    # Test the dimensions of matrices
    for r in range(n_r**2):
        siz = np.shape(locals()['A%s'%(r+1)])
        if siz[0] != n_z or siz[1] != n_z :
            print('The dimension of A matrices are not corrects!!!')
            exit(1)
    for r in range(n_r**2):
        siz = np.shape(locals()['B%s'%(r+1)])
        if siz[0] != n_z or siz[1] != n_z :
            print('The dimension of B matrices are not corrects!!!')
            exit(1)

    A = np.zeros(shape=(n_r**2, n_z, n_z))
    for l in range(n_r**2):
        A[l, :, :] = locals()['A%s'%(l+1)]

    B = np.zeros(shape=(n_r**2, n_z, n_z))
    Q = np.zeros(shape=(n_r**2, n_z, n_z))
    for l in range(n_r**2):
        B[l, :, :] = locals()['B%s'%(l+1)]
        Q[l, :, :] = np.dot(B[l, :, :], np.transpose(B[l, :, :]))

    # Obtention des matrices cov
    Cov = From_FQ_to_Cov_Lyapunov(A, Q, n_x)
    # Test if the matrices form a CGPMSM
    if Test_if_CGPMSM(Cov) == False:
        print('The cov matrices in the parameter file doest not respect shape for CGPMSM!!')
        exit(1)

    # Obtention des proba marg et cond à partir de proba joint
    if okJump == True:

        # teste si les rpoba sont de sproba
        if len(joint_proba[joint_proba<0.0]) > 0 or len(joint_proba[joint_proba>1.0]) > 0:
            print('Array joint_proba = ', joint_proba)
            exit('    --> PROBLEM Readin_CovMeansProba because all values must be between 0 and 1')

        # test si stationanire (les sum des colonnes doivent être identiques)
        Stationary = np.around(np.sum(joint_proba, axis=0), decimals=4)
        if np.all(Stationary == Stationary[0], axis = 0) == False:
            print('Array Stationary = ', Stationary)
            exit('PROBLEM : the matrix is not stationary')

        if np.abs(1.-sum(sum(joint_proba)))> 1E-4:
            print('Les proba jointes ne somment pas à 1!! -- normalisation')
            print('np.abs(1.-sum(sum(joint_proba)))=', np.abs(1.-sum(sum(joint_proba))))
            print('Avant norma --> Array joint_proba = ', joint_proba)
            joint_proba /= np.sum(joint_proba)
            print('Apres norma --> Array joint_proba = ', joint_proba)
            input('pause')
        init_proba, trans_proba, steady_proba = getprobamarkov(joint_proba)
        return (n_r, A, B, Q, Cov, Mean_X, Mean_Y, joint_proba, init_proba, trans_proba, steady_proba)
    else:
        return (n_r, A, B, Q, Cov, Mean_X, Mean_Y)


def Readin_CovMeansProba(filenameParam):

    K1      = 0
    flag    = 0
    m       = 0
    Name    = ''
    STEPS   = 0
    okJump  = False
    okSTEPS = False
    with open(filenameParam, 'r') as f:
        for line in f:

            if line.startswith("#"):
                flag=0
            else:
                flag=1
            if line.startswith("# number of fuzzy steps"):
                Name='STEPS'
            if line.startswith("# Cov"):
                Name = 'Cov'
                K1 = K1+1
            if line.startswith("# joint jump matrix"):
                Name='joint_proba'
            if line.startswith("# mean of X"):
                Name = 'Mean_X'
            if line.startswith("# mean of Y"):
                Name = 'Mean_Y'

            if (flag == 1 and Name == 'Cov'):
                locals()['Cov_%s'%K1] = mat_read(m, line)
            if (flag==1 and Name=='STEPS'):
                okSTEPS = True
                STEPS=int(mat_read(m,line).item())
            if (flag==1 and Name=='joint_proba'):
                okJump = True
                joint_proba=mat_read(m,line)
            if (flag == 1 and Name == 'Mean_X'):
                Mean_X = mat_read(m, line)
            if (flag == 1 and Name == 'Mean_Y'):
                Mean_Y = mat_read(m, line)

            m = flag

    # Verification of the dimensions of matrices and vectors
    if okJump == True:
        n_rp1, n_rp2 = np.shape(joint_proba)
    n_rX, n_x = np.shape(Mean_X)
    n_rY, n_y = np.shape(Mean_Y)
    n_rX -= STEPS
    n_rY -= STEPS
    n_z          = n_x + n_y
    n_z_2        = n_z*2

    if ((okJump == True) and (not (n_rp1 == n_rp2 == n_rX == n_rY))) or (okJump == False and (not (n_rX == n_rY))):
        input('probleme - n_r dimension incoherent in the parameter file')
        exit(1)
    n_r = n_rX

    # Test the number of matrices
    OK = True
    for r in range((n_r+STEPS)**2):
        if 'Cov_'+str(r+1) not in locals():
            print('r=', r)
            OK = False
            break
    if OK == False:
        input('PROBLEM - Readin_CovMeansProba')

    # Test the dimensions of matrices
    for r in range((n_r+STEPS)**2):
        siz = np.shape(locals()['Cov_%s'%(r+1)])
        if siz[0] != n_z_2 or siz[1] != n_z_2 :
            print('The dimension of cov matrices are not corrects!!!')
            exit(1)

    print((n_r+STEPS)**2)
    print(n_z_2)
    Cov = np.zeros(shape=((n_r+STEPS)**2, n_z_2, n_z_2))
    for l in range((n_r+STEPS)**2):
        Cov[l, :, :] = locals()['Cov_%s'%(l+1)]
        # test if they are positive definite
        if is_pos_def(Cov[l, :, :]) == False:
            print('l=', l, ' --> PROBLEM with Cov matrix in parameter file!!')
            print(Cov[l, :, :])
            input('pause')
            # exit(1)

    # Test if the matrices form a CGPMSM
    if Test_if_CGPMSM(Cov) == False:
        print('The cov matrices in the parameter file doest not respect shape for CGPMSM!!')
        exit(1)

    # Obtention des matrices A, B, Q
    A, Q = From_Cov_to_FQ(Cov)
    B    = np.zeros(shape=((n_r+STEPS)**2, n_z, n_z))
    for i in range((n_r+STEPS)**2):
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            B[i,:,:] = sp.linalg.sqrtm(Q[i,:,:])

    # Obtention des proba marg et cond à partir de proba joint
    if okJump == True:

        # teste si les rpoba sont de sproba
        if len(joint_proba[joint_proba<0.0]) > 0 or len(joint_proba[joint_proba>1.0]) > 0:
            print('Array joint_proba = ', joint_proba)
            exit('    --> PROBLEM Readin_CovMeansProba because all values must be between 0 and 1')

        # test si stationanire (les sum des colonnes doivent être identiques)
        Stationary = np.around(np.sum(joint_proba, axis=0), decimals=4)
        if np.all(Stationary == Stationary[0], axis = 0) == False:
            print('Array Stationary = ', Stationary)
            exit('PROBLEM : the matrix is not stationary')

        if np.abs(1.-sum(sum(joint_proba)))> 1E-4:
            print('Les proba jointes ne somment pas à 1!! -- normalisation')
            print('np.abs(1.-sum(sum(joint_proba)))=', np.abs(1.-sum(sum(joint_proba))))
            print('Avant norma --> Array joint_proba = ', joint_proba)
            joint_proba /= np.sum(joint_proba)
            print('Apres norma --> Array joint_proba = ', joint_proba)
            input('pause')
        init_proba, trans_proba, steady_proba = getprobamarkov(joint_proba)
        return (n_r, A, B, Q, Cov, Mean_X, Mean_Y, joint_proba, init_proba, trans_proba, steady_proba)
    else:
        return (n_r, A, B, Q, Cov, Mean_X, Mean_Y)

def Test_if_CGPMSM(Cov):

    n_r_2, n_z_2, useless = np.shape(Cov)
    n_r = int(np.sqrt(n_r_2))
    n_z = int(n_z_2/2)

    # Recuperation of the Sigma matrices
    Sigma = np.zeros(shape=(n_r, n_z, n_z))
    for r in range(n_r):
        l = r*n_r+r
        Sigma[r, :, :] = Cov[l, 0:n_z, 0:n_z]

    OK = True
    for l in range(n_r_2):
        j = l//n_r
        k = l%n_r
        sig1 = Cov[l, 0:n_z, 0:n_z]
        sig2 = Cov[l, n_z:2*n_z, n_z:2*n_z]
        if not np.allclose(sig1, Sigma[j,:,:], atol = 1E-8) or not np.allclose(sig2, Sigma[k,:,:], atol = 1E-8):
            print(not np.allclose(sig1, Sigma[j,:,:], atol = 1E-8))
            print(not np.allclose(sig2, Sigma[k,:,:], atol = 1E-8))
            print(Cov[l, :, :])
            print('The cov matrix l=', l, 'is not well shaped !!!')
            input('attente')
            #exit(1)
    return OK

def is_pos_def(A):
    if not np.any(A) == True:
        return True
    if np.allclose(A, A.T, atol=1E-8):
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            print('PB is_pos_def : ce n''est pas un matrice définie positive')
            std_ = np.sqrt(np.diag(A))
            corr = A / np.outer(std_, std_)
            print('correlation matrix=', corr)
            # input('pause')
            return False
    else:
        print('PB is_pos_def : la mat n''est pas symétrique')
        print(A)
        input('pause')
        return False


def SaveSimulatedData(X, R, Y, filenameXRY):
    f = open(filenameXRY, 'w')
    f.write('#=====================================#\n# X R Y\n#=====================================#\n#\n')
    f.close()
    f = open(filenameXRY, 'ab')
    np.savetxt(f, X, delimiter=" ", header='hidden process X\n================================================#',   footer='\n', fmt='%.4f')
    np.savetxt(f, R, delimiter=" ", header='markov chain R\n================================================#',     footer='\n', fmt='%d')
    np.savetxt(f, Y, delimiter=" ", header='observed process Y\n================================================#', footer='\n', fmt='%.4f')
    f.close()

def SaveSimulatedFuzzyData(X, R, Y, filenameXRY):
    f = open(filenameXRY, 'w')
    f.write('#=====================================#\n# X R Y\n#=====================================#\n#\n')
    f.close()
    f = open(filenameXRY, 'ab')
    np.savetxt(f, X, delimiter=" ", header='hidden process X\n================================================#',     footer='\n', fmt='%.4f')
    np.savetxt(f, R, delimiter=" ", header='fuzzy markov chain R\n================================================#', footer='\n', fmt='%.4f')
    np.savetxt(f, Y, delimiter=" ", header='observed process Y\n================================================#',   footer='\n', fmt='%.4f')
    f.close()

def ReadSimulatedFuzzyData(filenameXRY):
    f = open(filenameXRY, 'r')
    f.readline()
    f.readline()
    f.readline()
    f.readline()
    f.readline()
    f.readline()
    Array = np.loadtxt(f, delimiter=" ")
    f.close()
    return np.expand_dims(Array[0, :], axis=0), np.expand_dims(Array[1, :], axis=0), np.expand_dims(Array[2, :], axis=0)

def SaveNoiseJump(U, V, R, filenameUVR):
    f = open(filenameUVR, 'w')
    f.write('#=====================================#\n# U V R\n#=====================================#\n#\n')
    f.close()
    f = open(filenameUVR, 'ab')
    np.savetxt(f, U, delimiter=" ", header='Noise U\n================================================#',        footer='\n', fmt='%.4f')
    np.savetxt(f, V, delimiter=" ", header='Noise V\n================================================#',        footer='\n', fmt='%.4f')
    np.savetxt(f, R, delimiter=" ", header='Markov Chain R\n================================================#', footer='\n', fmt='%d')
    f.close()

def Est_p_R1_R2(R, N, psi=None):
    if psi is None:
        R = np.array(np.squeeze(R), ndmin=1, copy=False)
        n_r = len(np.unique(R))
        R_value = np.linspace(0, n_r-1, n_r).astype(int)
        Couple_R = np.vstack((R[0:-1], R[1:]))
        num_R1_R2 = np.zeros(shape=(n_r**2))
        for i in range(n_r**2):
            Sign_R1_R2_temp = (np.sum(Couple_R == np.array([[R_value[i//n_r]],\
             [R_value[i%n_r]]]), 0) == 2)
            num_R1_R2[i] = np.sum(Sign_R1_R2_temp)
        p_R1_R2 = np.reshape(num_R1_R2/(N-1), (n_r, n_r))
    else:
        p_R1_R2 = 1.0/(N-1.0)*np.sum(psi, axis=0)
    return p_R1_R2

def mat_read(the_mark_previousline, line):
    global Q
    if the_mark_previousline == 0:
        Q = np.fromstring(line, dtype=float, sep=' ')
        Q = Q.reshape(1, np.size(Q))
    else:
        q = np.fromstring(line, dtype=float, sep=' ')
        Q = np.append(Q, q.reshape(1, np.size(q)), axis=0)

    return Q


class Colors:
    RED = "\033[1;31m"
    BLUE = "\033[1;34m"
    CYAN = "\033[1;36m"
    GREEN = "\033[0;32m"
    MAGENTA = "\033[1;35m"
    RESET = "\033[0;0m"
    BOLD = "\033[;1m"
    REVERSE = "\033[;7m"
