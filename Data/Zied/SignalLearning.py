#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
#import matplotlib.pyplot as plt
import clipboard

Prefix     = './inputs/'
pathToSave = './results/'
listFiles  = ['Xtrain_extract.out', 'Ytrain_extract.out', 'Rtrain_extract.out']

def main():

    hard, filt, smooth = 1, 1, 0
    chWork = str(hard) + ',' + str(filt) + ',' + str(smooth)
    #steps   = '0'
    steps   = '1,5'
    #steps   = '1,2,3,4,5,7,9'
    # steps   = '1,2,3,4,5,6,8,10,12,14,16'
    verbose = 2
    plot    = 1

    # first file ##################################
    filenameorig = [pathToSave+listFiles[0], pathToSave+listFiles[1], pathToSave+listFiles[2]]
    
    # Reading the files
    X = np.loadtxt(filenameorig[0])
    Y = np.loadtxt(filenameorig[1])
    R = np.loadtxt(filenameorig[2])
    N = Y.shape[0]
    print('number of data : ', N)

    n_x, n_y = 1, 1
    n_z      = n_x + n_y
    n_r      = 2

    ##################################
    # Apprentissage des paramètres X,Y cond R --> enregistrement du fichier de paramètres
    filanemaneParam = 'SignalZIED.param'

    # Learning parameters mean and cov
    meanX, meanY, cpt = getMean      (X, Y, R, n_r, n_x, n_y, n_z)
    print('meanX=', meanX)
    print('meanY=', meanY)
    print('cpt=', cpt)
    Cov               = getCov_CGPMSM(X, Y, R, n_r, n_x, n_y, n_z, meanX, meanY, cpt)

    # Arrondis pour la sauvegarde
    meanX = np.around(meanX, decimals=4)
    meanY = np.around(meanY, decimals=4)

    #############
    # print('Cov=', Cov)
    Cov = GetParamNearestCGO_cov(Cov, n_x=n_x)
    print('Cov=', Cov)
    # input('pause tempo')
    #############

    # Enregistrement du fichier de paramètres
    printfileParam('../../Parameters/Signal/' + filanemaneParam, Cov, meanX, meanY, n_r)

    ##################################
    # Apprentissage des paramètres de R
    # coeffdelta = 16
    # coeffalpha = 1./6.
    # alpha, beta, gamma, delta, = getRParam_FMC2(DOE_df, np.sum(cpt), coeffalpha=coeffalpha, coeffdelta=coeffdelta)
    # print('alpha=', alpha, ', beta=', beta, ', gamma = ', gamma, ', delta=', delta)
    coeffalpha = 1.#0.9
    alpha, beta, delta, eta = getRParam_FMC1(X, Y, R, np.sum(cpt), coeffalpha)
    print('alpha=', alpha, ', beta=', beta, ', delta = ', delta, ', eta=', eta)


    #################################
    # Commande d'appel au programme
    name1='./Data/Zied/results/XYR.csv'
    # A  = 'python3 CGOFMSM_SignalRest.py ./Parameters/Signal/' + filanemaneParam + ' 4:' + str(alpha) + ':' + str(gamma) + ':' + str(delta) + ' '
    # A += chWork + ' ' + name1 + ' ' + steps + ' ' + str(verbose) + ' ' + str(plot)
    A  = 'python3 CGOFMSM_SignalRest.py ./Parameters/Signal/' + filanemaneParam + ' 2:' + str(alpha) + ':' + str(beta) + ':' + str(eta) + ':' + str(delta) + ' '
    A += chWork + ' ' + name1 + ' ' + steps + ' ' + str(verbose) + ' ' + str(plot) 

    clipboard.copy(A.strip()) # mise en memoire de la commande à exécuter
    print('pour restaurer le signal:')
    print('\n', A, '\n')

    # Commande pour simuler un signal de la même forme et le restaurer par CGOFMSM
    N = 1000
    NbExp = 1
    # B = 'python3 CGOFMSM_SimRest.py ./Parameters/Signal/' + filanemaneParam + ' 4:' + str(alpha) + ':' + str(gamma) + ':' + str(delta) + ' '
    # B += chWork + ' ' + str(N) + ' ' + steps + ' ' + str(NbExp) + ' ' + str(verbose) + ' ' + str(plot)
    B = 'python3 CGOFMSM_SimRest.py ./Parameters/Signal/' + filanemaneParam + ' 2:' + str(alpha) + ':' + str(beta) + ':' + str(eta) + ':' + str(delta) + ' '
    B += chWork + ' ' + str(N) + ' ' + steps + ' ' + str(NbExp) + ' ' + str(verbose) + ' ' + str(plot)
    print('pour simuler + restaurer:')
    print('\n', B, '\n')


def getRParam_FMC1(X, Y, R, sumcpt01, coeffalpha):

    # estimation of beta
    beta = 0.0 # Pas de transition entre 0 et 1 et inversement

    # estimation of delta
    delta = 0.0

    PH = sumcpt01 / Y.shape[0] * coeffalpha
    print('PH=', PH)

    eta   = 6./5. * (-2./3. * delta + (1.-PH)/2.)
    alpha = (PH-(delta+eta)) / 2. - beta

    return alpha, beta, delta, eta

def printfileParam(filanemane, Cov, meanX, meanY, n_r):

    # L'entete
    f = open(filanemane, 'w')
    f.write('#=====================================#\n# parameters for CGPMSM            # \n#=====================================# \n# \n# \n# matrix Cov_XY \n# ===============================================#\n# \n')
    f.close()

    f = open(filanemane, 'ab')
    
    # Les covariances
    for j in range(n_r):
        for k in range(n_r):
            l = j*n_r+k
            np.savetxt(f, Cov[l, :, :], delimiter=" ", header='Cov_xy'+str(j)+str(k)+'\n----------------------------', footer='\n', fmt='%.4f')

    # Les moyennes
    np.savetxt(f, meanX, delimiter=" ", header='mean of X'+'\n================================', footer='\n', fmt='%.4f')
    np.savetxt(f, meanY, delimiter=" ", header='mean of Y'+'\n================================', footer='\n', fmt='%.4f')

    f.close()

def getCov_CGPMSM(X, Y, R, n_r, n_x, n_y, n_z, meanX, meanY, cpt):

    Gamma = np.zeros((n_r, n_z, n_z))
    for i, s in enumerate(R):
        if s == 0. or s == 1.:
            r = int(s)
            Gamma[r, 0:n_x,   0:n_x  ] += np.dot(X[i] - meanX[r, :], X[i] - meanX[r, :])
            Gamma[r, 0:n_x,   n_x:n_z] += np.dot(X[i] - meanX[r, :], Y[i] - meanY[r, :])
            Gamma[r, n_x:n_z, 0:n_x  ] += np.dot(Y[i] - meanY[r, :], X[i] - meanX[r, :])
            Gamma[r, n_x:n_z, n_x:n_z] += np.dot(Y[i] - meanY[r, :], Y[i] - meanY[r, :])

    for r in range(n_r):
        if cpt[r] != 0:
            Gamma[r, :, :] /= cpt[r]
    #     print('Gamma[r, :, :]=\n', Gamma[r, :, :])
    #     print(is_pos_def(Gamma[r, :, :]))
    # input('pause Gamma')

    Sigma = np.zeros((n_r**2, n_z, n_z))
    cpt2  = np.zeros((n_r**2))
    for i, s in enumerate(R[:-1]):
        if s == 0. or s == 1.:
            r = int(s)
            if R[i+1] == 0. or R[i+1] == 1.:
                k = int(R[i+1])
                l = r*n_r+k
                cpt2[l] += 1.
                Sigma[l, 0:n_x,   0:n_x  ] += np.dot(X[i] - meanX[r, :], X[i+1] - meanX[k, :])
                Sigma[l, 0:n_x,   n_x:n_z] += np.dot(X[i] - meanX[r, :], Y[i+1] - meanY[k, :])
                Sigma[l, n_x:n_z, 0:n_x  ] += np.dot(Y[i] - meanY[r, :], X[i+1] - meanX[k, :])
                Sigma[l, n_x:n_z, n_x:n_z] += np.dot(Y[i] - meanY[r, :], Y[i+1] - meanY[k, :])

   
    for l in range(n_r**2):
        if cpt2[l] != 0.:
            Sigma[l, :, :] /= cpt2[l]
    #     print('Sigma[l, :, :]=\n', Sigma[l, :, :])
    # input('pause Sigma')
        # else:
        #   j = l//n_r
        #   k = l%n_r
        #   rho = 0.1
        #   for m in range(n_z):
        #       for n in range(n_z):
        #           sig1 = np.sqrt(Gamma[j, m, m])
        #           sig2 = np.sqrt(Gamma[k, n , n])
        #           Sigma[l, m, n] = rho * sig1 * sig2

    Cov = np.zeros((n_r**2, n_z*2, n_z*2))
    for l in range(n_r**2):
        j = l//n_r
        k = l%n_r
        print(j,k)
        Cov[l, 0:n_z, 0:n_z] = Gamma[j, :, :]
        Cov[l, n_z:,  n_z:]  = Gamma[k, :, :]
        Cov[l, 0:n_z, n_z:]  = Sigma[l, :, :]
        Cov[l, n_z:,  0:n_z] = Sigma[l, :, :].T
        if is_pos_def(Cov[l, :, :]) == False:
            input('Nous avons un pb - les matrices de cov ne sont pas definies positives')
            print(Cov[l, 0:n_z, 0:n_z])
            input('pause')
    # print('Cov=\n', Cov)
    # input('pause Cov')

    return Cov

def getMean(X, Y, R, n_r, n_x, n_y, n_z):

    meanX = np.zeros(shape=(n_r, n_x))
    meanY = np.zeros(shape=(n_r, n_y))
    cpt   = np.zeros((2), dtype = int)
    for i, fuzzylevel in enumerate(R):
        if fuzzylevel == 0.0 or fuzzylevel == 1.0:
            fl = int(fuzzylevel)
            cpt[fl] += 1
            meanX[fl, 0] += X[i]
            meanY[fl, 0] += Y[i]
    meanX[0, 0] /= cpt[0]
    meanY[0, 0] /= cpt[0]
    meanX[1, 0] /= cpt[1]
    meanY[1, 0] /= cpt[1]

    return meanX, meanY, cpt


def is_pos_def(A):
    if not np.any(A) == True:
        return True
    if np.allclose(A, A.T, atol=1E-8):
        try:
            Z = np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            print('PB is_pos_def : ce n''est pas un matrice définie positive')
            corr = np.zeros(shape = np.shape(A))
            for i in range(np.shape(corr)[0]):
                for j in range(np.shape(corr)[1]):
                    corr[i, j] = A[i, j]/np.sqrt(A[i,i] * A[j,j])
            print(corr)
            input('pause')
            return False
    else:
        print('PB is_pos_def : la mat n''est pas symétrique')
        print(A)
        input('pause')
        return False

def GetParamNearestCGO_cov(Cov, n_x):

    Cov_CGO = np.copy(Cov)

    # Si c'est déjà un CGO, alors rien à faire
    if Test_isCGOMSM_from_Cov(Cov, n_x) == False:

        n_r_2, n_z_mp2, useless2 = np.shape(Cov)
        n_z = n_z_mp2//2
        n_r = int(np.sqrt(n_r_2))
        
        for l in range(n_r**2):
            Bj      = Cov[l, 0:n_x,   n_x:n_z]
            GammaYY = Cov[l, n_x:n_z, n_x:n_z]
            Cjk     = Cov[l, n_x:n_z, n_z+n_x:2*n_z]
            Cov_CGO[l, 0:n_x        , n_z+n_x:2*n_z] = np.dot( np.dot( Bj, np.transpose(np.linalg.inv(GammaYY))), Cjk)
            Cov_CGO[l, n_z+n_x:2*n_z, 0:n_x]         = Cov_CGO[l, 0:n_x, n_z+n_x:2*n_z].T

    return Cov_CGO


def Test_isCGOMSM_from_Cov(Cov, n_x):
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

        all_zeros = not SOL.any()
        if all_zeros == False:
            Ok = False
            break

    return Ok


if __name__ == '__main__':
    main()
