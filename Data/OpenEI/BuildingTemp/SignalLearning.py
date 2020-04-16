#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import numpy as np
import pandas as pd
import datetime as dt
import sys

import clipboard
#import PlotSignals as PS

listJune    = ['building1retail_June_Week_672', 'building2retail_June_Week_672', 'building3retail_June_Week_672', 'building4retail_June_Week_672', 'building5retail_June_Week_672']
listJanuary = ['building1retail_January_Week_667', 'building2retail_January_Week_667', 'building3retail_January_Week_667', 'building4retail_January_Week_667', 'building5retail_January_Week_667']

Prefix='../../../../Data_CGPMSM/OpenEI/BuildingTemp/'

def main():

    hard, filt, smooth = 1, 1, 0
    chWork = str(hard) + ',' + str(filt) + ',' + str(smooth)
    #steps   = '0'
    steps   = '1,5'
    steps   = '1,2,3,4,5,7,9'
    # steps   = '1,2,3,4,5,6,8,10,12,14,16'
    verbose = 2
    plot    = 1

    name = listJune[4]
    name1='../Data_CGPMSM/OpenEI/BuildingTemp/input/'+ name + '_GT.csv'

    ##################################
    # Lecture des données
    DOE_df = pd.read_csv(Prefix + 'input/' + name + '_GT.csv', parse_dates=[0])
    pd.to_datetime(DOE_df['Timestamp'])
    DOE_df.sort_values(by=['Timestamp'])
    listeHeader = list(DOE_df)
    print(listeHeader)

    datemin = DOE_df['Timestamp'].iloc[0]
    datemax = DOE_df['Timestamp'].iloc[-1]
    # input('pause')
    print('Building name : ', name)
    print('  -->Date départ série temporelle = ', datemin)
    print('  -->Date fin    série temporelle = ', datemax)
    listeHeader = list(DOE_df)
    print(listeHeader)

    n_x, n_y = 1, 1
    n_z      = n_x + n_y
    n_r      = 2

    ##################################
    # Apprentissage des paramètres X,Y cond R --> enregistrement du fichier de paramètres
    filanemaneParam = 'SignalDOE.param'

    # Learning parameters mean and cov
    meanX, meanY, cpt = getMean      (DOE_df, n_r, n_x, n_y, n_z)
    Cov               = getCov_CGPMSM(DOE_df, n_r, n_x, n_y, n_z, meanX, meanY)

    # Arrondis pour la sauvegarde
    meanX = np.around(meanX, decimals=4)
    meanY = np.around(meanY, decimals=4)
    print('meanX=', meanX)
    print('meanY=', meanY)
    print('cpt=', cpt)

    #############
    # print('Cov=', Cov)
    Cov = GetParamNearestCGO_cov(Cov, n_x=n_x)
    # print('Cov=', Cov)
    # input('pause tempo')
    #############

    # Enregistrement du fichier de paramètres
    printfileParam('../../../Parameters/Signal/' + filanemaneParam, Cov, meanX, meanY, n_r)

    ##################################
    # Apprentissage des paramètres de R
    # coeffdelta = 16
    # coeffalpha = 1./6.
    # alpha, beta, gamma, delta_d, delta_u = getRParam_FMC2(DOE_df, np.sum(cpt), coeffalpha=coeffalpha, coeffdelta=coeffdelta)
    # print('alpha=', alpha, ', beta=', beta, ', gamma = ', gamma, ', delta_d=', delta_d, ', delta_u=', delta_u)
    coeffalpha = 1.7
    alpha, beta, delta, eta = getRParam_FMC1(DOE_df, np.sum(cpt), coeffalpha)
    alpha = 0.0
    eta   = 0.50
    print('alpha=', alpha, ', beta=', beta, ', delta = ', delta, ', eta=', eta)

    #################################
    # Commande d'appel au programme
    # A  = 'python3 Test_CGOFMSM_Signals.py ./Parameters/Signal/' + filanemaneParam + ' 4:' + str(alpha) + ':' + str(gamma) + ':' + str(delta_d) + ':' + str(delta_u) + ' '
    # A += chWork + ' ' + name1 + ' ' + steps + ' ' + str(verbose) + ' ' + str(plot)
    A  = 'python3 Test_CGOFMSM_Signals.py ./Parameters/Signal/' + filanemaneParam + ' 2:' + str(alpha) + ':' + str(beta) + ':' + str(eta) + ':' + str(delta) + ' '
    A += chWork + ' ' + name1 + ' ' + steps + ' ' + str(verbose) + ' ' + str(plot) 

    clipboard.copy(A.strip()) # mise ne moire de la commande à exécuter
    print('pour restaurer le signal:')
    print('\n', A, '\n')

    # Commande pour simuler un signal de la même forme et le restaurer par CGOFMSM
    N = 1000
    NbExp = 1
    # B = 'python3 CGOFMSM_Simulation.py ./Parameters/Signal/' + filanemaneParam + ' 4:' + str(alpha) + ':' + str(gamma) + ':' + str(delta_d) + ':' + str(delta_u) + ' '
    # B += chWork + ' ' + str(N) + ' ' + steps + ' ' + str(NbExp) + ' ' + str(verbose) + ' ' + str(plot)
    B = 'python3 CGOFMSM_Simulation.py ./Parameters/Signal/' + filanemaneParam + ' 2:' + str(alpha) + ':' + str(beta) + ':' + str(eta) + ':' + str(delta) + ' '
    B += chWork + ' ' + str(N) + ' ' + steps + ' ' + str(NbExp) + ' ' + str(verbose) + ' ' + str(plot)
    print('pour simuler + restaurer:')
    print('\n', B, '\n')

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


def getRParam_FMC1(df, sumcpt01, coeffalpha):

    # estimation of beta
    beta = 0.0 # Pas de transition entre 0 et 1 et inversement

    # estimation of delta
    delta = 0.0

    PH = sumcpt01 / df['Y'].count() * coeffalpha
    print('PH=', PH)

    eta   = 6./5. * (-2./3. * delta + (1.-PH)/2.)
    alpha = (PH-(delta+eta)) / 2. - beta

    return alpha, beta, delta, eta


def getRParam_FMC2(df, meancpt01, coeffalpha, coeffdelta):

    # estimation of beta
    beta = 0.0 # Pas de transition entre 0 et 1 et inversement

    # estimation de alpha par la proprotion d'être 0 OU 1
    alpha = meancpt01 / df['Y'].count() * coeffalpha
    # print('Array alpha = ', alpha)

    # estimation de delta par la pente maximum sur AT_R_GT
    delta = 0.
    for i, fuzzylevel in enumerate(df['R_GT'][:-1]):
        if fuzzylevel != 0.0 and fuzzylevel != 1.0:
            delta = max(delta, df['R_GT'][i+1] - df['R_GT'][i])
    # print(df['Timestamp'][1] - df['Timestamp'][0])
    # print((df['Timestamp'][2] - df['Timestamp'][1]).total_seconds())
    # input('pause tiemstamp')
    # print('Array delta = ', delta)
    delta *= coeffdelta
    # print('Array delta = ', delta)
    # input('pause')

    # estimationde gamma selon l'eq. (32) du papier
    M = 6*delta - delta*delta
    gamma = (1.-2.*(beta + alpha)) / M

    return alpha, beta, gamma, delta, delta

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

def getMean(df, n_r, n_x, n_y, n_z):

    meanX = np.zeros(shape=(n_r, n_x))
    meanY = np.zeros(shape=(n_r, n_y))
    cpt   = np.zeros((2), dtype = int)
    for i, fuzzylevel in enumerate(df['R_GT']):
        if fuzzylevel == 0.0 or fuzzylevel == 1.0:
            fl = int(fuzzylevel)
            cpt[fl] += 1
            meanX[fl, 0] += df['X'][i]
            meanY[fl, 0] += df['Y'][i]
    meanX[0, 0] /= cpt[0]
    meanY[0, 0] /= cpt[0]
    meanX[1, 0] /= cpt[1]
    meanY[1, 0] /= cpt[1]

    return meanX, meanY, cpt

def getCov_CGPMSM(df, n_r, n_x, n_y, n_z, meanX, meanY):

    Gamma = np.zeros((n_r, n_z, n_z))
    cpt1  = np.zeros(n_r)
    for i, s in enumerate(df['R_GT'][:]):
        if s == 0. or s == 1.:
            r = int(s)
            cpt1[r] += 1.
            # print(i, r)
            # print(df['X'][i] - meanX[r, :])
            # print(np.dot(df['X'][i] - meanX[r, :], df['X'][i] - meanX[r, :]))
            Gamma[r, 0:n_x,   0:n_x  ] += np.dot(df['X'][i] - meanX[r, :], df['X'][i] - meanX[r, :])
            Gamma[r, 0:n_x,   n_x:n_z] += np.dot(df['X'][i] - meanX[r, :], df['Y'][i] - meanY[r, :])
            Gamma[r, n_x:n_z, 0:n_x  ] += np.dot(df['Y'][i] - meanY[r, :], df['X'][i] - meanX[r, :])
            Gamma[r, n_x:n_z, n_x:n_z] += np.dot(df['Y'][i] - meanY[r, :], df['Y'][i] - meanY[r, :])
    for r in range(n_r):
        if cpt1[r] != 0:
            Gamma[r, :, :] /= cpt1[r]
    # print('Gamma=\n', Gamma)
    # input('pause Gamma')

    Sigma = np.zeros((n_r**2, n_z, n_z))
    cpt2  = np.zeros((n_r**2))
    for i, s in enumerate(df['R_GT'][:-1]):
        if s == 0. or s == 1.:
            r = int(s)
            if df['R_GT'][i+1] == 0. or df['R_GT'][i+1] == 1.:
                k = int(df['R_GT'][i+1])
                l = r*n_r+k
                cpt2[l] += 1.
                Sigma[l, 0:n_x,   0:n_x  ] += np.dot(df['X'][i] - meanX[r, :], df['X'][i+1] - meanX[k, :])
                Sigma[l, 0:n_x,   n_x:n_z] += np.dot(df['X'][i] - meanX[r, :], df['Y'][i+1] - meanY[k, :])
                Sigma[l, n_x:n_z, 0:n_x  ] += np.dot(df['Y'][i] - meanY[r, :], df['X'][i+1] - meanX[k, :])
                Sigma[l, n_x:n_z, n_x:n_z] += np.dot(df['Y'][i] - meanY[r, :], df['Y'][i+1] - meanY[k, :])

    for l in range(n_r**2):
        if cpt2[l] != 0:
            Sigma[l, :, :] /= cpt2[l]
        # else:
        #   j = l//n_r
        #   k = l%n_r
        #   rho = 0.1
        #   for m in range(n_z):
        #       for n in range(n_z):
        #           sig1 = np.sqrt(Gamma[j, m, m])
        #           sig2 = np.sqrt(Gamma[k, n , n])
        #           Sigma[l, m, n] = rho * sig1 * sig2
    # print('Sigma=\n', Sigma)
    # input('pause Sigma')

    Cov = np.zeros((n_r**2, n_z*2, n_z*2))
    for i in range(n_r**2):
        j = i//n_r
        k = i%n_r
        Cov[i, 0:n_z, 0:n_z] = Gamma[j, :, :]
        Cov[i, n_z:,  n_z:]  = Gamma[k, :, :]
        Cov[i, 0:n_z, n_z:]  = Sigma[i, :, :]
        Cov[i, n_z:,  0:n_z] = Sigma[i, :, :].T
        if is_pos_def(Cov[i, :, :]) == False:
            input('Nous avons un pb - les matrices de cov ne sont pas definies positives')
    # print('Cov=\n', Cov)
    # input('pause Cov')

    return Cov

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

# def getCov_CGPMSM_OLD(df, n_r, n_x, n_y, n_z, meanX, meanY):

#   Cov = np.zeros((n_r*n_r, n_z*2, n_z*2))
#   cpt  =np.zeros((n_r*n_r), dtype = int)

#   cptCouples=0
#   for i, fuzzylevel in enumerate(df['R_GT'][:-1]):
#       if df['R_GT'][i] == 0. or df['R_GT'][i] == 1.:
#           j = int(df['R_GT'][i])
#           if df['R_GT'][i+1] == 0. or df['R_GT'][i+1] == 1.:
#               k = int(df['R_GT'][i+1])

#               l = j*n_r+k

#               cpt[l] += 1
#               # print('j=', j, ', k=', k)
#               # input('attent')
#               Cov[l, 0, 0] += (df['X'][i]-meanX[j])   * (df['X'][i]   - meanX[j])
#               Cov[l, 0, 1] += (df['X'][i]-meanX[j])   * (df['Y'][i]   - meanY[j])
#               Cov[l, 0, 2] += (df['X'][i]-meanX[j])   * (df['X'][i+1] - meanX[k])
#               Cov[l, 0, 3] += (df['X'][i]-meanX[j])   * (df['Y'][i+1] - meanY[k])

#               Cov[l, 1, 0] += (df['Y'][i]-meanY[j])   * (df['X'][i]   - meanX[j])
#               Cov[l, 1, 1] += (df['Y'][i]-meanY[j])   * (df['Y'][i]   - meanY[j])
#               Cov[l, 1, 2] += (df['Y'][i]-meanY[j])   * (df['X'][i+1] - meanX[k])
#               Cov[l, 1, 3] += (df['Y'][i]-meanY[j])   * (df['Y'][i+1] - meanY[k])

#               Cov[l, 2, 0] += (df['X'][i+1]-meanX[k]) * (df['X'][i]   - meanX[j])
#               Cov[l, 2, 1] += (df['X'][i+1]-meanX[k]) * (df['Y'][i]   - meanY[j])
#               Cov[l, 2, 2] += (df['X'][i+1]-meanX[k]) * (df['X'][i+1] - meanX[k])
#               Cov[l, 2, 3] += (df['X'][i+1]-meanX[k]) * (df['Y'][i+1] - meanY[k])

#               Cov[l, 3, 0] += (df['Y'][i+1]-meanY[k]) * (df['X'][i]   - meanX[j])
#               Cov[l, 3, 1] += (df['Y'][i+1]-meanY[k]) * (df['Y'][i]   - meanY[j])
#               Cov[l, 3, 2] += (df['Y'][i+1]-meanY[k]) * (df['X'][i+1] - meanX[k])
#               Cov[l, 3, 3] += (df['Y'][i+1]-meanY[k]) * (df['Y'][i+1] - meanY[k])

#   for j in range(n_z):
#       for k in range(n_z):
#           l = j*n_r+k
#           Cov[l, :, :] = (Cov[l, :, :] + Cov[l, :, :].transpose())
#           if cpt[l] != 0:
#               Cov[l, :, :] = Cov[l, :, :] / (2.*cpt[l])
#   # Au cas ou des matrices sont remplies de 0 --> on recopie une autre
#   Cov[1, :, :] = Cov[0,:,:].copy()
#   Cov[2, :, :] = Cov[3,:,:].copy()

#   return Cov

if __name__ == '__main__': 
    main()

