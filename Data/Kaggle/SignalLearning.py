#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import numpy as np
import pandas as pd
import datetime as dt
import sys

import clipboard

Prefix='../../../Data_CGPMSM/Kaggle/'

listStations = ['35thAveSW_SWMyrtleSt', 'AlaskanWayViaduct_KingSt', 'AlbroPlaceAirportWay', 'AuroraBridge', 'HarborAveUpperNorthBridge', 'MagnoliaBridge', 'NE45StViaduct', 'RooseveltWay_NE80thSt', 'SpokaneSwingBridge', 'JoseRizalBridgeNorth']


def main():

    n_x, n_y = 1, 1
    n_z      = n_x + n_y
    n_r      = 2

    #ch ='_all_resample_5209_GT.csv'
    #ch = '_all_resample_1303_GT_excerpt.csv'
    ch = '_all_resample_1303_GT.csv'
    name = Prefix + 'input/'+ listStations[9] + ch

    ##################################
    # Lecture des données
    temperature_df = pd.read_csv(name, parse_dates=[0])
    pd.to_datetime(temperature_df['DateTime'])
    temperature_df.sort_values(by=['DateTime'])
    # for y in temperature_df.columns:
    #   print (temperature_df[y].dtype, end=', ')
    # print('count=', temperature_df.count())
    # print('-->Date départ série temporelle = ', temperature_df['DateTime'].iloc[0])
    # print('-->Date fin    série temporelle = ', temperature_df['DateTime'].iloc[-1])
    temperature_df.set_index('DateTime', inplace=True)
    listHeader = list(temperature_df)
    print(listHeader)
    #print(temperature_df.head(5))

    ##################################
    # Apprentissage des paramètres X,Y cond R --> enregistrement du fichier de paramètres
    filanemaneParam = 'Signal.param'

    # Learning parameters mean
    meanX, meanY, cpt = getMean(temperature_df)
    # print('meanX=', meanX)
    # print('meanY=', meanY)
    # print('cpt=', cpt)

    # Leanrnig parameters Cov
    Cov = getCov(temperature_df, n_r, n_z, meanX, meanY)
    # print('Cov=', Cov)

    # Enregistrement du fichier de paramètres
    printfileParam('../../Parameters/Signal/' + filanemaneParam, Cov, meanX, meanY, n_z)

    ##################################
    # Apprentissage des paramètres de R
    coeffmult = 15
    alpha, beta, gamma, delta = getRParam_FMC2(temperature_df, np.sum(cpt), coeff= coeffmult)
    print('alpha=', alpha, ', beta=', beta, ', gamma = ', gamma, ', delta=', delta, ', delta_u=', delta_u)
    # alpha, beta, delta, eta = getRParam_FMC1(temperature_df, np.sum(cpt))
    # print('alpha=', alpha, ', beta=', beta, ', delta = ', delta, ', eta=', eta)

    #################################
    # Commande d'appel au programme
    hard, filt, smooth = 1, 1, 0
    chWork = str(hard) + ',' + str(filt) + ',' + str(smooth)
    steps   = '16,21'
    verbose = 2
    plot    = 1
    name1 = '../Data_CGPMSM/Kaggle/input/'+ listStations[9] + ch

    A  = 'python3 CGOFMSM_SignalRest.py ./Parameters/Signal/' + filanemaneParam + ' 4:' + str(alpha) + ':' + str(gamma) + ':' + str(delta) + ' '
    A += chWork + ' ' + name1 + ' ' + steps + ' ' + str(verbose) + ' ' + str(plot)
    # A  = 'python3 CGOFMSM_Signals.py ./Parameters/Signal/' + filanemaneParam + ' 2:' + str(alpha) + ':' + str(beta) + ':' + str(eta) + ':' + str(delta) + ' '
    # A += chWork + ' ' + name1 + ' ' + steps + ' ' + str(verbose) + ' ' + str(plot) 

    clipboard.copy(A.strip()) # mise ne moire de la commande à exécuter
    print('\n', A, '\n')

    # Commande pour simuler un signal de la même forme et le restaurer par CGOFMSM
    N = 2000
    NbExp = 1
    B = 'python3 CGOFMSM_SimRest.py ./Parameters/Signal/' + filanemaneParam + ' 4:' + str(alpha) + ':' + str(gamma) + ':' + str(delta) + ' '
    B += chWork + ' ' + str(N) + ' ' + steps + ' ' + str(NbExp) + ' ' + str(verbose) + ' ' + str(plot)
    # B = 'python3 CGOFMSM.py ./Parameters/Signal/' + filanemaneParam + ' 4:' + str(alpha) + ':' + str(beta) + ':' + str(eta) + ':' + str(delta) + ' '
    # B += chWork + ' ' + str(N) + ' ' + steps + ' ' + str(NbExp) + ' ' + str(verbose) + ' ' + str(plot)
    print('pour simuler + restaurer:')
    print('\n', B, '\n')

def getRParam_FMC1(df, sumcpt01):

    # estimation of beta
    beta = 0.1 # Pas de transition entre 0 et 1 et inversement

    # estimation of delta
    delta = 0.0 

    PH = sumcpt01 / df.AirTemperature.count()
    #print('PH=', PH)

    eta   = 6./5. * (-2./3. * delta + (1.-PH)/2.)
    alpha = (PH-(delta+eta)) / 2. - beta

    return alpha, beta, delta, eta





def getRParam_FMC2(df, meancpt01, coeff):

    # estimation of beta
    beta = 0.0 # Pas de transition entre 0 et 1 et inversement

    # estimation de alpha par la proprotion d'être 0 OU 1
    alpha = meancpt01 / df.AirTemperature.count()
    print('Array alpha = ', alpha)

    # estimation de delta par la pente maximum sur AT_R_GT
    delta = 0.
    for i, fuzzylevel in enumerate(df.AT_R_GT[:-1]):
        if fuzzylevel != 0.0 and fuzzylevel != 1.0:
            delta = max(delta, df.AT_R_GT[i+1] - df.AT_R_GT[i])
    print('Array delta = ', delta)
    delta *= coeff
    print('Array delta = ', delta)
    input('pause getRParam_FMC2')
    
    # estimation de gamma selon l'eq. (32) du papier
    M = delta * (6.- delta)
    gamma = (1.-2.*(beta + alpha)) / M

    return alpha, beta, gamma, delta

def printfileParam(filanemane, Cov, meanX, meanY, n_z):

    # L'entete
    f = open(filanemane, 'w')
    f.write('#=====================================#\n# parameters for CGPMSM            # \n#=====================================# \n# \n# \n# matrix Cov_XY \n# ===============================================#\n# \n')
    f.close()

    f = open(filanemane, 'ab')
    
    # Les covariances
    for j in range(n_z):
        for k in range(n_z):
            np.savetxt(f, Cov[j,k,:,:], delimiter=" ", header='Cov_xy'+str(j)+str(k)+'\n----------------------------', footer='\n', fmt='%.4f')
    
    # Les moyennes
    np.savetxt(f, meanX, delimiter=" ", header='mean of X'+'\n================================', footer='\n', fmt='%.4f')
    np.savetxt(f, meanY, delimiter=" ", header='mean of Y'+'\n================================', footer='\n', fmt='%.4f')

    f.close()

def getMean(df):
    meanX=np.zeros((2))
    meanY=np.zeros((2))
    cpt  =np.zeros((2), dtype = int)
    for i, fuzzylevel in enumerate(df.AT_R_GT):
        if fuzzylevel == 0.0 or fuzzylevel == 1.0:
            fl = int(fuzzylevel)
            cpt[fl] += 1
            meanX[fl] += df.AT_X_GT[i]
            meanY[fl] += df.AirTemperature[i]
    meanX[0] /= cpt[0]
    meanY[0] /= cpt[0]
    meanX[1] /= cpt[1]
    meanY[1] /= cpt[1]

    return meanX, meanY, cpt

def getCov(df, n_r, n_z, meanX, meanY):

    Cov = np.zeros((n_r, n_r, n_z*2, n_z*2))
    cpt  =np.zeros((n_r, n_r), dtype = int)

    cptCouples=0
    for i, fuzzylevel in enumerate(df.AT_R_GT[:-1]):
        if df.AT_R_GT[i] == 0. or df.AT_R_GT[i] == 1.:
            j = int(df.AT_R_GT[i])
            if df.AT_R_GT[i+1] == 0. or df.AT_R_GT[i+1] == 1.:
                k = int(df.AT_R_GT[i+1])

                cpt[j,k] += 1
                # print('j=', j, ', k=', k)
                # input('attent')
                Cov[j, k, 0, 0] += (df.AT_X_GT[i]-meanX[j])          * (df.AT_X_GT[i]          - meanX[j])
                Cov[j, k, 0, 1] += (df.AT_X_GT[i]-meanX[j])          * (df.AirTemperature[i]   - meanY[j])
                Cov[j, k, 0, 2] += (df.AT_X_GT[i]-meanX[j])          * (df.AT_X_GT[i+1]        - meanX[k])
                Cov[j, k, 0, 3] += (df.AT_X_GT[i]-meanX[j])          * (df.AirTemperature[i+1] - meanY[k])

                Cov[j, k, 1, 0] += (df.AirTemperature[i]-meanY[j])   * (df.AT_X_GT[i]          - meanX[j])
                Cov[j, k, 1, 1] += (df.AirTemperature[i]-meanY[j])   * (df.AirTemperature[i]   - meanY[j])
                Cov[j, k, 1, 2] += (df.AirTemperature[i]-meanY[j])   * (df.AT_X_GT[i+1]        - meanX[k])
                Cov[j, k, 1, 3] += (df.AirTemperature[i]-meanY[j])   * (df.AirTemperature[i+1] - meanY[k])

                Cov[j, k, 2, 0] += (df.AT_X_GT[i+1]-meanX[k])        * (df.AT_X_GT[i]          - meanX[j])
                Cov[j, k, 2, 1] += (df.AT_X_GT[i+1]-meanX[k])        * (df.AirTemperature[i]   - meanY[j])
                Cov[j, k, 2, 2] += (df.AT_X_GT[i+1]-meanX[k])        * (df.AT_X_GT[i+1]        - meanX[k])
                Cov[j, k, 2, 3] += (df.AT_X_GT[i+1]-meanX[k])        * (df.AirTemperature[i+1] - meanY[k])

                Cov[j, k, 3, 0] += (df.AirTemperature[i+1]-meanY[k]) * (df.AT_X_GT[i]          - meanX[j])
                Cov[j, k, 3, 1] += (df.AirTemperature[i+1]-meanY[k]) * (df.AirTemperature[i]   - meanY[j])
                Cov[j, k, 3, 2] += (df.AirTemperature[i+1]-meanY[k]) * (df.AT_X_GT[i+1]        - meanX[k])
                Cov[j, k, 3, 3] += (df.AirTemperature[i+1]-meanY[k]) * (df.AirTemperature[i+1] - meanY[k])

    for j in range(n_z):
        for k in range(n_z):
            Cov[j, k, :, :] = (Cov[j, k, :, :] + Cov[j, k, :, :].transpose())
            if cpt[j,k] != 0:
                Cov[j, k, :, :] = Cov[j, k, :, :] / (2.*cpt[j,k])
    
    # Au cas ou des matrices sont remplies de 0 --> on recopie une autre
    Cov[0,1,:,:] = Cov[0, 0,:,:].copy()
    Cov[1,0,:,:] = Cov[1, 1,:,:].copy()
    # for j in range(n_z):
    #   for k in range(n_z):
    #       #print(Cov[j,k,:,:].ravel().nonzero()[0])
    #       if Cov[j,k,:,:].ravel().nonzero()[0].size == 0:
                
    return Cov


if __name__ == '__main__': 
    main()

