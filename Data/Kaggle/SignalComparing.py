#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import numpy as np
import pandas as pd
import datetime as dt
import sys

import PlotSignals as PS

Prefix='../../../Data_CGPMSM/Kaggle/'

listStations = ['35thAveSW_SWMyrtleSt', 'AlaskanWayViaduct_KingSt', 'AlbroPlaceAirportWay', 'AuroraBridge', 'HarborAveUpperNorthBridge', 'MagnoliaBridge', 'NE45StViaduct', 'RooseveltWay_NE80thSt', 'SpokaneSwingBridge', 'JoseRizalBridgeNorth']

def main():

    n_x,n_y = 1, 1
    n_z     = n_x + n_y
    n_r     = 2

    #steps = '1,3,5,7,9,15' #,7,9'
    steps = '1,3,5,8,12,16,21'
    STEPS = list(map(int, steps.split(',')))

    PH = '0_7555758131155664' # delat*15
    #PH = '0_4789328405106386' # alpha /3

    #ch = '_all_resample_5209_GT.csv'
    #ch = '_all_resample_5209_GT_excerpt.csv'
    #ch   = '_all_resample_1303_GT_excerpt.csv'
    ch   = '_all_resample_1303_GT.csv'
    name = Prefix + 'input/'+ listStations[9] + ch

    ##################################
    # Lecture des données originales
    temperature_df = pd.read_csv(name, parse_dates=[0])
    pd.to_datetime(temperature_df['DateTime'])
    temperature_df.sort_values(by=['DateTime'])
    # for y in temperature_df.columns:
    #   print (temperature_df[y].dtype, end=', ')
    # print('count=', temperature_df.count())
    # print('-->Date départ série temporelle = ', temperature_df['DateTime'].iloc[0])
    # print('-->Date fin    série temporelle = ', temperature_df['DateTime'].iloc[-1])
    temperature_df.set_index('DateTime', inplace=True)
    # listHeader = list(temperature_df)
    # print(listHeader)
    # print(temperature_df.head(5))

    Y = temperature_df.iloc[:,0:1].values
    X_GT_array = temperature_df.iloc[:,2:3].values
    R_GT_array = temperature_df.iloc[:,3:4].values
    #print('MSE X_GT_array, Y')
    MSEYX = MSE(X_GT_array, Y)

    name1 = Prefix + 'figures/'+ listStations[9] + '_all_resample_1303_GT_orig.png'
    PS.plot_GT2(name1, temperature_df.index, Y, X_GT_array, legend1='Resampled observations', legend2='Hand-made state ground-truth', sf=True)

    ##################################
    # Lecture de X et R filtrés et calcul MSE
    
    MSE_X = np.zeros((len(STEPS)+1)) # +1 pour mettre le resultat hard
    MSE_R = np.zeros((len(STEPS)+1))

    # Modèle dur************************************************
    ch1   = 'FS4_pH_' + PH + '_Temper_FILT_HARD_'
    ch2XY = 'XY_CGOFMSM_restored.csv'
    ch2R  = 'R_CGOFMSM_restored.csv'

    chXY = ch1 + ch2XY
    Xlu = np.loadtxt('../../Result/Result_csv/'+chXY)
    Xest = Xlu.reshape((len(Xlu),1))

    MSE_X[0] = MSE(X_GT_array, Xest)

    name1 = Prefix + 'figures/'+ listStations[9] + '_all_resample_1303_GT_HARD_X.png'
    PS.plot_GT3(name1, 'hard filter', temperature_df.index, Y, X_GT_array, Xest, legend1='Resampled observations', legend2='Hand-made state ground-truth', legend3='Estimated states (hard model)', sf=True)

    chR = ch1 + ch2R
    Rlu = np.loadtxt('../../Result/Result_csv/'+chR)
    Rest = Rlu.reshape((len(Rlu),1))
    MSE_R[0] = MSE(R_GT_array, Rest)

    name1 = Prefix + 'figures/'+ listStations[9] + '_all_resample_1303_GT_HARD_R.png'
    PS.plot_GT2(name1, temperature_df.index, R_GT_array, Rest, legend1='Hand-made fuzzy-jump ground-truth', legend2='Estimated jumps (hard model)', sf=True)

    # Modèle flous**********************************************
    ch1   = 'FS4_pH_' + PH + '_Temper_FILT_FUZZY_STEP_'
    ch2XY = '_XY_CGOFMSM_restored.csv'
    ch2R  = '_R_CGOFMSM_restored.csv'
    

    for i,s in enumerate(STEPS):
        chXY = ch1 + str(s) + ch2XY
        Xlu = np.loadtxt('../../Result/Result_csv/'+chXY)
        Xest = Xlu.reshape((len(Xlu),1))
        #print(np.shape(X_GT_array[10:15]))
        #print(np.shape(Xest[10:15]))
        #print('MSE X_GT_array, Xest')

        MSE_X[i+1] = MSE(X_GT_array, Xest)

        name1 = Prefix + 'figures/'+ listStations[9] + '_all_resample_1303_GT_step' + str(s) + '_X.png'
        PS.plot_GT3(name1, ' fuzzy filter (F=' + str(s) + ')', temperature_df.index, Y, X_GT_array, Xest, legend1='Resampled observations', legend2='Hand-made state ground-truth', legend3='Estimated states (fuzzy model)', sf=True)

        chR = ch1 + str(s) + ch2R
        Rlu = np.loadtxt('../../Result/Result_csv/'+chR)
        Rest = Rlu.reshape((len(Rlu),1))
        MSE_R[i+1] = MSE(R_GT_array, Rest)

        name1 = Prefix + 'figures/'+ listStations[9] + '_all_resample_1303_GT_step' + str(s) + '_R.png'
        PS.plot_GT2(name1, temperature_df.index, R_GT_array, Rest, legend1='Hand-made fuzzy-jump ground-truth', legend2='Estimated jumps (fuzzy model)', sf=True)

    print('MSE entre Y et X = ', MSEYX)
    print('Résultats de filtrage pour differentes valeurs de F (X): ', MSE_X)
    print('Résultats de filtrage pour differentes valeurs de F (R): ', MSE_R)


def MSE(Est, real):
    """MSE of estimated X"""
    SP_mse_sum = np.mean((Est[3:-3]-real[3:-3])**2)
    #SP_mse_sum = np.mean((Est-real)**2)
    return SP_mse_sum

if __name__ == '__main__': 
    main()

