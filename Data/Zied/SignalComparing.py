#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import numpy as np
import pandas as pd
import datetime as dt
import sys

import PlotSignals as PS

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

PrefixPlot = './'
name = 'Zied'

def main():

    steps   = '1,2,3,4,5,7,9'
    steps   = '1,5'
    STEPS = list(map(int, steps.split(',')))
    PH = '0_7584'
    FS = 'FS2'

    ##################################
    # Lecture des données originales
    DOE_df = pd.read_csv('./results/XYR.csv', parse_dates=[0])
    pd.to_datetime(DOE_df['Timestamp'])
    DOE_df.sort_values(by=['Timestamp'])

    datemin=DOE_df['Timestamp'].iloc[0]
    datemax=DOE_df['Timestamp'].iloc[-1]
    # input('pause')
    print('  -->Date départ série temporelle = ', datemin)
    print('  -->Date fin    série temporelle = ', datemax)
    DOE_df.set_index('Timestamp', inplace=True)

    listeHeader = list(DOE_df)
    print(listeHeader)
    #input('pause')

    Y    = DOE_df['Y'].values
    X    = DOE_df['X'].values
    R_GT = DOE_df['R_GT'].values
    
    print('MSE(X, Y)')
    MSEYX = MSE(X, Y)

    # name1 = PrefixPlot + 'figures/'+ name + '_GT_orig.png'
    # PS.plot_GT2(name1, '', DOE_df.index, Y, X, legend1='Observed power (kW)', legend2='Outdoor air temperature (True)', sf=True)

    ##################################
    # Lecture de X et R filtrés et calcul MSE
    MSE_X = np.zeros((len(STEPS)+1)) # +1 pour mettre le resultat hard
    MSE_R = np.zeros((len(STEPS)+1))

    # Modèle dur************************************************
    ch1   = FS + '_pH_' + PH + '_Temper_FILT_HARD_'
    ch2XY = 'XY_CGOFMSM_restored.csv'
    ch2R  = 'R_CGOFMSM_restored.csv'

    chXY = ch1 + ch2XY
    Xest = np.loadtxt('../../Result/Fuzzy/Result_csv/'+chXY)
    #Xest = Xlu.reshape((len(Xlu),1))

    print('MSE(X, Xest) pour dur')
    MSE_X[0] = MSE(X, Xest)

    name1 = PrefixPlot + 'figures/'+ name + '_GT_HARD_X.png'
    #PS.plot_GT3(name1, 'Estimation of temperatures for hard filter', DOE_df.index, Y, X, Xest, legend1='Observed power (kW)', legend2='Outdoor air temperature (True)', legend3='Estimated Outdoor air temperature (hard model)', sf=True)
    PS.plot_GT2(name1, 'Estimation of temperatures for hard filter', DOE_df.index, X, Xest, legend1='Outdoor air temperature (True)', legend2='Estimated Outdoor air temperature (hard model)', sf=True)

    chR = ch1 + ch2R
    Rest = np.loadtxt('../../Result/Fuzzy/Result_csv/'+chR)
    #Rest = Rlu.reshape((len(Rlu),1))
    print('MSE(R_GT, Rest) pour dur')
    MSE_R[0] = MSE(R_GT, Rest)

    name1 = PrefixPlot + 'figures/'+ name + '_GT_HARD_R.png'
    PS.plot_GT2(name1, 'Estimation of jumps for hard filter', DOE_df.index, R_GT, Rest, legend1='Hand-made fuzzy-jump ground-truth', legend2='Estimated fuzzy jumps (hard model)', sf=True)

    # Modèle flous**********************************************
    ch1   = 'FS2_pH_' + PH + '_Temper_FILT_FUZZY_STEP_'
    ch2XY = '_XY_CGOFMSM_restored.csv'
    ch2R  = '_R_CGOFMSM_restored.csv'

    for i,s in enumerate(STEPS):
        chXY = ch1 + str(s) + ch2XY
        Xest = np.loadtxt('../../Result/Fuzzy/Result_csv/'+chXY)
        print('MSE(X, Xest)')
        MSE_X[i+1] = MSE(X, Xest)

        name1 = PrefixPlot + 'figures/'+ name + '_GT_step' + str(s) + '_X.png'
        #PS.plot_GT3(name1, 'Estimation of temperatures for fuzzy filter (F=' + str(s) + ')', DOE_df.index, Y, X, Xest, legend1='Observed power (kW)', legend2='Outdoor air temperature (True)', legend3='Estimated Outdoor air temperature (fuzzy model)', sf=True)
        PS.plot_GT2(name1, 'Estimation of temperatures for fuzzy filter (F=' + str(s) + ')', DOE_df.index, X, Xest, legend1='Outdoor air temperature (True)', legend2='Estimated Outdoor air temperature (fuzzy model)', sf=True)

        chR = ch1 + str(s) + ch2R
        Rest = np.loadtxt('../../Result/Fuzzy/Result_csv/'+chR)
        print('MSE(R_GT, Rest)')
        MSE_R[i+1] = MSE(R_GT, Rest)

        name1 = PrefixPlot + 'figures/'+ name + '_GT_step' + str(s) + '_R.png'
        PS.plot_GT2(name1, 'Estimation of jumps for fuzzy filter (F=' + str(s) + ')', DOE_df.index, R_GT, Rest, legend1='Hand-made fuzzy-jump ground-truth', legend2='Estimated fuzzy jumps (fuzzy model)', sf=True)

    print('MSE entre Y et X = ', MSEYX)
    print('Résultats de filtrage pour differentes valeurs de F (X): ', MSE_X, sep='\t')
    print('Résultats de filtrage pour differentes valeurs de F (R): ', MSE_R, sep='\t')


def MSE(Est, real):
    """MSE of estimated X"""
    if Est.shape != real.shape:
        print('size(Est)=', Est.shape)
        print('size(real)=', real.shape)
        print(Est[53:58])
        print(real[53:58])
        input('pause')
    #SP_mse_sum = np.mean((Est[3:-3]-real[3:-3])**2)
    SP_mse_sum = np.mean((Est-real)**2)
    return SP_mse_sum

if __name__ == '__main__': 
    main()

