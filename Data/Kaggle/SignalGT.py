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


if __name__ == '__main__':

    # name = Prefix + 'input/' + listStations[9] + '5209.csv'
    name = Prefix + 'input/' + listStations[9] + '_all_resample_1303.csv'
    temperature_df = pd.read_csv(name, parse_dates=[0])
    
    pd.to_datetime(temperature_df['DateTime'])
    temperature_df.sort_values(by=['DateTime'])
    # for y in temperature_df.columns:
    #   print (temperature_df[y].dtype, end=', ')
    # print('count=', temperature_df.count())
    # print(list(temperature_df))

    print('-->Date départ série temporelle = ', temperature_df['DateTime'].iloc[0])
    print('-->Date fin    série temporelle = ', temperature_df['DateTime'].iloc[-1])
    temperature_df.set_index('DateTime', inplace=True)

    ################ Ground truth
    sLength = len(temperature_df['AirTemperature'])
    temperature_df['AT_X_GT']         = pd.Series(np.zeros(sLength), index=temperature_df.index)
    temperature_df['AT_X_GT_NONOISE'] = pd.Series(np.zeros(sLength), index=temperature_df.index)
    temperature_df['AT_R_GT']         = pd.Series(np.zeros(sLength), index=temperature_df.index)

    # X ground truth
    dgt = []
    ygt = []
    #Pour le fichier ch = 'JoseRizalBridgeNorth_all_resample_15624'
    # dgt.append(temperature_df.index[0]),     ygt.append(52)
    # dgt.append(temperature_df.index[1000]),  ygt.append(69)
    # dgt.append(temperature_df.index[1880]),  ygt.append(69)
    # dgt.append(temperature_df.index[2760]),  ygt.append(45)
    # dgt.append(temperature_df.index[3800]),  ygt.append(45)
    # dgt.append(temperature_df.index[5200]),  ygt.append(72)
    # dgt.append(temperature_df.index[6000]),  ygt.append(72)
    # dgt.append(temperature_df.index[7200]),  ygt.append(42)
    # dgt.append(temperature_df.index[7800]),  ygt.append(42)
    # dgt.append(temperature_df.index[9890]),  ygt.append(68)
    # dgt.append(temperature_df.index[10500]), ygt.append(68)
    # dgt.append(temperature_df.index[12000]), ygt.append(38)
    # dgt.append(temperature_df.index[12250]), ygt.append(38)
    # dgt.append(temperature_df.index[14350]), ygt.append(71)
    # dgt.append(temperature_df.index[14900]), ygt.append(71)
    # dgt.append(temperature_df.index[-1]),    ygt.append(53)

    #Pour le fichier ch = 'JoseRizalBridgeNorth_all_resample_5209'
    # dgt.append(temperature_df.index[0]),     ygt.append(52)
    # dgt.append(temperature_df.index[333]),  ygt.append(69)
    # dgt.append(temperature_df.index[627]),  ygt.append(69)
    # dgt.append(temperature_df.index[int(2760/3)]),  ygt.append(45)
    # dgt.append(temperature_df.index[int(3800/3)]),  ygt.append(45)
    # dgt.append(temperature_df.index[int(5200/3)]),  ygt.append(72)
    # dgt.append(temperature_df.index[int(6000/3)]),  ygt.append(72)
    # dgt.append(temperature_df.index[int(7200/3)]),  ygt.append(42)
    # dgt.append(temperature_df.index[int(7800/3)]),  ygt.append(42)
    # dgt.append(temperature_df.index[int(9890/3)]),  ygt.append(68)
    # dgt.append(temperature_df.index[int(10500/3)]), ygt.append(68)
    # dgt.append(temperature_df.index[3960]), ygt.append(38)
    # dgt.append(temperature_df.index[int(12250/3)]), ygt.append(38)
    # dgt.append(temperature_df.index[int(14350/3)]), ygt.append(71)
    # dgt.append(temperature_df.index[int(14900/3)]), ygt.append(71)
    # dgt.append(temperature_df.index[-1]),    ygt.append(53)

    #Pour le fichier ch = 'JoseRizalBridgeNorth_all_resample_1303'
    dgt.append(temperature_df.index[0]),     ygt.append(52)
    dgt.append(temperature_df.index[int(333/4.)]),  ygt.append(69)
    dgt.append(temperature_df.index[int(627/4.)]),  ygt.append(69)
    dgt.append(temperature_df.index[int(2760/3./4.)]),  ygt.append(45)
    dgt.append(temperature_df.index[int(3800/3./4.)]),  ygt.append(45)
    dgt.append(temperature_df.index[int(5200/3./4.)]),  ygt.append(72)
    dgt.append(temperature_df.index[int(6000/3./4.)]),  ygt.append(72)
    dgt.append(temperature_df.index[int(7200/3./4.)]),  ygt.append(42)
    dgt.append(temperature_df.index[int(7800/3./4.)]),  ygt.append(42)
    dgt.append(temperature_df.index[int(9890/3./4.)]),  ygt.append(68)
    dgt.append(temperature_df.index[int(10500/3./4.)]), ygt.append(68)
    dgt.append(temperature_df.index[int(3960/4.)]), ygt.append(38)
    dgt.append(temperature_df.index[int(12250/3./4.)]), ygt.append(38)
    dgt.append(temperature_df.index[int(14350/3./4.)]), ygt.append(71)
    dgt.append(temperature_df.index[int(14900/3./4.)]), ygt.append(71)
    dgt.append(temperature_df.index[-1]),    ygt.append(53)

    # Rfuzzy ground truth
    fuzzygt = []
    fuzzygt.append((ygt[0]-45)/(ygt[1]-45))
    fuzzygt.append(1.)
    fuzzygt.append(1.)
    fuzzygt.append(0.)
    fuzzygt.append(0.)
    fuzzygt.append(1.)
    fuzzygt.append(1.)
    fuzzygt.append(0.)
    fuzzygt.append(0.)
    fuzzygt.append(1.)
    fuzzygt.append(1.)
    fuzzygt.append(0.)
    fuzzygt.append(0.)
    fuzzygt.append(1.)
    fuzzygt.append(1.)
    fuzzygt.append((ygt[-1]-38)/(ygt[-2]-38))

    # print(fuzzygt)
    # print(ygt)

    for i, date_i in enumerate(dgt[:-1]):
        # X GT
        periode_iip1 = temperature_df.loc[dgt[i]:dgt[i+1], 'AT_X_GT']
        rampe_iip1   = np.linspace(ygt[i], ygt[i+1], len(periode_iip1))
        temperature_df.loc[dgt[i]:dgt[i+1], 'AT_X_GT_NONOISE'] = rampe_iip1
        temperature_df.loc[dgt[i]:dgt[i+1], 'AT_X_GT'] = rampe_iip1 + np.random.normal(0, 0.5, len(rampe_iip1))
        # fuzzy R GT
        periode_iip1 = temperature_df.loc[dgt[i]:dgt[i+1], 'AT_R_GT']
        rampe_iip1   = np.linspace(fuzzygt[i], fuzzygt[i+1], len(periode_iip1))
        temperature_df.loc[dgt[i]:dgt[i+1], 'AT_R_GT'] = rampe_iip1

    name = Prefix + 'input/'+ listStations[9] + '_all_resample_1303_GT.csv'
    temperature_df.to_csv(name, columns=['AirTemperature', 'AT_X_GT', 'AT_X_GT_NONOISE', 'AT_R_GT'], header=True, index=True)

    legend1 = 'Resampled observations'
    legend2 = 'Hand-made state ground truth'
    ch1 = Prefix + 'figures/Process/'+ listStations[9] + '_all_resample_1303_GT.png'
    PS.plot_GT2(None, temperature_df.index, temperature_df.AirTemperature, temperature_df.AT_X_GT_NONOISE, legend1 = legend1, legend2=legend2, sf=False)
    PS.plot_GT2(ch1,  temperature_df.index, temperature_df.AirTemperature, temperature_df.AT_X_GT_NONOISE, legend1 = legend1, legend2=legend2, sf=True)

    # un extrait avec uniquement croissance des températures
    # excerpt = temperature_df[(temperature_df.index >= '2016-12-25 00:00:00') & (temperature_df.index <= '2017-09-01 23:59:59')]
    # legend1 = 'Resampled Observations'
    # legend2 = 'State ground truth'
    # ch1 = Prefix + 'figures/Process/'+ listStations[9] + '_all_resample_1303_GT_excerpt.png'
    # PS.plot_GT2(None, excerpt.index, excerpt.AirTemperature, excerpt.AT_X_GT_NONOISE, legend1 = legend1, legend2=legend2, sf=False)
    # PS.plot_GT2(ch1,  excerpt.index, excerpt.AirTemperature, excerpt.AT_X_GT_NONOISE, legend1 = legend1, legend2=legend2, sf=True)
    # name = Prefix + 'input/'+ listStations[9] + '_all_resample_1303_GT_excerpt.csv'
    # excerpt.to_csv(name, columns=['AirTemperature', 'AT_X_GT', 'AT_X_GT_NONOISE', 'AT_R_GT'], header=True, index=True)
