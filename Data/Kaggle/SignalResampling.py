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

    name = Prefix + 'input/' + listStations[9] + '.csv'
    temperature_df = pd.read_csv(name, parse_dates=[0])
    
    pd.to_datetime(temperature_df['DateTime'])
    temperature_df.sort_values(by=['DateTime'])
    # for y in temperature_df.columns:
    #   print (temperature_df[y].dtype, end=', ')
    # print('count=', temperature_df.count())
    # print(list(temperature_df))
    print('-->Date départ série temporelle = ', temperature_df['DateTime'].iloc[0])
    print('-->Date fin    série temporelle = ', temperature_df['DateTime'].iloc[-1])
    #print(temperature_df.head(10))

    # LA TOTALE
    excerpt = temperature_df[(temperature_df['DateTime'] >= temperature_df['DateTime'].iloc[0]) & (temperature_df['DateTime'] <= temperature_df['DateTime'].iloc[-1])]
    ch, resampling_freq  = 'all', '1D' # '6H'

    # # LES 3 ANNEES
    # excerpt = temperature_df[(temperature_df['DateTime'] >= '2014-07-1 00:00:00') & (temperature_df['DateTime'] <= '2015-06-30 23:59:59')]
    # ch, resampling_freq = 'year2014', 'M'
    # excerpt = temperature_df[(temperature_df['DateTime'] >= '2015-07-1 00:00:00') & (temperature_df['DateTime'] <= '2016-06-30 23:59:59')]
    # ch, resampling_freq  = 'year2015', 'M'
    # excerpt = temperature_df[(temperature_df['DateTime'] >= '2016-07-1 00:00:00') & (temperature_df['DateTime'] <= '2017-06-30 23:59:59')]
    # ch, resampling_freq  = 'year2016', 'M'

    # # LES 4 MOIS de juin
    # excerpt = temperature_df[(temperature_df['DateTime'] >= '2014-06-1 00:00:00') & (temperature_df['DateTime'] <= '2014-06-30 23:59:59')]
    # ch, resampling_freq = 'June2014', 'D'
    # excerpt = temperature_df[(temperature_df['DateTime'] >= '2015-06-1 00:00:00') & (temperature_df['DateTime'] <= '2015-06-30 23:59:59')]
    # ch, resampling_freq = 'June2015', 'D'
    # excerpt = temperature_df[(temperature_df['DateTime'] >= '2016-06-1 00:00:00') & (temperature_df['DateTime'] <= '2016-06-30 23:59:59')]
    # ch, resampling_freq = 'June2016', 'D'
    # excerpt = temperature_df[(temperature_df['DateTime'] >= '2017-06-1 00:00:00') & (temperature_df['DateTime'] <= '2017-06-30 23:59:59')]
    # ch, resampling_freq = 'June2017', 'D'

    # # 4 PREMIERES SEMAINES DE JUIN
    # excerpt = temperature_df[(temperature_df['DateTime'] >= '2014-06-1 00:00:00') & (temperature_df['DateTime'] <= '2014-06-07 23:59:59')]
    # ch, resampling_freq = SemaineJune2014', 'H'
    # excerpt = temperature_df[(temperature_df['DateTime'] >= '2015-06-1 00:00:00') & (temperature_df['DateTime'] <= '2015-06-07 23:59:59')]
    # ch, resampling_freq = SemaineJune2015', 'H'
    # excerpt = temperature_df[(temperature_df['DateTime'] >= '2016-06-1 00:00:00') & (temperature_df['DateTime'] <= '2016-06-07 23:59:59')]
    # ch, resampling_freq = 'SemaineJune2016', 'H'
    # excerpt = temperature_df[(temperature_df['DateTime'] >= '2017-06-1 00:00:00') & (temperature_df['DateTime'] <= '2017-06-07 23:59:59')]
    # ch, resampling_freq = SemaineJune2017', 'H'

    # plot of the excerpt data
    name = Prefix + 'input/'+ listStations[9] +'_' + ch + '_' + str(excerpt['AirTemperature'].count()) + '.csv'
    excerpt.set_index('DateTime', inplace=True)
    excerpt.to_csv(name, columns=['AirTemperature'], header=True, index=True)
    ch1 = Prefix + 'figures/Process/' + listStations[9] + '_'+ ch + '_' + str(excerpt['AirTemperature'].count()) + '_orig.png'
    PS.plot(ch1, excerpt.index, excerpt.AirTemperature, label='Original observations')
    PS.plot(ch1, excerpt.index, excerpt.AirTemperature, label='Original observations', sf=False)

    # Down-sampling and interpolation
    e_resample = excerpt.resample(resampling_freq).mean()
    #print('Avant interpolate=', e_resample.AirTemperature[2209])
    e_resample.interpolate(method='linear', inplace=True)
    #print('Apres interpolate=', e_resample.AirTemperature[2209], label='Observations')
    #input('pause')
    name = Prefix + 'input/'+ listStations[9] +'_' + ch + '_resample_' + str(e_resample['AirTemperature'].count()) + '.csv'
    e_resample.to_csv(name, columns=['AirTemperature'], header=True, index=True)
    ch1 = Prefix + 'figures/Process/' + listStations[9] + '_'+ ch + '_' + str(e_resample['AirTemperature'].count()) + '.png'
    PS.plot(ch1, e_resample.index, e_resample.AirTemperature, label='Resampled observations (1 day)')
    PS.plot(ch1, e_resample.index, e_resample.AirTemperature, label='Resampled observations (1 day)', sf=False)



