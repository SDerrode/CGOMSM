#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import datetime as dt

import PlotSignals as PS

def get_Nation_df(name, sensorName):

    # Check out the data
    Nation_df = pd.read_csv('input/' + name + '.csv', sep=';')
    Nation_df['Timestamp'] = Nation_df['Timestamp'].astype('datetime64[s]')
    print('count=', Nation_df.count())
    # print(Nation_df.head(5))
    # print(Nation_df.columns)
    #input('pause')

    Nation_exerpt_df = Nation_df.loc[Nation_df['Host'] == sensorName]
    #pd.to_datetime(Nation_exerpt_df['Timestamp'], infer_datetime_format=True, errors='coerce')

    Nation_exerpt_df.set_index('Timestamp', inplace=True)
    Nation_exerpt_df_SORT = Nation_exerpt_df.sort_index()

    #HostNames = Nation_df.Host.unique()
    #print(HostNames)

    print('count=', Nation_exerpt_df_SORT.count())
    # print(Nation_exerpt_df_SORT.head(5))
    # print(Nation_exerpt_df_SORT.columns)
    # print(Nation_exerpt_df_SORT.index.name)
    #print(Nation_exerpt_df_SORT.index.is_all_dates)
    # print(Nation_exerpt_df_SORT.index[0], type(Nation_exerpt_df_SORT.index[0]))
    # print(Nation_exerpt_df_SORT.index[1], type(Nation_exerpt_df_SORT.index[1]))
    # print(Nation_exerpt_df_SORT.index[2], type(Nation_exerpt_df_SORT.index[2]))
    # print(Nation_exerpt_df_SORT.index[3], type(Nation_exerpt_df_SORT.index[3]))
    # input('pause')

    return Nation_exerpt_df_SORT

def excerpt_data(df, datemin, datemax):

    print('-->Date départ série temporelle = ', df.index[0])
    print('-->Date fin    série temporelle = ', df.index[-1])
    
    # Extraction
    excerpt = df[(df.index >= datemin) & (df.index <= datemax)]
    print('count=', excerpt.count())
    # ch, resampling_freq  = 'all', '1D' # '6H'
    # e_resample = excerpt.resample(resampling_freq).mean()
    # #print('Avant interpolate=', e_resample.AirTemperature[2209])
    # e_resample.interpolate(method='linear', inplace=True)

    return excerpt

if __name__ == '__main__':

    datemin=dt.datetime.strptime('2016-11-01 00:00:00','%Y-%m-%d %H:%M:%S')
    datemax=dt.datetime.strptime('2016-11-15 23:59:59','%Y-%m-%d %H:%M:%S')

    fileNameList = []
    sensorNameList= []
    legendList = []
    fileNameList.append('place-de-la-nation-temperature')
    fileNameList.append('place-de-la-nation-bruit-valeur-moyenne')
    fileNameList.append('place-de-la-nation-pression0')
    sensorNameList.append('CISCO_PARIS_001')
    sensorNameList.append('NATION-CISCO-P11')
    sensorNameList.append('CISCO_PARIS_001')
    legendList.append('Temperature')
    legendList.append('Bruit moyen')
    legendList.append('Pression')
    print(sensorNameList)
    print(fileNameList)
    print(legendList)

    # # TEMPERTAURES *****************************
    # Nation_df_temperature = get_Nation_df(fileNameList[0], sensorNameList[0])
    # Nation_excerpt_df_temperature = excerpt_data(Nation_df_temperature, datemin, datemax)

    # name1 = fileNameList[0] + '_' + sensorNameList[0]
    # Nation_excerpt_df_temperature.to_csv('input/' + name1 +'.csv', columns=['Value'], header=True, index=True)
    # ch = './figures/' + fileNameList[0] + '.png'
    # PS.plot1(ch, Nation_excerpt_df_temperature.index, Nation_excerpt_df_temperature.Value, legend1=legendList[0], c='blue',   sf=True)

    # # BRUIT MOYEN *****************************
    # Nation_df_bruit  = get_Nation_df(fileNameList[1], sensorNameList[1])
    # Nation_excerpt_df_bruit = excerpt_data(Nation_df_bruit, datemin, datemax)

    # name1 = fileNameList[1] + '_' + sensorNameList[1]
    # Nation_excerpt_df_bruit.to_csv('input/' + name1 +'.csv', columns=['Value'], header=True, index=True)
    # ch = './figures/' + fileNameList[1] + '.png'
    # PS.plot1(ch, Nation_excerpt_df_bruit.index,       Nation_excerpt_df_bruit.Value,       legend1=legendList[1], c='orange', sf=True)

    # PRESSION EXTERIEUR *****************************
    Nation_df_pression  = get_Nation_df(fileNameList[2], sensorNameList[2])
    Nation_excerpt_df_pression = excerpt_data(Nation_df_pression, datemin, datemax)

    name1 = fileNameList[2] + '_' + sensorNameList[2]
    Nation_excerpt_df_pression.to_csv('input/' + name1 +'.csv', columns=['Value'], header=True, index=True)
    ch = './figures/' + fileNameList[2] + '.png'
    PS.plot1(ch, Nation_excerpt_df_pression.index,       Nation_excerpt_df_pression.Value,       legend1=legendList[2], c='orange', sf=True)
