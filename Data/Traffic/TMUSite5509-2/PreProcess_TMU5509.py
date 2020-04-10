#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import numpy as np
import pandas as pd
import datetime as dt
import sys
import pathlib

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as md

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


if __name__ == '__main__':

    # Données à partir de : http://tris.highwaysengland.co.uk/detail/trafficflowdata

    sensor  = 'TMUSite5509-2'
    filename = sensor+'_DailyReport.csv' # Lien de téléchargement : http://tris.highwaysengland.co.uk/download/cb768ab7-ecad-457a-8d9c-4ab9fd5147ec
    Plot     = True

    # print(' . filename =', filename)
    # print('      ', pathlib.Path(filename).stem)
    # print('      ', pathlib.Path(filename).suffix)
    # print('      ', pathlib.Path(filename).name)
    # print(' . Plot     =', Plot)
    # print('\n')

    # Lecture des données
    df = pd.read_csv(filename, header=2, skipinitialspace=True)
    # listeHeader = list(df)
    # print('Entête des columns : ', listeHeader)

    # La base de temps
    df['Timestamp'] = pd.to_datetime(df['Local Date'] + ' ' + df['Local Time'])
    df1 = pd.DataFrame() #creates a new dataframe that's empty
    df1['Timestamp'] = df['Timestamp'].copy()
    #df1 = df1.set_index(pd.DatetimeIndex(df1['Timestamp']), inplace=True)
    
    # La vitesse (représente Y)
    df1[sensor + ' - Speed Value'] = df['Speed Value'].copy()
    # Le traffic flow (représente X))
    df1[sensor + ' - Total Carriageway Flow'] = df['Total Carriageway Flow'].copy()

    print(df1.head())
    print(df1.index.name)
    datemin = df1['Timestamp'].iloc[0]
    datemax = df1['Timestamp'].iloc[-1]
    print('  -->Date départ série temporelle = ', datemin)
    print('  -->Date fin    série temporelle = ', datemax)

    # saves on disk #################################################
    # ALL the data
    name=sensor+'_all.csv'
    df1.to_csv(name, header=True, index=False)
    
    # train
    name=sensor+'_train.csv'
    Train = df1[(df1['Timestamp'] >= '2018-01-01') & (df1['Timestamp'] < '2018-01-30')]
    Train.to_csv(name, header=True, index=False)
    name=sensor+'_trainY.txt'
    Train[sensor + ' - Speed Value'].to_csv(name, header=False, index=False)
    name=sensor+'_trainX.txt'
    Train[sensor + ' - Total Carriageway Flow'].to_csv(name, header=False, index=False)

    name=sensor+'_train_300.csv'
    Train300 = df1[(df1['Timestamp'] >= '2018-01-01') & (df1['Timestamp'] <= '2018-01-04 02:59:00')]
    Train300.to_csv(name, header=True, index=False)
    name=sensor+'_trainY_300.txt'
    Train300[sensor + ' - Speed Value'].to_csv(name, header=False, index=False)
    name=sensor+'_trainX_300.txt'
    Train300[sensor + ' - Total Carriageway Flow'].to_csv(name, header=False, index=False)


    # test
    name=sensor+'_test.csv'
    Test = df1[(df1['Timestamp'] >= '2018-01-30')]
    Test.to_csv(name, header=True, index=False)
    name=sensor+'_testY.txt'
    Train[sensor + ' - Speed Value'].to_csv(name, header=False, index=False)
    name=sensor+'_testX.txt'
    Train[sensor + ' - Total Carriageway Flow'].to_csv(name, header=False, index=False)
    # test
    
    listeHeader = list(df1)
    print('Entête des columns : ', listeHeader)

