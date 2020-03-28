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

    #filename = './Data/Traffic/TMU5509/DailyStandard_Report_1_7524_01_01_2018_31_01_2018.csv'
    #filename = './DailyStandard_Report_1_7524_01_01_2018_31_01_2018.csv'
    filename = 'TMUSite5509-2_DailyReport.csv' # Lien de téléchargement : http://tris.highwaysengland.co.uk/download/cb768ab7-ecad-457a-8d9c-4ab9fd5147ec
    prefix   = './generated/TMU5509'
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
    df1['Y'] = df['Speed Value'].copy()
    # Le traffic flow (représente X))
    df1['Xtrue'] = df['Total Carriageway Flow'].copy()

    print(df1.head())
    print(df1.index.name)
    datemin = df1['Timestamp'].iloc[0]
    datemax = df1['Timestamp'].iloc[-1]
    print('  -->Date départ série temporelle = ', datemin)
    print('  -->Date fin    série temporelle = ', datemax)

    # saves on disk #################################################
    # ALL the data
    name=prefix+'_all.csv'
    df1.to_csv(name, header=True, index=False)
    
    # train
    name=prefix+'_train.csv'
    Train = df1[(df1['Timestamp'] >= '2018-01-01') & (df1['Timestamp'] < '2018-01-30')]
    Train.to_csv(name, header=True, index=False)
    name=prefix+'_trainY.txt'
    Train['Y'].to_csv(name, header=False, index=False)
    name=prefix+'_trainX.txt'
    Train['Xtrue'].to_csv(name, header=False, index=False)

    name=prefix+'_train_300.csv'
    Train300 = df1[(df1['Timestamp'] >= '2018-01-01') & (df1['Timestamp'] <= '2018-01-04 02:59:00')]
    Train300.to_csv(name, header=True, index=False)
    name=prefix+'_trainY_300.txt'
    Train300['Y'].to_csv(name, header=False, index=False)
    name=prefix+'_trainX_300.txt'
    Train300['Xtrue'].to_csv(name, header=False, index=False)


    # test
    name=prefix+'_test.csv'
    Test = df1[(df1['Timestamp'] >= '2018-01-30')]
    Test.to_csv(name, header=True, index=False)
    name=prefix+'_testY.txt'
    Train['Y'].to_csv(name, header=False, index=False)
    name=prefix+'_testX.txt'
    Train['Xtrue'].to_csv(name, header=False, index=False)
    # test
    
    listeHeader = list(df1)
    print('Entête des columns : ', listeHeader)

