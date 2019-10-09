#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

import PlotSignals as PS


if __name__ == '__main__':

    INSEE=92 # 75 77
    name = 'indices_QA_commune_IDF_2016_2018'
    name1 = name + '_' + str(INSEE)

    # Check out the data
    IQA_excerpt_df = pd.read_csv('input/' + name1 +'.csv', parse_dates=[0], dayfirst=True)
    
    print(IQA_excerpt_df.head(5))
    print(IQA_excerpt_df.columns)
    pd.to_datetime(IQA_excerpt_df['date'])
    IQA_excerpt_df.sort_values(by=['date'])

    #print(IQA_excerpt_df.columns[2])
    print('count=', IQA_excerpt_df.count())
    print('-->Date départ série temporelle = ', IQA_excerpt_df['date'].iloc[0])
    print('-->Date fin    série temporelle = ', IQA_excerpt_df['date'].iloc[-1])

    #IQA_excerpt_df.set_index('date', inplace=True)
    legend1 = 'no2'
    legend2 = 'o3'
    legend3 = 'pm10'
    ch = './figures/' + name1 + '.png'
    PS.plot3(ch, IQA_excerpt_df.date, IQA_excerpt_df.no2, IQA_excerpt_df.o3, IQA_excerpt_df.pm10, \
        legend1=legend1, legend2=legend2, legend3=legend3, sf=True)

    ch = './figures/' + name1 + '_no2.png'
    PS.plot1(ch, IQA_excerpt_df.date, IQA_excerpt_df.no2, legend1=legend1, c='blue', sf=True)
    ch = './figures/' + name1 + '_o3.png'
    PS.plot1(ch, IQA_excerpt_df.date, IQA_excerpt_df.o3, legend1=legend2, c='orange', sf=True)
    ch = './figures/' + name1 + '_pm10.png'
    PS.plot1(ch, IQA_excerpt_df.date, IQA_excerpt_df.pm10, legend1=legend3, c='green', sf=True)

