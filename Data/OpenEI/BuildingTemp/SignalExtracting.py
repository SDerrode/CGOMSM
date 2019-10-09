#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import numpy as np
import pandas as pd
import datetime as dt
import sys

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as md
from matplotlib import pyplot

import matplotlib.dates as md
years    = md.YearLocator()   # every year
months   = md.MonthLocator()  # every month
days     = md.DayLocator()    # every month
yearsFmt = md.DateFormatter('%Y      ')
monthFmt = md.DateFormatter('%B  ') # mois en toute lettre
dayFmt   = md.DateFormatter('%d')

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

import PlotSignals as PS

Prefix='../../../../Data_CGPMSM/OpenEI/BuildingTemp/'

listBuildingNames=['building1retail', 'building2retail', 'building3retail', 'building4retail', 'building5retail']
#listBuildingNames2=['building60preoffice', 'building61duringoffice', 'building62postoffice', 'building70preoffice', 'building71duringoffice', 'building72postoffice']
#listBuildingNames=['building1retail']

def main():

    for buildingName in listBuildingNames:
        # dateminexecerpt, datemaxexecerpt, ch = None, None, 'all'                                             # pour toutes les données
        # dateminexecerpt, datemaxexecerpt, ch = '2010-06-01 00:00:00', '2010-06-30 23:59:59', 'June_Month'    # pour le mois de Juin
        # dateminexecerpt, datemaxexecerpt, ch = '2010-01-01 00:00:00', '2010-01-30 23:59:59', 'January_Month' # pour le mois de Janvier
        #dateminexecerpt, datemaxexecerpt, ch = '2010-01-01 01:00:00', '2010-01-08 00:59:59', 'January_Week'  # pour une semaine en Janvier
        # dateminexecerpt, datemaxexecerpt, ch = '2010-06-01 01:00:00', '2010-06-08 00:59:59', 'June_Week'     # pour une semaine en Janvier
        dateminexecerpt, datemaxexecerpt, ch = '2010-06-01 00:00:00', '2010-06-07 23:59:59', 'June_Week'     # pour une semaine en Janvier

        excerpt, listeHeader = excerptData(buildingName, ch, dateminexecerpt, datemaxexecerpt)

        # plot of the excerpt data
        name1 = Prefix + 'input/'+ buildingName +'_' + ch + '_' + str(excerpt[listeHeader[1]].count()) + '.csv'
        excerpt.set_index(listeHeader[0], inplace=True)
        excerpt.to_csv(name1, columns=[listeHeader[1], listeHeader[2]], header=True, index=True)
        name2 = Prefix + 'figures/' + buildingName + '_'+ ch + '_' + str(excerpt[listeHeader[1]].count()) + '_orig.png'
        PS.plot2(name2, excerpt.index, excerpt[listeHeader[1]], excerpt[listeHeader[2]], 'Outdoor Air Temperature (F)', 'Power (kW)', sf=True)

def excerptData(name, ch, dateminexecerpt, datemaxexecerpt):

    temperature_df = pd.read_csv(Prefix + 'input/' + name + '.csv', parse_dates=[0])
    listeHeader = list(temperature_df)
    pd.to_datetime(temperature_df[listeHeader[0]])
    temperature_df.sort_values(by=[listeHeader[0]])

    datemin=temperature_df[listeHeader[0]].iloc[0]
    datemax=temperature_df[listeHeader[0]].iloc[-1]
    # input('pause')
    print('Building name : ', name)
    print('  -->Date départ série temporelle = ', datemin)
    print('  -->Date fin    série temporelle = ', datemax)

    if dateminexecerpt == None:
        dateminexecerpt = datemin
    if datemaxexecerpt == None:
        datemaxexecerpt = datemax
    excerpt = temperature_df[(temperature_df[listeHeader[0]] >= dateminexecerpt) & (temperature_df[listeHeader[0]] <= datemaxexecerpt)]

    return excerpt, listeHeader

if __name__ == '__main__':
    main()
