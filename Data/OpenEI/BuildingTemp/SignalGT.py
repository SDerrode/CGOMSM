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

listJune    = ['building1retail_June_Week_672', 'building2retail_June_Week_672', 'building3retail_June_Week_672', 'building4retail_June_Week_672', 'building5retail_June_Week_672']
listJanuary = ['building1retail_January_Week_672', 'building2retail_January_Week_672', 'building3retail_January_Week_672', 'building4retail_January_Week_672', 'building5retail_January_Week_672']

Prefix='../../../../Data_CGPMSM/OpenEI/BuildingTemp/'

if __name__ == '__main__':

    for name in listJune:
    
        DOE_df = pd.read_csv(Prefix + 'input/' + name + '.csv', parse_dates=[0])
        listeHeader = list(DOE_df)
        pd.to_datetime(DOE_df[listeHeader[0]])
        DOE_df.sort_values(by=[listeHeader[0]])
        DOE_df[listeHeader[1]].astype(float)

        datemin=DOE_df[listeHeader[0]].iloc[0]
        datemax=DOE_df[listeHeader[0]].iloc[-1]
        # input('pause')
        print('Building name : ', name)
        print('  -->Date départ série temporelle = ', datemin)
        print('  -->Date fin    série temporelle = ', datemax)
        DOE_df.set_index(listeHeader[0], inplace=True)

        ################ Ground truth
        sLength = len(DOE_df[listeHeader[1]])
        DOE_df['R_GT'] = pd.Series(np.zeros(sLength), index=DOE_df.index)
        listeHeader = list(DOE_df)
        print(listeHeader)

        # Rfuzzy ground truth
        fuzzygt = []
        dgt = []
        for j in range(7):
            dgt.append(DOE_df.index[ 0*4+j*96]), fuzzygt.append(0.)
            dgt.append(DOE_df.index[ 4*4+j*96]), fuzzygt.append(0.)
            dgt.append(DOE_df.index[12*4+j*96]), fuzzygt.append(1.)
            dgt.append(DOE_df.index[16*4+j*96]), fuzzygt.append(1.)
        dgt.append(DOE_df.index[ -1]), fuzzygt.append(0.)

        for i, date_i in enumerate(dgt[:-1]):
            # fuzzy R GT
            periode_iip1 = DOE_df.loc[dgt[i]:dgt[i+1], 'R_GT']
            rampe_iip1   = np.linspace(fuzzygt[i], fuzzygt[i+1], len(periode_iip1))
            DOE_df.loc[dgt[i]:dgt[i+1], 'R_GT'] = rampe_iip1

        PS.plot3(Prefix + 'figures/' + name + '_GT.png', DOE_df.index, DOE_df[listeHeader[0]], DOE_df[listeHeader[1]], DOE_df[listeHeader[2]], 'Outdoor Air Temperature (F)', 'Power (kW)', 'Fuzzy jumps', sf=True)
        #PS.plot2('toto.png', DOE_df.index, DOE_df[listeHeader[0]], DOE_df[listeHeader[1]],  'Outdoor Air Temperature (F)', 'Power (kW)', sf=True)
        #PS.plot2(None, DOE_df.index, DOE_df[listeHeader[0]], DOE_df[listeHeader[2]], 'Outdoor Air Temperature (F)', 'Fuzzy jumps ground-truth', sf=False)

        # Renommage des colums
        # DOE_df.columns = ['Y', 'X', 'R_GT']
        # DOE_df[listeHeader[0]].astype(float)
        # DOE_df[listeHeader[1]].astype(float)
        DOE_df.to_csv(Prefix + 'input/' + name + '_GT.csv', columns=DOE_df.columns, header=True, index=True)
        