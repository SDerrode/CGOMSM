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

Prefix = '../../../Data_CGPMSM/Kaggle/'

#listStations = ['35thAveSW_SWMyrtleSt', 'AlaskanWayViaduct_KingSt', 'AlbroPlaceAirportWay', 'AuroraBridge', 'HarborAveUpperNorthBridge', 'MagnoliaBridge', 'NE45StViaduct', 'RooseveltWay_NE80thSt', 'SpokaneSwingBridge', 'JoseRizalBridgeNorth']
listStations = ['JoseRizalBridgeNorth']

listTemp = ['RSTemp', 'ATemp']

if __name__ == '__main__':

    print(listStations)

    for stationname in listStations:

        temperature_df = pd.read_csv(Prefix + 'input/'+stationname+'.csv', parse_dates=[0])
        # for y in temperature_df.columns:
        #   print (temperature_df[y].dtype, end=', ')
        #   print (temperature_df[y], end=' -- ')
        #print(temperature_df.head(5))
        print('count=', temperature_df.count())

        #Road-Surface Temperature
        plt.plot(temperature_df.DateTime, temperature_df.RoadSurfaceTemperature)
        ax = plt.gca()
        plt.xticks(rotation=45)
        plt.savefig(Prefix + 'figures/' + stationname+'_RSTemperature.png', bbox_inches='tight')
        plt.close()

        #Air Temperature
        plt.plot(temperature_df.DateTime, temperature_df.AirTemperature)
        ax = plt.gca()
        plt.xticks(rotation=45)
        plt.savefig(Prefix + 'figures/' + stationname+'_ATemperature.png', bbox_inches='tight')
        plt.close()

