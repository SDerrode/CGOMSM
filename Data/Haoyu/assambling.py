#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

def main():

    Prefix='../../../Data_CGPMSM/Haoyu/'

    # reading of long lat ground truth data
    name = Prefix + 'Data/ArroundITECH_360m_GroundTruth_interp.txt'
    print('name=', name)
    LongLat_GT_df = pd.read_csv(name, parse_dates=[0])
    print(LongLat_GT_df.head(5))
    for y in LongLat_GT_df.columns:
        print (LongLat_GT_df[y].dtype, end=', ')

    # reading steps data
    name = Prefix + 'Data/step.txt'
    Steps_df = pd.read_csv(name, parse_dates=[0])
    print(Steps_df.head(5))
    for y in Steps_df.columns:
        print (Steps_df[y].dtype, end=', ')

    # reading observation data
    name = Prefix + 'Data/Y.txt'
    Y_df = pd.read_csv(name, parse_dates=[0])
    print(Y_df.head(5))
    for y in Y_df.columns:
        print (Y_df[y].dtype, end=', ')

    # print(LongLat_GT_df.count()[0])
    # print(Steps_df.count()[0])
    # print(Y_df.count()[0])


    allData_df = pd.concat([Y_df, Steps_df, LongLat_GT_df], sort=False, axis=1)
    allData_df.columns = ['AccelX', 'AccelY', 'AccelZ', 'VelX', 'VelY', 'VelZ', 'Steps', 'Long_GT', 'Lat_GT']
    # for y in allData_df.columns:
    #   print (allData_df[y].dtype, end=', ')
    listHeader = list(allData_df)
    print(listHeader)
    print(allData_df.head(20))
    name = Prefix + 'Data/DFBSmoothData.txt'
    allData_df.to_csv(name, header=True, index=False)

if __name__ == '__main__':
    main()